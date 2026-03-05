import math
import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup


class _WarmupThenScheduler:
    """
    Simple wrapper that uses a warmup schedule for the first `warmup_steps`,
    then switches to a target scheduler (e.g., CosineAnnealingWarmRestarts).
    Provides a minimal interface: step(), state_dict(), load_state_dict(), get_last_lr().
    """

    def __init__(self, optimizer, warmup_steps: int, after_scheduler_ctor, after_kwargs: dict):
        self.optimizer = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self._warmup = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps
        ) if self.warmup_steps > 0 else None
        self._after = after_scheduler_ctor(optimizer, **after_kwargs)
        self.last_epoch = -1

    def _step_warmup_to_epoch(self, target_epoch: int):
        if self._warmup is None:
            return
        warmup_target = min(target_epoch, self.warmup_steps - 1)
        current = getattr(self._warmup, "last_epoch", -1)
        while current < warmup_target:
            self._warmup.step()
            current = getattr(self._warmup, "last_epoch", current + 1)

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)

        if self._warmup is not None and self.last_epoch < self.warmup_steps:
            if epoch is None:
                self._warmup.step()
            else:
                self._step_warmup_to_epoch(self.last_epoch)
        else:
            if epoch is None:
                self._after.step()
            else:
                after_epoch = max(0, self.last_epoch - self.warmup_steps)
                self._after.step(after_epoch)

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "warmup_steps": self.warmup_steps,
            "warmup": self._warmup.state_dict() if self._warmup is not None else None,
            "after": self._after.state_dict(),
        }

    def load_state_dict(self, state):
        self.last_epoch = state.get("last_epoch", self.last_epoch)
        if self._warmup is not None and state.get("warmup") is not None:
            self._warmup.load_state_dict(state["warmup"])
        if state.get("after") is not None:
            self._after.load_state_dict(state["after"])

    def get_last_lr(self):
        try:
            if self._warmup is not None and self.last_epoch < self.warmup_steps:
                return self._warmup.get_last_lr()
            return self._after.get_last_lr()
        except Exception:
            return [group.get("lr", None) for group in self.optimizer.param_groups]


class _WarmupThenPhasedCosineScheduler:
    """
    Piecewise cosine scheduler with a single warmup phase at the start.
    Each phase defines: {"steps": int, "start_lr": float, "end_lr": float}.
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_steps: int,
            phases: list,
            warmup_init_lr: float = 0.0,
            reference_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = int(max(0, warmup_steps))
        self.warmup_init_lr = float(max(0.0, warmup_init_lr))
        self.phases = self._normalize_phases(phases)
        self.last_epoch = -1

        self._base_lrs = []
        for group in self.optimizer.param_groups:
            base_lr = float(group.get("initial_lr", group["lr"]))
            group["initial_lr"] = base_lr
            self._base_lrs.append(base_lr)

        if reference_lr is None:
            reference_lr = max(self._base_lrs) if len(self._base_lrs) > 0 else 1.0
        self.reference_lr = float(reference_lr) if float(reference_lr) > 0 else 1.0
        self._last_lrs = [group["lr"] for group in self.optimizer.param_groups]

    @staticmethod
    def _normalize_phases(phases: list):
        if not isinstance(phases, list) or len(phases) == 0:
            raise ValueError("For phased cosine scheduler, `phases` must be a non-empty list.")

        normalized = []
        previous_end_lr = None
        for idx, phase in enumerate(phases):
            if not isinstance(phase, dict):
                raise ValueError(f"Phase at index {idx} must be an object/dict.")

            steps = phase.get("steps", phase.get("num_steps", phase.get("total_iters")))
            if steps is None:
                raise ValueError(f"Phase at index {idx} is missing `steps`.")
            steps = int(steps)
            if steps <= 0:
                raise ValueError(f"Phase at index {idx} has non-positive `steps`: {steps}.")

            if "start_lr" in phase:
                start_lr = float(phase["start_lr"])
            elif "max_lr" in phase:
                start_lr = float(phase["max_lr"])
            elif previous_end_lr is not None:
                start_lr = previous_end_lr
            else:
                raise ValueError(
                    f"Phase at index {idx} is missing `start_lr`/`max_lr` and no previous phase exists."
                )

            if "end_lr" in phase:
                end_lr = float(phase["end_lr"])
            elif "min_lr" in phase:
                end_lr = float(phase["min_lr"])
            elif "eta_min" in phase:
                end_lr = float(phase["eta_min"])
            else:
                raise ValueError(f"Phase at index {idx} is missing `end_lr`/`min_lr`/`eta_min`.")

            normalized.append({
                "steps": steps,
                "start_lr": start_lr,
                "end_lr": end_lr,
            })
            previous_end_lr = end_lr

        return normalized

    def _get_target_lr_for_epoch(self, epoch_idx: int) -> float:
        first_start_lr = self.phases[0]["start_lr"]
        if self.warmup_steps > 0 and epoch_idx < self.warmup_steps:
            progress = float(epoch_idx + 1) / float(self.warmup_steps)
            return self.warmup_init_lr + (first_start_lr - self.warmup_init_lr) * progress

        phase_epoch = epoch_idx - self.warmup_steps
        for phase in self.phases:
            steps = phase["steps"]
            if phase_epoch < steps:
                if steps == 1:
                    return phase["end_lr"]
                t = float(phase_epoch) / float(steps - 1)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * t))
                return phase["end_lr"] + (phase["start_lr"] - phase["end_lr"]) * cosine_decay
            phase_epoch -= steps

        return self.phases[-1]["end_lr"]

    def _apply_target_lr(self, target_lr: float):
        scale = target_lr / self.reference_lr if self.reference_lr > 0 else 1.0
        self._last_lrs = []
        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = float(self._base_lrs[idx]) * scale
            self._last_lrs.append(group["lr"])

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)

        target_lr = self._get_target_lr_for_epoch(max(0, self.last_epoch))
        self._apply_target_lr(target_lr)

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch,
            "warmup_steps": self.warmup_steps,
            "warmup_init_lr": self.warmup_init_lr,
            "reference_lr": self.reference_lr,
            "base_lrs": self._base_lrs,
            "phases": self.phases,
        }

    def load_state_dict(self, state):
        self.last_epoch = int(state.get("last_epoch", self.last_epoch))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.warmup_init_lr = float(state.get("warmup_init_lr", self.warmup_init_lr))
        self.reference_lr = float(state.get("reference_lr", self.reference_lr))

        loaded_base_lrs = state.get("base_lrs")
        if isinstance(loaded_base_lrs, list) and len(loaded_base_lrs) == len(self._base_lrs):
            self._base_lrs = [float(v) for v in loaded_base_lrs]

        loaded_phases = state.get("phases")
        if isinstance(loaded_phases, list) and len(loaded_phases) > 0:
            self.phases = self._normalize_phases(loaded_phases)

        target_lr = self._get_target_lr_for_epoch(max(0, self.last_epoch))
        self._apply_target_lr(target_lr)

    def get_last_lr(self):
        return list(self._last_lrs)


def _normalize_phase_shortcuts(kwargs: dict):
    phases = kwargs.pop("phases", None)
    if phases is not None:
        return phases

    parsed = []
    for idx in range(1, 10):
        step_key = f"phase{idx}_steps"
        if step_key not in kwargs:
            continue
        phase = {
            "steps": kwargs.pop(step_key),
        }
        for key in ("start_lr", "end_lr", "max_lr", "min_lr", "eta_min"):
            source = f"phase{idx}_{key}"
            if source in kwargs:
                phase[key] = kwargs.pop(source)
        parsed.append(phase)

    return parsed if len(parsed) > 0 else None


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        if 't_mult' in kwargs and 'T_mult' not in kwargs:
            kwargs['T_mult'] = kwargs.pop('t_mult')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        kwargs.pop('total_iters', None)
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    elif name in ("warmup_then_three_phase_cosine", "warmup_then_phased_cosine", "three_phase_cosine"):
        warmup_steps = int(kwargs.pop("warmup_steps", kwargs.pop("num_warmup_steps", 0)))
        warmup_init_lr = float(kwargs.pop("warmup_init_lr", 0.0))
        reference_lr = kwargs.pop("reference_lr", None)
        phases = _normalize_phase_shortcuts(kwargs)
        if phases is None:
            raise ValueError(
                "Three-phase cosine scheduler requires `phases`, "
                "or shortcut keys like `phase1_steps`, `phase1_start_lr`, `phase1_end_lr`."
            )
        return _WarmupThenPhasedCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            phases=phases,
            warmup_init_lr=warmup_init_lr,
            reference_lr=reference_lr,
        )
    elif name in ("warmup_then_cosine_restarts", "warmup_then_cosine_with_restarts"):
        warmup_steps = int(kwargs.pop("warmup_steps", kwargs.pop("num_warmup_steps", 0)))
        if 't_mult' in kwargs and 'T_mult' not in kwargs:
            kwargs['T_mult'] = kwargs.pop('t_mult')
        phases = _normalize_phase_shortcuts(kwargs)
        if phases is not None:
            warmup_init_lr = float(kwargs.pop("warmup_init_lr", 0.0))
            reference_lr = kwargs.pop("reference_lr", None)
            return _WarmupThenPhasedCosineScheduler(
                optimizer=optimizer,
                warmup_steps=warmup_steps,
                phases=phases,
                warmup_init_lr=warmup_init_lr,
                reference_lr=reference_lr,
            )
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        ctor = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        return _WarmupThenScheduler(optimizer, warmup_steps, ctor, kwargs)
    else:
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )


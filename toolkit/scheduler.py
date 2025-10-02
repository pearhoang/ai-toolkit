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
        # use diffusers' constant_with_warmup for a smooth ramp from 0 -> base lr
        self._warmup = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.warmup_steps
        ) if self.warmup_steps > 0 else None
        self._after = after_scheduler_ctor(optimizer, **after_kwargs)
        self.last_epoch = -1  # keep parity with torch schedulers

    def step(self):
        self.last_epoch += 1
        if self._warmup is not None and self.last_epoch < self.warmup_steps:
            self._warmup.step()
        else:
            self._after.step()

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
            # Fallback: read from optimizer
            return [group.get("lr", None) for group in self.optimizer.param_groups]

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
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    
    if name in ("warmup_then_cosine_restarts", "warmup_then_cosine_with_restarts"):
        warmup_steps = int(kwargs.pop("warmup_steps", kwargs.pop("num_warmup_steps", 0)))
        # Map our friendly keys to torch's expected keys
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        ctor = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        return _WarmupThenScheduler(optimizer, warmup_steps, ctor, kwargs)
else:
        # try to use a diffusers scheduler
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

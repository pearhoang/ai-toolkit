"""CPU regression of actual optimizer factory, no model dependencies."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('optimizer_under_test',Path(__file__).resolve().parents[1]/'toolkit/optimizer.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)


class AdamWEpsilonTests(unittest.TestCase):
    def test_legacy_updates_match_previous_constructor_exactly(self):
        a=torch.nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float32))
        b=torch.nn.Parameter(a.detach().clone())
        kwargs=dict(betas=(.9,.99),weight_decay=.01)
        old=torch.optim.AdamW([a],lr=4e-5,eps=1e-6,**kwargs)
        new=factory.get_optimizer([b],'adamw',4e-5,kwargs)
        for gradient in [[1e-7,-2e-6],[-3e-8,4e-7],[.02,-.03]]:
            a.grad=torch.tensor(gradient);b.grad=a.grad.clone();old.step();new.step()
            self.assertTrue(torch.equal(a,b))
            for key in old.state[a]:self.assertTrue(torch.equal(old.state[a][key],new.state[b][key]))
        self.assertNotIn('eps',kwargs)

    def test_explicit_epsilon_without_duplicate_kwargs_or_config_mutation(self):
        config=dict(eps=1e-8,betas=(.9,.99),weight_decay=.01,fused=False)
        before=config.copy()
        p=torch.nn.Parameter(torch.zeros(2))
        opt=factory.get_optimizer([p],'adamw',4e-5,config)
        self.assertEqual(opt.param_groups[0]['eps'],1e-8)
        self.assertEqual(config,before)
        p.grad=torch.full_like(p,1e-7);opt.step()
        self.assertTrue(torch.isfinite(p).all())

    def test_small_gradient_mechanism_and_zero_gradient_remain_finite(self):
        results=[]
        for eps in [1e-6,1e-8]:
            p=torch.nn.Parameter(torch.zeros(2))
            opt=factory.get_optimizer([p],'adamw',4e-5,dict(eps=eps,weight_decay=0.))
            p.grad=torch.tensor([1e-7,0.]);opt.step()
            results.append(p[0].detach().abs().item());self.assertEqual(p[1].item(),0.)
        self.assertAlmostEqual(results[1]/results[0],10.,places=4)

    def test_none_and_per_group_epsilon(self):
        p=torch.nn.Parameter(torch.zeros(1))
        self.assertEqual(factory.get_optimizer([p],'adamw',optimizer_params=None).param_groups[0]['eps'],1e-6)
        opt=factory.get_optimizer([dict(params=[p],eps=2e-8)],'adamw',optimizer_params=dict(eps=1e-8))
        self.assertEqual(opt.param_groups[0]['eps'],2e-8)

    def test_state_resume_keeps_checkpoint_epsilon_like_pytorch(self):
        p=torch.nn.Parameter(torch.zeros(1));q=torch.nn.Parameter(torch.zeros(1))
        a=factory.get_optimizer([p],'adamw',optimizer_params=dict(eps=1e-8))
        b=factory.get_optimizer([q],'adamw',optimizer_params=dict(eps=1e-6))
        p.grad=torch.ones(1);a.step();b.load_state_dict(a.state_dict())
        self.assertEqual(b.param_groups[0]['eps'],1e-8)


if __name__=='__main__':unittest.main(verbosity=2)

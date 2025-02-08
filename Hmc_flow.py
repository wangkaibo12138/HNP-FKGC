import torch
import torch.nn as nn
import normflows as nf
from torch.distributions import kl_divergence

class Flow(nn.Module):
    def __init__(self, latent_size, flow, K):
        super().__init__()
        if flow == "Planar":
            flows = [nf.Planar((latent_size,)) for _ in range(K)]
        self.flows = nn.ModuleList(flows)

    def forward(self, z, base_dist, prior=None):
        ld = 0.0
        p_0 = torch.sum(base_dist.log_prob(z), -1)
        for flow in self.flows:
            z.requires_grad_(True)
            z, ld_ = flow(z)
            ld += ld_
        kld = p_0 - torch.sum(prior.log_prob(z), -1) - ld.view(-1) if prior else torch.zeros_like(p_0)
        return z, kld

class HMCModule(nn.Module):
    def __init__(self, latent_size, step_size=0.5, num_steps=10):
        super(HMCModule, self).__init__()
        self.latent_size = latent_size
        self.step_size = step_size
        self.num_steps = num_steps

    def forward(self, z, base_dist, prior=None):
        z.requires_grad_(True)
        p = torch.randn_like(z)
        z_init, p_init = z.clone(), p.clone()
        potential_energy = -base_dist.log_prob(z).sum(dim=-1)
        kinetic_energy = (p ** 2).sum(dim=-1) / 2
        hamiltonian_init = potential_energy + kinetic_energy

        z, p = self.leapfrog(z, p, base_dist)
        potential_energy = -base_dist.log_prob(z).sum(dim=-1)
        kinetic_energy = (p ** 2).sum(dim=-1) / 2
        hamiltonian_final = potential_energy + kinetic_energy

        accept_prob = torch.exp(hamiltonian_init - hamiltonian_final)
        accept = (torch.rand_like(accept_prob) < accept_prob)
        z = torch.where(accept.unsqueeze(-1), z, z_init)

        kld = kl_divergence(base_dist, prior).sum(-1) if prior else None
        return z, kld

    def leapfrog(self, z, p, base_dist):
        for _ in range(self.num_steps):
            log_prob = base_dist.log_prob(z).sum()
            grad_z = torch.autograd.grad(log_prob, z, retain_graph=True)[0]
            p -= 0.5 * self.step_size * grad_z
            z = z + self.step_size * p
            z.requires_grad_(True)
            log_prob = base_dist.log_prob(z).sum()
            grad_z = torch.autograd.grad(log_prob, z, retain_graph=True)[0]
            p -= 0.5 * self.step_size * grad_z
        return z, -p

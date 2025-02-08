from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from Multi import RelationalPathGNN
from Lstm import LSTM_attn, SequenceAttention
from Hmc_flow import Flow, HMCModule

class EmbeddingModule(nn.Module):
    def __init__(self, emb_dim, z_dim, out_size):
        super().__init__()
        self.head_encoder = nn.Linear(emb_dim, emb_dim)
        self.tail_encoder = nn.Linear(emb_dim, emb_dim)
        self.dr = nn.Linear(z_dim, 1)

    def forward(self, h, t, r, pos_num, z):
        d_r = self.dr(z)
        z = z.unsqueeze(2)
        h = h + self.head_encoder(z)
        t = t + self.tail_encoder(z)
        tmp_score = torch.norm(h + r - t, 2, -1)
        score = - torch.norm(tmp_score - d_r ** 2, 2, -1)
        return score[:, :pos_num], score[:, pos_num:]

class RelationLatentEncoder(nn.Module):
    def __init__(self, few, embed_size=100, num_hidden1=500, num_hidden2=200, r_dim=100, dropout_p=0.5):
        super().__init__()
        self.rel_fc1 = nn.Sequential(
            nn.Linear(2 * embed_size + 1, num_hidden1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p)
        )
        self.rel_fc2 = nn.Sequential(
            nn.Linear(num_hidden1, num_hidden2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p)
        )
        self.rel_fc3 = nn.Linear(num_hidden2, r_dim)

    def forward(self, inputs, y):
        x = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        label = torch.ones_like(x[..., :1]) if y == 1 else torch.zeros_like(x[..., :1])
        x = torch.cat([x, label], dim=-1)
        x = self.rel_fc1(x)
        x = self.rel_fc2(x)
        return self.rel_fc3(x)

class DistributionEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super().__init__()
        self.r_to_hidden = nn.Linear(r_dim, r_dim)
        self.hidden_to_mu = nn.Linear(r_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = F.relu(self.r_to_hidden(torch.mean(r, dim=1)))
        mu = self.hidden_to_mu(hidden)
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return Normal(mu, sigma)

class HNPModel(nn.Module):
    def __init__(self, g, dataset, parameter, num_symbols, embed=None):
        super().__init__()
        self.device = parameter['device']
        self.r_path_gnn = RelationalPathGNN(g, dataset['ent2id'], len(dataset['rel2emb']), parameter)
        self.relation_learner = LSTM_attn(
            embed_size=parameter['embed_dim'],
            n_hidden=parameter['lstm_hiddendim'],
            out_size=parameter['embed_dim'],
            layers=parameter['lstm_layers'],
            dropout=parameter['dropout_p']
        )
        self.latent_encoder = RelationLatentEncoder(
            parameter['few'],
            embed_size=parameter['embed_dim'],
            num_hidden1=500,
            num_hidden2=200,
            r_dim=parameter['z_dim']
        )
        self.xy_to_mu_sigma = DistributionEncoder(parameter['r_dim'], parameter['z_dim'])
        self.flows = Flow(parameter['z_dim'], parameter['flow'], parameter['K']) if parameter['flow'] != 'none' else None
        self.hmc = HMCModule(parameter['z_dim'])
        self.embedding_learner = EmbeddingModule(parameter['embed_dim'], parameter['z_dim'], parameter['embed_dim'])
        self.loss_func = nn.MarginRankingLoss(parameter['margin'])

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False):
        support, query = task['support'], task['query']
        support_emb = self.r_path_gnn(support['triples'])
        query_emb = self.r_path_gnn(query['triples'])
        rel_emb = self.relation_learner(support_emb)
        latent_r = self.latent_encoder(rel_emb, y=1)
        prior = self.xy_to_mu_sigma(latent_r)
        z = prior.rsample()
        if self.flows:
            z, kld = self.flows(z, prior, prior)
        else:
            z, kld = self.hmc(z, prior, prior)
        pos_score, neg_score = self.embedding_learner(support_emb[:, :, 0], support_emb[:, :, 1], rel_emb, support['pos_num'], z)
        loss = self.loss_func(pos_score, neg_score, torch.ones_like(pos_score))
        return loss, kld

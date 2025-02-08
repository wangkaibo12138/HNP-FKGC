import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class RPGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hop, n_rel, levels):
        super().__init__()
        emb_dim = in_features
        self.conv_in = RPLayer(emb_dim, in_features, hidden_features, n_rel, levels)
        self.conv_out = RPLayer(emb_dim, hidden_features, out_features, n_rel, levels)
        self.hop = hop
        if hop > 2:
            self.conv_hidden = nn.ModuleList(
                [RPLayer(emb_dim, hidden_features, hidden_features, n_rel, levels) for _ in range(hop - 2)])

    def forward(self, blocks, x):
        x = F.relu(self.conv_in(blocks[0], x))
        if self.hop > 2:
            for i, conv in enumerate(self.conv_hidden):
                x = F.relu(conv(blocks[i + 1], x))
        x = F.relu(self.conv_out(blocks[-1], x))
        return x

class RelationalPathGNN(nn.Module):
    def __init__(self, g, ent2id, num_rel, parameter):
        super(RelationalPathGNN, self).__init__()
        self.ent2id_dict = ent2id
        self.device = parameter['device']
        self.hop = parameter['hop']
        self.es = parameter['embed_dim']
        self.g_batch = parameter['g_batch']
        self.g = g
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(parameter['hop'], prefetch_node_feats=['feat'],
                                                                     prefetch_edge_feats=['feat', 'eid'])
        levels = parameter.get('levels', [1,3,5])
        self.gcn = RPGNN(self.es, self.es * 2, self.es, self.hop, num_rel, levels)
        self.num_rel = num_rel

    def ent2id(self, triples):
        idx = [[[self.ent2id_dict[t[0]], self.ent2id_dict[t[2]]] for t in batch] for batch in triples]
        idx = torch.LongTensor(idx).to(self.device)
        return idx  # B * few * 2

    def forward(self, triples):
        '''
        inputs:
            task: Batch triplets, B * few
        outputs:
            emb: B * few * es
        '''

        idx = self.ent2id(triples)
        batch_size, few_shot = idx.shape[0], idx.shape[1]
        idx = idx.view(-1)
        dataloader = dgl.dataloading.DataLoader(
            self.g, idx, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        out_emb = []
        for input_nodes, output_nodes, blocks in dataloader:
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            out_emb.append(out_features)
        out_emb = torch.cat(out_emb, dim=0)
        out_emb = out_emb.view(batch_size, few_shot, 2, -1)
        return out_emb


class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features, allow_zero_in_degree=True)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x


class AdaptiveMultiLevelRelationModule(nn.Module):
    def __init__(self, in_features, out_features, levels):
        super(AdaptiveMultiLevelRelationModule, self).__init__()
        self.levels = levels
        self.linears = nn.ModuleList([nn.Linear(in_features, out_features) for _ in levels])
        self.attn_fc = nn.Linear(out_features, 1, bias=False)

    def forward(self, g, feat):
        multi_level_feats = [F.relu(linear(feat)) for linear in self.linears]
        attn_scores = [self.attn_fc(feat) for feat in multi_level_feats]
        attn_scores = torch.stack(attn_scores, dim=-1)
        attn_scores = F.softmax(attn_scores, dim=-1)
        weighted_level_feats = [attn_scores[..., i] * multi_level_feats[i] for i in range(len(self.levels))]
        out_feat = torch.stack(weighted_level_feats, dim=-1).sum(dim=-1)
        return out_feat


class RPLayer(nn.Module):
    def __init__(self, emb_dim, in_feat, out_feat, num_rels, levels):
        super().__init__()
        self.num_rels = num_rels
        self.linear_r = dgl.nn.pytorch.TypedLinear(in_feat + emb_dim * 2, out_feat, num_rels)
        self.attn_fc = nn.Linear(emb_dim + out_feat, 1, bias=False)
        self.h_bias = nn.Parameter(torch.Tensor(out_feat))
        self.loop_weight = nn.Parameter(torch.Tensor(emb_dim, out_feat))
        self.amrm = AdaptiveMultiLevelRelationModule(out_feat, out_feat, levels)
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    def edge_agg(self, edges):
        """Relation Message Passing"""
        x = torch.cat([edges.src['h'], edges.data['feat'], edges.dst['feat']], dim=1)
        m = self.linear_r(x, edges.data['eid'])
        attn = F.leaky_relu(self.attn_fc(torch.cat([edges.dst['feat'], m], dim=1)))
        return {'h': m, 'z': attn}

    def forward(self, g, feat):
        with g.local_scope():
            # Norm
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
            g.srcdata['h'] = feat
            g.apply_edges(self.edge_agg)
            e = g.edata.pop('z')
            a = edge_softmax(g, e)
            g.edata['h'] = a * g.edata['h']
            g.update_all(dgl.function.copy_e('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.dstdata['h']
            h = h + g.dstdata['feat'] @ self.loop_weight
            # Norm 
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = h * norm
            h = rst + self.h_bias

            # Adaptive Multi-Level Relation Module
            h = self.amrm(g, h)

            return h

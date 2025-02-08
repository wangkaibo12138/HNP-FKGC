import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceAttention(nn.Module):
    def __init__(self, tau=0.1):
        super(SequenceAttention, self).__init__()
        self.tau = tau  # Threshold value to filter low scores

    def forward(self, x):
        # Compute similarity scores using dot product
        scores = torch.bmm(x, x.transpose(1, 2)) / x.size(-1) ** 0.5
        
        # Filter scores based on threshold tau
        scores = torch.where(scores > self.tau, scores, torch.tensor(-1e6).to(x.device))
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum to get context
        context = torch.bmm(weights, x)
        return context, weights

class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embed_size * 2, n_hidden, layers, bidirectional=True, dropout=dropout)
        self.attention = SequenceAttention()
        self.out = nn.Linear(n_hidden * 2, out_size)

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = torch.zeros(self.layers * 2, size[0], self.n_hidden).to(inputs.device)
        cell_state = torch.zeros(self.layers * 2, size[0], self.n_hidden).to(inputs.device)
        output, _ = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attention_output, _ = self.attention(output)
        output = self.out(attention_output[:, -1, :])
        return output.view(size[0], 1, 1, self.out_size)

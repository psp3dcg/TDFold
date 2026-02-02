import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_k, bias=False)
        self._norm_fact = dim_k ** -0.5

    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        # import pdb
        # pdb.set_trace()
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact

        dist = torch.softmax(dist, dim=-1)

        att = torch.bmm(dist, v)
        return att
    
class FeatEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatEncoder, self).__init__()

        self.proj = nn.Linear(input_dim, output_dim)
        self.self_attn = SelfAttention(output_dim,output_dim,output_dim)

    def forward(self, x):

        return self.self_attn(self.proj(x))
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim, seq_len, base = 10000, device = None):
        super().__init__()
        inv_freq = 1 / (base ** (torch.arange(0, hidden_dim, 2).float().to(device) / hidden_dim))
        self.register_buffer("inv_freq", inv_freq)
        pos = torch.arange(0, seq_len, device = self.inv_freq.device, dtype = self.inv_freq.dtype)
        freqs = torch.enisum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistant = False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistant = False)

    def forward(self, input, seq_len=None):
        return self.cos[:, :, :seq_len, ...].to(dtype = input.dtype), self.sin[:, :, :seq_len, ...].to(dtype = input.dtype)


def apply_rope(key, query, cos, sin, position_ids):
    gather_id = position_ids[:, None, :, None].repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_id.shape[0], 1, 1, 1), 2, gather_id)
    sin = torch.gather(sin.repeat(gather_id.shape[0], 1, 1, 1), 2, gather_id)
    key_ebd = key * cos + torch.cat((-key[:, :, :, key.shape[-1]//2:], key[:, :, :, :key.shape[-1]//2]), dim = -1) * sin
    query_ebd = query * cos + torch.cat((-query[:, :, :, query.shape[-1]//2:], query[:, :, :, :query.shape[-1]//2]), dim = -1) * sin
    return key_ebd, query_ebd

class TransformerAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, max_pos):
        super().__init__()
        self.att_hidden = hidden_dim // num_head
        self.num_head = num_head
        self.proj_k = nn.linear(hidden_dim, self.att_hidden * num_head)
        self.proj_q = nn.linear(hidden_dim, self.att_hidden * num_head)
        self.proj_v = nn.linear(hidden_dim, self.att_hidden * num_head)
        self.proj_o = nn.linear(self.att_hidden * num_head, hidden_dim)
        self.pos_ebd = RotaryEmbedding(hidden_dim, max_pos)

    def forward(self, input):
        bsz, seq_len, hidden_dim = input.size()
        key = self.proj_k(input).view(bsz, seq_len, self.num_head, self.att_hidden).transpose(1, 2)
        query = self.proj_q(input).view(bsz, seq_len, self.num_head, self.att_hidden).transpose(1, 2)
        cos, sin = self.pos_ebd(key, seq_len)
        position_ids = torch.arange(0, seq_len, dtype = torch.long, device = input.device).unsqueeze(0).view(-1, seq_len)
        key, query = apply_rope(key, query, cos, sin, position_ids)
        key = key.transpose(-2, -1)
        value = self.proj_v(input).view(bsz, seq_len, self.num_head, self.att_hidden).transpose(1, 2)
        scores = torch.matmul(query, key) / torch.sqrt(self.att_hidden)
        mask = torch.ones(self.att_hidden, self.att_hidden).tril(diagonal = 0)
        scores.masked_fill_(mask.logical_not(), float("-inf"))

        p_attn = torch.nn.functional.softmax(scores, dim = -1)
        output = torch.matmul(p_attn, value).view(bsz, seq_len, self.att_hidden * self.num_head)
        output = self.proj_o(output)
        return output
    
class RMSNorm(nn.module):
    def __init__(self, hidden_dim, esp = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.var = esp

    def forward(self, input):
        variance = input.to(torch.float32).pow(2).mean(-1, keepdim = True)
        output = input * torch.rsqrt(variance + self.var)
        return self.weight * output
        

        

class TransformerMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias = False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias = False)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias = False)
        self.dropout = nn.Dropout(p)

    def forward(self, input):
        return self.dropout(self.down_proj(torch.nn.functional.silu(self.gate_proj(input)) * self.up_proj(input)))


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_head):
        super(TransformerLayer, self).__init__()
        self.attention = TransformerAttention(hidden_dim, num_head)
        self.mlp = TransformerMLP(hidden_dim, intermediate_dim)
        self.pre_norm = RMSNorm(hidden_dim)
        self.post_attn_norm = RMSNorm(hidden_dim)


    def forward(self, input):
        residual = input
        input = self.pre_norm(input)
        output = self.attention(input)
        output = output + residual
        residual = output
        output = self.post_attn_norm(output)
        output = self.mlp(output)
        return output + residual


class Transformer(torch.nn.Module):
    def __init__(self, num_layer, hidden_dim, intermediate_dim, num_head, vocab_size, drop_p):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList([TransformerLayer(hidden_dim, intermediate_dim, num_head) for i in range(num_layer)])
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(drop_p)
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias = False)

    def forward(self, input):
        # bz * seq_len
        output = self.embedding(input)
        output = self.dropout(output)
        for layer in self.layers:
            output = layer(output)
        output = self.norm(output)
        output = self.dropout(output)
        return self.lm_head(output)
        

input = torch.tensor([])
target = torch.tensor([])
model = Transformer()
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
optimizer.zero_grad()
output = model.forward(input)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, target)
loss.backward()
optimizer.step()


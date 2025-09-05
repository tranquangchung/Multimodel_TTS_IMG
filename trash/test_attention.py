import torch
import torch.nn as nn

B = 2        # batch size
L1 = 50      # length từ T5 (text tokens)
L2 = 300     # length từ audio encoder (audio frames)
D1 = 768     # hidden dim T5
D2 = 512     # hidden dim audio
d_model = 256  # projection dim cho attention

# Giả sử h: output từ T5 Encoder
h = torch.randn(B, L1, D1)

# Giả sử h2: output từ audio encoder
h2 = torch.randn(B, L2, D2)

# Linear projection để đưa về cùng dim
proj_q = nn.Linear(D2, d_model)  # query từ audio
proj_k = nn.Linear(D1, d_model)  # key từ text
proj_v = nn.Linear(D1, d_model)  # value từ text

Q = proj_q(h2)   # [B, L2, d_model]
K = proj_k(h)    # [B, L1, d_model]
V = proj_v(h)    # [B, L1, d_model]

# Multi-head cross attention
attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
out, weights = attn(Q, K, V)

print(out.shape)      # [B, L2, d_model]
print(weights.shape)  # [B, L2, L1]

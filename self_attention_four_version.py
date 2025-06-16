import math
import torch
import torch.nn as nn
print(torch.__version__)

## version 1
class SelfAttention(nn.Module):
    def __init__(self,embedding_dim: int  = 768) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.query_proj = nn.Linear(embedding_dim,embedding_dim)
        self.key_proj = nn.Linear(embedding_dim,embedding_dim)
        self.value_proj = nn.Linear(embedding_dim,embedding_dim)
    
    def forward(self,x):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        v = self.value_proj(x)

        attention_value = torch.matmul(
            Q , K.transpose(-1, -2)
        )
        attention_weight = torch.softmax(
            attention_value / math.sqrt(self.embedding_dim),
            dim = -1
        )

        output = torch.matmul(attention_weight , v)

        return output
import math
import torch
import torch.nn as nn
print(torch.__version__)

## version 1 : normal version
class SelfAttentionv1(nn.Module):
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
    
## version 2 : optimize matrix calculations
class SelfAttentionV2(nn.Moudle):
    def __init__(self,embedding):
        super().__init__()
        self.embedding = embedding
        self.proj = nn.Linear(embedding,embedding*3)

    def forward(self,x):
        QKV = self.proj(x)
        Q , K , V = torch.split(QKV,self.embedding,dim = -1)
        attention_weight = torch.softmax(
            torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(self.embedding),
            dim = -1
        )
        print(attention_weight)
        output = torch.matmul(attention_weight,V)

        return output
    

## version 3 : add some details 
## 1. dropout position
## 2. attetion mask
## 3. output matrix projection

class SelfAttentionV3(nn.Module):
    def __init__(self,embedding,dropout_rate = 0.1):
        super().__init__()
        self.embedding = embedding
        self.proj = nn.Linear(embedding,embedding*3)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_proj = nn.Linear(embedding,embedding)

    def forward(self,x,attention_mask = None):
        # x(batch, seq, embedding)
        QKV = self.proj(x)
        Q, K, V = torch.split(QKV, self.embedding, dim = -1)

        #attention_weight(batch, seq, seq)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.embedding)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-1e20")
            )
        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )
        attention_result = attention_weight @ V
        output = self.output_proj(attention_result)
        return output
 
## formal version


x = torch.rand(3, 4, 2)
    





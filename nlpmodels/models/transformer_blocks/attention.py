import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadedAttention(nn.Module):
    """
    The multi-headed self-attention layer of the Transformer.

    Self-attention works by mapping a query, key, value for each token in a sequence
    and learns the relative importance of each to a target. Multi-headed means there
    are N of them working in parallel.

    The particular attention function employed is “Scaled Dot-Product Attention" (scaled by 1/sqrt(d_k)).
    """

    def __init__(self, num_heads: int, dim_model: int, dropout: float):
        """
           Args:
               num_heads (int): number of heads/ simultanenous attention layers
               dim_model (int): total concatenated size of attention layer output
               dropout (float): dropout probability
        """
        super(MultiHeadedAttention, self).__init__()

        # parameter dimensions
        # Per paper, "set d_v == d_k == d_model//h==64"
        # d_k in paper, dimensions of queries and keys (apply dot-product together)
        self._dim_keys = dim_model // num_heads
        self._dim_values = self._dim_keys  # d_v in paper, dimensions of values
        self._num_heads = num_heads  # h in paper, or number of heads ("h==8")

        # linear layers
        self._linear_layer_queries = nn.Linear(dim_model, dim_model)
        self._linear_layer_keys = nn.Linear(dim_model, dim_model)
        self._linear_layer_values = nn.Linear(dim_model, dim_model)
        self._linear_layer_final = nn.Linear(dim_model, dim_model)
        self._attention_tensor = None  # Caching results for any potential future visualization/inspection
        self._dropout = nn.Dropout(p=dropout)

    def compute_attention(self,query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor):
        """
        Computes attention math, returns attention matrix
        Args:
            query (torch.Tensor): query matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            key (torch.Tensor): key matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            value (torch.Tensor): values matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            mask (torch.Tensor): values matrix of (batch_size,1,1,max_seq_length) size
        Returns:
            returns the attention matrix, attention probabilities (softmax output)
        """
        assert query.size(-1) == self._dim_keys
        # Scaled dot-product calculation (Q,K) -> scores
        scores = torch.matmul(query, key.transpose(2,3)) / math.sqrt(self._dim_keys)
        # Wherever mask == False is where padding is, so ignore values
        scores = scores.masked_fill(mask == False, -1e9)
        # Calculate soft-max of scores -> attention_probas (probabilities for each value)
        attention_probas = self._dropout(F.softmax(scores, dim=-1))
        # attention is attention_probas * values
        attention_values = torch.matmul(attention_probas, value)
        return attention_values, attention_probas

    def forward(self, query, key, value, mask):
        """
        Main function call for attention mechanism.
        maps (q,k,v) -> linear -> self_attn -> concat -> linear
        Args:
            query (torch.Tensor): query matrix of (batch_size,_dim_keys) size
            key (torch.Tensor): key matrix of (batch_size,_dim_keys) size
            value (torch.Tensor): values matrix of (batch_size,_dim_values) size
            mask (torch.Tensor): bool mask of (batch_size,1,max_seq_length) size
        Returns:
            returns final linear layer output of (batch_size, max_seq_length, dim_model) size
        """

        batch_size = query.size(0)
        max_seq_length = mask.size(-1)

        # 1) Apply linear to query, key, and value, convert to 4D tensor
        # (batch_size, num_heads,max_seq_length, dim_keys)
        query = self._linear_layer_queries(query).view(batch_size, self._num_heads, max_seq_length, self._dim_keys)
        key = self._linear_layer_keys(key).view(batch_size, self._num_heads, max_seq_length, self._dim_keys)
        value = self._linear_layer_values(value).view(batch_size, self._num_heads, max_seq_length, self._dim_keys)
        # print(query.shape,self._num_heads,self._dim_keys,mask.shape)

        # Same mask applied to all h heads.
        # mask dims will become (batch_size,1,1,max_seq_length)
        mask = mask.unsqueeze(1)

        # 2) compute attention
        x, self._attention_tensor = self.compute_attention(query, key, value, mask)

        # 3) concatenate
        x = x.transpose(1, 2).contiguous().view(batch_size, max_seq_length, self._num_heads * self._dim_keys)
        # x is now (batch_size, max_seq_length, dim_model)

        # 4) apply final linear
        x = self._linear_layer_final(x)

        return x
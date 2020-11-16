import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """
    The multi-headed self-attention layer of the Transformer.
    This is one of the most critical components
    of the Transformer architecture and the basis of continued research.

    Self-attention works by mapping a query, key, value for each token in a sequence
    and learns the relative importance of each to a target. Multi-headed means there
    are N of them working in parallel.

    This attention object is utilized in 2 ways:
    (1) encoder-decoder attention (decoder[l-1]::queries, encoder::keys and values)
    (2) self-attention (encoder[l-1]::queries,keys,and values)

    The particular attention mechanism employed is
    “scaled dot-product attention" (scaled by 1/sqrt(d_k)).

    """

    def __init__(self, num_heads: int,
                 dim_model: int,
                 dropout: float,
                 residual_dropout: float = 0.0):
        """
        Args:
           num_heads (int): number of heads/ simultanenous attention layers
           dim_model (int): total concatenated size of attention layer output + embedding input size
           dropout (float): dropout probability for attention mechanism
           residual_dropout (float): dropout for self-attention (GPT specific)
        """
        super(MultiHeadedAttention, self).__init__()

        # parameter dimensions
        # Per paper, "set d_v == d_k == d_model//h==64"
        # d_k in paper, dimensions of queries and keys (apply dot-product together)
        # calling d_v, d_k == "dim_head"
        self._dim_head = dim_model // num_heads
        self._num_heads = num_heads  # h in paper, or number of heads ("h==8")

        # linear layers
        self._linear_layer_queries = nn.Linear(dim_model, dim_model)
        self._linear_layer_keys = nn.Linear(dim_model, dim_model)
        self._linear_layer_values = nn.Linear(dim_model, dim_model)
        self._linear_layer_final = nn.Linear(dim_model, dim_model)
        # Caching results for any potential future visualization/inspection
        self._attention_tensor = None
        self._dropout = nn.Dropout(p=dropout)
        # GPT specific
        self._residual_dropout = nn.Dropout(p=residual_dropout)

    def compute_attention(self, query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor, mask: torch.Tensor):
        """
        Attention function to compute scaled dot-product attention between query,
        key and determine which values to "pay attention to".


        Args:
            query (torch.Tensor):
                query 4D matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            key (torch.Tensor):
                key 4D matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            value (torch.Tensor):
                values 4D matrix of (batch_size, num_heads,max_seq_length, dim_keys) size
            mask (torch.Tensor):
                values 4D matrix of (batch_size,1,1,max_seq_length) size
        Returns:
            returns the attention values, attention probabilities (softmax output)
        """
        # confirm that dim 4 is correct
        assert query.size(-1) == self._dim_head

        # Scaled dot-product calculation (Q,K) -> scores
        # matmul considers last 2 dimensions in batch matrix multiplication.
        # (batch_size, num_heads,max_seq_length, dim_keys) x
        # (batch_size, num_heads, dim_keys, max_seq_length)  ->
        # scores is (batch_size, num_heads, max_seq_length, max_seq_length)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self._dim_head)
        # Wherever mask == False is where padding is, so set scores to -inf for softmax calculation
        # Note: would use float('-inf'), but softmax issue
        scores = scores.masked_fill(mask == 0, -1.e9)
        # Calculate soft-max of scores -> attention_probas (probabilities for each value)
        attention_probas = self._dropout(F.softmax(scores, dim=-1))
        # attention is expected values = attention_probas * values
        # (batch_size, num_heads, max_seq_length, max_seq_length) x
        # (batch_size, num_heads, max_seq_length, dim_keys) ->
        # attn_values is (batch_size, num_heads, max_seq_length, dim_keys)
        attention_values = torch.matmul(attention_probas, value)
        return attention_values, attention_probas

    def forward(self, query, key, value, mask):
        """
        Main function call for attention mechanism.
        maps (q,k,v) -> linear -> self_attn -> concat -> linear.
        Args:
            query (torch.Tensor): query matrix of (batch_size, max_seq_length, dim_model) size
            key (torch.Tensor): key matrix of (batch_size, max_seq_length, dim_model) size
            value (torch.Tensor): values matrix of (batch_size, max_seq_length, dim_model) size
            mask (torch.Tensor): bool mask 3D tensor of (batch_size,1,max_seq_length) size
        Returns:
            returns final linear layer output of (batch_size, max_seq_length, dim_model) size
        """

        batch_size, max_seq_length, dim_model = query.size()

        assert self._num_heads * self._dim_head == dim_model

        # 1) Apply linear to query, key, and value matrices
        # and convert 3D tensor -> 4D matrix
        # (batch_size, max_seq_length, dim_model) ->
        # (batch_size, num_heads,max_seq_length, dim_keys)
        # Note: we separate batch_size x max_seq_length  and dim_model // head_size then re-order
        query = self._linear_layer_queries(query) \
            .view(batch_size, max_seq_length, self._num_heads, self._dim_head).transpose(1, 2)
        key = self._linear_layer_keys(key)\
            .view(batch_size, max_seq_length, self._num_heads, self._dim_head).transpose(1, 2)
        value = self._linear_layer_values(value)\
            .view(batch_size, max_seq_length, self._num_heads, self._dim_head).transpose(1, 2)

        # (batch_size,1,max_seq_length) -> (batch_size,1,1,max_seq_length)
        mask = mask.unsqueeze(1)

        # 2) compute dot-product attention
        attention_values, self._attention_tensor = \
            self.compute_attention(query, key, value, mask)

        # 3) "concatenate" (batch_size,1,1,max_seq_length)->
        # (batch_size, max_seq_length, dim_model)
        # basically, we are re-assembling all heads side by side
        attention_values = attention_values.transpose(1, 2).contiguous().\
            view(batch_size, max_seq_length, self._num_heads * self._dim_head)

        # 4) apply final linear layer
        attention_values = self._linear_layer_final(attention_values)

        # 5) GPT: Add residual attention drop
        attention_values = self._residual_dropout(attention_values)

        return attention_values

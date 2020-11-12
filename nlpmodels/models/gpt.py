import torch
import torch.nn as nn
from nlpmodels.models.transformer_blocks import sublayers, attention, gpt_decoder
from nlpmodels.utils.gpt_batch import GPTBatch


class GPT(nn.Module):
    """
    The GPT class of the decoder-only Transformer.

    MORE DESCRIPTION.
    """

    def __init__(self,
                 vocab_size: int,
                 num_layers_per_stack: int = 12,
                 dim_model: int = 768,
                 dim_ffn: int = 3072,
                 num_heads: int = 12,
                 block_size: int = 512,
                 dropout: float = 0.1):
        """
        Args:
           vocab_size (int): size of the target vocabulary.
           num_layers_per_stack (int): number of sequential encoder/decoder layers.
           dim_model (int): size of the embedding space.
           dim_ffn (int): size of the residual/skip-connection hidden layer.
           num_heads (int): number of simultaneous attention heads calculated during attention.
           block_size (int): the context window of the input/output sequences.
           dropout (float): Hyper-parameter used in drop-out regularization in training.
        """
        super(GPT, self).__init__()

        # (1) calculate embeddings
        self._embeddings = sublayers.NormalizedEmbeddingsLayer(vocab_size, dim_model)
        # (2) calculate positional_encoding (learn-able this time) + drop-out
        self._pos_encoding = sublayers.GPTPositionalEncodingLayer(dim_model,dropout,block_size)
        # (3) Pass embeddings + pe to GPT decoder block
        self._decoder_block = gpt_decoder.GPTCompositeDecoder(
            gpt_decoder.GPTDecoderBlock(block_size,
                                     attention.MultiHeadedAttention(num_heads, dim_model, dropout, dropout),
                                     # replace activation function with GELU
                                     sublayers.PositionWiseFFNLayer(dim_model, dim_ffn, nn.GELU()),
                                     dropout),num_layers_per_stack)

        # (4) put through final linear layer
        self._final_linear = nn.Linear(dim_model, vocab_size)

        # init weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize all parameters to be N(0,0.02) per GPT-1 paper.
        """
        for parameter in self.parameters():
            if parameter.dim() - 1:
                if isinstance(parameter, (nn.Linear, nn.Embedding)):
                    parameter.weight.data.normal_(mean=0.0, std=0.02)

    def _decode(self, data: GPTBatch) -> torch.Tensor:

        embeddings = self._embeddings(data.src)
        # Add output embeddings to pos_encoding, apply drop out
        pos_encoding = self._pos_encoding(embeddings)

        return self._decoder_block(pos_encoding, data.src_mask)

    def forward(self, data: GPTBatch) -> torch.Tensor:

        # pass through decoder blocks
        decode = self._decode(data)
        # calculate yhat. Don't apply softmax before passing to cross entropy
        yhat = self._final_linear(decode)

        return yhat
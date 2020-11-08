import torch
import torch.nn as nn
from nlpmodels.models.transformer_blocks import sublayers, attention, gpt_decoder
from nlpmodels.utils.transformer_batch import TransformerBatch


class GPT(nn.Module):
    """
    The GPT class of the decoder-only Transformer.
    """

    def __init__(self,
                 vocab_size: int,
                 num_layers_per_stack: int = 12,
                 dim_model: int = 768,
                 dim_ffn: int = 3072,
                 num_heads: int = 12,
                 max_length: int = 1000,
                 dropout: float = 0.1):
        """
        Args:
           vocab_size (int): size of the target vocabulary.
           num_layers_per_stack (int): number of sequential encoder/decoder layers.
           dim_model (int): size of the embedding space.
           dim_ffn (int): size of the residual/skip-connection hidden layer.
           num_heads (int): number of simultaneous attention heads calculated during attention.
           max_length (int): the max_seq_length of the input/output sequencies.
           dropout (float): Hyper-parameter used in drop-out regularization in training.
        """
        super(GPT, self).__init__()

        # (1) calculate embeddings
        self._embeddings = sublayers.NormalizedEmbeddingsLayer(vocab_size, dim_model)
        # (2) calculate positional_encoding (learn-able this time)
        self._pos_encoding = None # Note: this must apply drop-out after
        # (3) Pass embeddings + pe to GPT decoder block
        self._decoder_block = gpt_decoder.CompositeDecoder(
            gpt_decoder.DecoderBlock(max_length,
                                     attention.MultiHeadedAttention(num_heads, dim_model, dropout,dropout),
                                     # replace activation function with GELU
                                     sublayers.PositionWiseFFNLayer(dim_model, dim_ffn, nn.GELU()),
                                     dropout),num_layers_per_stack)

        # (4) put through final linear layer
        self._final_linear = nn.Linear(dim_model, vocab_size)

        # init weights
        self._init_weights()

    def _decode(self, index) -> torch.Tensor:

        embeddings = self._embeddings(index)
        # Add output embeddings to pos_encoding, apply drop out
        pos_encoding = self._pos_encoding(embeddings)

        return self._decoder_block(pos_encoding)

    def forward(self, data: TransformerBatch) -> torch.Tensor:

        # pass through decoder blocks
        decode = self._decode(data.src)
        # calculate "logits"
        yhat = self._final_linear(decode)

        return yhat


if __name__ == '__main__':
    pass

"""
This module contains the composite Transformer model.
"""
import torch
import torch.nn as nn

from nlpmodels.models.transformer_blocks import sublayers, attention, decoder, encoder
from nlpmodels.utils.elt.transformer_batch import TransformerBatch


class Transformer(nn.Module):
    """
    The O.G. Transformer model from "Attention is All You Need" (2017).

    This is an encoder-decoder style architecture.

    The hyper-params are derived from the paper's specifications.

    Derived in part from logic found in "Annotated Transformer":
    https://nlp.seas.harvard.edu/2018/04/03/attention.html.
    """

    def __init__(self, source_vocab_size: int,
                 target_vocab_size: int,
                 num_layers_per_stack: int = 6,
                 dim_model: int = 512,
                 dim_ffn: int = 2048,
                 num_heads: int = 8,
                 max_length: int = 1000,
                 dropout: float = 0.1):
        """
        Args:
            source_vocab_size (int): size of the source vocabulary.
            target_vocab_size (int): size of the target vocabulary.
            num_layers_per_stack (int): number of sequential encoder/decoder layers.
            dim_model (int): size of the embedding space.
            dim_ffn (int): size of the residual/skip-connection hidden layer.
            num_heads (int): number of simultaneous attention heads calculated during attention.
            max_length (int): the max_seq_length of the input/output sequencies.
            dropout (float): Hyper-parameter used in drop-out regularization in training.
        """
        super(Transformer, self).__init__()

        # (1) Read src/input embeddings
        self._input_embeddings = sublayers.NormalizedEmbeddingsLayer(source_vocab_size, dim_model)
        # (2) calculate src/input pe
        self._input_pe = sublayers.PositionalEncodingLayer(dim_model, dropout, max_length)

        # (3) Pass features through encoder block
        self._encoder_block = encoder.CompositeEncoder(
            encoder.EncoderBlock(max_length,
                                 attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 sublayers.PositionWiseFFNLayer(dim_model, dim_ffn), dropout),
            num_layers_per_stack)

        # (4) calculate target/output embeddings
        self._output_embeddings = sublayers.NormalizedEmbeddingsLayer(target_vocab_size, dim_model)
        # (5) calculate target/output pe
        self._output_pe = sublayers.PositionalEncodingLayer(dim_model, dropout, max_length)

        # (6) Pass encoder output + output embedding to  decoder block
        self._decoder_block = decoder.CompositeDecoder(
            decoder.DecoderBlock(max_length,
                                 attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 sublayers.PositionWiseFFNLayer(dim_model, dim_ffn),
                                 dropout),
            num_layers_per_stack)

        # (7) put through final linear layer
        self._final_linear = nn.Linear(dim_model, target_vocab_size)
        # (8) compute log(softmax) only because KL divergence expects log probabilities
        # From KLDivergence documents:
        # "As with NLLLoss, the input given is expected to contain log-probabilities
        # and is not restricted to a 2D Tensor."
        self._final_softmax = nn.LogSoftmax(dim=-1)

        # Xavier norm all the parameters that are not fixed
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize all parameters to be trained using Xavier Uniform.
        Note: parameters added to buffer will not be affected.
        """
        for parameter in self.parameters():
            if parameter.dim() - 1:
                nn.init.xavier_uniform_(parameter)

    def _encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate all the layers in the encoder side of the model.
        Args:
            src (torch.Tensor):
                matrix of (batch_size, max_seq_length) size from source sequence.
            src_mask (torch.Tensor):
                matrix of (batch_size, max_seq_length) masking source so model doesn't see padding.

        Returns:
            encoder "memory" matrix of (batch_size, max_seq_length, dim_model) for decoder block.
        """
        embeddings = self._input_embeddings(src)
        pos_encoding = self._input_pe(embeddings)

        return self._encoder_block(pos_encoding, src_mask)

    def _decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor,
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate all the layers in the decoder side of the model.
        Args:
            memory (torch.Tensor):
                matrix of (batch_size, max_seq_length, dim_model) size from encoder output.
            src_mask (torch.Tensor):
                matrix of (batch_size, max_seq_length) masking source so model doesn't see padding.
            tgt (torch.Tensor):
                matrix of (batch_size, max_seq_length) size of target sequence.
            tgt_mask (torch.Tensor):
                matrix of (batch_size, max_seq_length) masking target  Masking target so model
                doesn't see padding or next sequential values.

        Returns:
            encoder "memory" matrix of (batch_size, max_seq_length, dim_model) for decoder block.
        """
        embeddings = self._output_embeddings(tgt)
        pos_encoding = self._output_pe(embeddings)

        return self._decoder_block(pos_encoding, memory, src_mask, tgt_mask)

    def forward(self, data: TransformerBatch) -> torch.Tensor:
        """
        Main call of transformer model. Pass through encoder-decoder architecture.

        Args:
            data(TransformerBatch):
                Class of batch data containing source, source_mask, target, target_mask.
        Returns:
            output matrix of probabilities of size (batch_size,max_seq_length,target_vocab_size).
        """

        encode = self._encode(data.src, data.src_mask)
        decode = self._decode(encode, data.src_mask, data.tgt, data.tgt_mask)

        y_hat = self._final_softmax(self._final_linear(decode))

        return y_hat

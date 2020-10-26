from nlpmodels.models.transformer_blocks import sublayers,attention,decoder,encoder
import torch.nn as nn
from copy import deepcopy
from nlpmodels.utils.transformer_batch import TransformerBatch


class Transformer(nn.Module):
    """
    The O.G. Transformer model from "Attention is All You Need" (2017).

    This is an encoder-decoder style architecture.

    The hyper-params are derived from the paper's specifications.
    """

    def __init__(self, source_vocab_size: int, target_vocab_size: int, num_layers_per_stack: int = 6,
                 dim_model: int = 512, dim_ffn: int = 2048, num_heads: int = 8,
                 max_length: int = 1000, dropout: float = 0.1):
        """
           Args:
                source_vocab_size (int):
                target_vocab_size (int):
                num_layers_per_stack (int):
                dim_model (int):
                dim_ffn (int):
                num_heads (int):
                max_length (int):
                dropout (float):
        """
        super(Transformer, self).__init__()

        # (1) Read src/input embeddings
        self._input_embeddings = sublayers.NormalizedEmbeddingsLayer(source_vocab_size,dim_model)
        # (2) calculate src/input pe
        self._input_pe = sublayers.PositionalEncodingLayer(dim_model, dropout, max_length)

        # (3) Pass features through encoder block
        self._encoder_block = encoder.CompositeEncoder(
            #dim_model
            encoder.EncoderBlock(max_length, attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 sublayers.PositionWiseFFNLayer(dim_model, dim_ffn), dropout), num_layers_per_stack)

        # (4) calculate target/output embeddings
        self._output_embeddings = sublayers.NormalizedEmbeddingsLayer(target_vocab_size,dim_model)
        # (5) calculate target/output pe
        self._output_pe = sublayers.PositionalEncodingLayer(dim_model, dropout, max_length)

        # (6) Pass encoder output + output embedding to  decoder block
        self._decoder_block = decoder.CompositeDecoder(
            #dim_model
            decoder.DecoderBlock(max_length, attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 attention.MultiHeadedAttention(num_heads, dim_model, dropout),
                                 sublayers.PositionWiseFFNLayer(dim_model, dim_ffn),
                                 dropout), num_layers_per_stack)

        # (7) put through final linear layer
        self._final_linear = nn.Linear(dim_model, target_vocab_size)
        # (8) compute softmax
        self._final_softmax = nn.LogSoftmax(dim=-1)

        # Xavier norm all the parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize all parameters to be trained using Xavier Uniform.
        Note: parameters added to buffer will not be affected.
        """
        for p in self.parameters():
            if p.dim() - 1:
                nn.init.xavier_uniform_(p)

    def _encode(self, src, src_mask):

        embeddings = self._input_embeddings(src)
        pos_encoding = self._input_pe(embeddings)

        return self._encoder_block(pos_encoding, src_mask)

    def _decode(self, memory, src_mask, tgt, tgt_mask):

        embeddings = self._output_embeddings(tgt)
        pos_encoding = self._output_pe(embeddings)

        return self._decoder_block(pos_encoding, memory, src_mask, tgt_mask)

    def forward(self, data: TransformerBatch):
        """
        Main call of transformer model. Pass through encoder-decoder architecture.
        """

        encode = self._encode(data.src, data.src_mask)
        decode = self._decode(encode, data.src_mask, data.tgt, data.tgt_mask)

        return self._final_softmax(self._final_linear(decode))


if __name__ == '__main__':
    # TESTING THE MODEL
    from nlpmodels.utils import train, utils, transformer_dataset
    from argparse import Namespace
    utils.set_seed_everywhere()
    args = Namespace(
        # Model hyper-parameters
        num_layers_per_stack=1,  # original value = 6
        dim_model=512,
        dim_ffn=2048,
        num_heads=8,
        max_sequence_length=20,  # original value = 1000
        dropout=0.1,
        # Label smoothing loss function hyper-parameters
        label_smoothing=0.1,
        # Training hyper-parameters
        num_epochs=30,
        learning_rate=0.0,
        batch_size=50,
    )

    train_dataloader, vocab_source, vocab_target = transformer_dataset.TransformerDataset.get_training_dataloader(args)
    vocab_source_size = len(vocab_source)
    vocab_target_size = len(vocab_target)
    model = Transformer(vocab_source_size, vocab_target_size,
                                    args.num_layers_per_stack, args.dim_model,
                                    args.dim_ffn, args.num_heads, args.max_sequence_length,
                                    args.dropout)
    trainer = train.TransformerTrainer(args, vocab_target_size, vocab_target.eos_index, model, train_dataloader)
    trainer.run()
from nlpmodels.utils.transformer_dataset import TransformerDataset
from nlpmodels.utils.vocabulary import NLPVocabulary
import torch
from argparse import Namespace


def test_padded_string_to_integer_conversion():

    token_list = [["the","cow","jumped","over","the","moon"]]
    vocab = NLPVocabulary.build_vocabulary(token_list)
    max_seq_length = 10
    padded_integers = TransformerDataset.padded_string_to_integer(token_list, max_seq_length, vocab)
    assert padded_integers[0] == [3,4,5,6,3,7,2,0,0,0]


def test_transformer_dataset_returns_two_tensors():

    src_tokens = [["the", "cow", "jumped", "over", "the", "moon"], ["the","british","are","coming"]]
    tgt_tokens = [["la" ,"vache" ,"a" ,"sauté", "sur", "la", "lune"], ["les", "britanniques", "arrivent"]]
    dictionary_source = NLPVocabulary.build_vocabulary(src_tokens)
    dictionary_target = NLPVocabulary.build_vocabulary(tgt_tokens)
    max_seq_length = 20
    src_padded = TransformerDataset.padded_string_to_integer(src_tokens, max_seq_length, dictionary_source)
    tgt_padded = TransformerDataset.padded_string_to_integer(tgt_tokens, max_seq_length + 1, dictionary_target)

    dataset = TransformerDataset(list(zip(src_padded,tgt_padded)),dictionary_source)

    batch = dataset[1]

    assert type(batch[0]) == torch.Tensor and type(batch[1]) == torch.Tensor and len(batch) == 2


def test_training_dataloader_batchsize():
    args_test = Namespace(
            batch_size=100,
            max_sequence_length=10,
        )
    transformer_dataloader = TransformerDataset.get_training_dataloader(args_test)
    batch = next(iter(transformer_dataloader[0]))
    src_batch, tgt_batch = batch
    assert src_batch.size() == torch.Size([args_test.batch_size,args_test.max_sequence_length]) and tgt_batch.size() == torch.Size([args_test.batch_size,args_test.max_sequence_length+1])

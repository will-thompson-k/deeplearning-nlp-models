from nlpmodels.utils.vocabulary import NLPVocabulary
from nlpmodels.utils.transformer_dataset import TransformerDataset
from nlpmodels.utils.transformer_batch import TransformerBatch
import torch


def test_transformer_batch_dimensions():
    src_tokens = [["the", "cow", "jumped", "over", "the", "moon"], ["the", "british", "are", "coming"]]
    tgt_tokens = [["la", "vache", "a", "sauté", "sur", "la", "lune"], ["les", "britanniques", "arrivent"]]
    batch_size = len(src_tokens)
    dictionary_source = NLPVocabulary.build_vocabulary(src_tokens)
    dictionary_target = NLPVocabulary.build_vocabulary(tgt_tokens)
    max_seq_length = 20
    src_padded = TransformerDataset.padded_string_to_integer(src_tokens, max_seq_length, dictionary_source)
    tgt_padded = TransformerDataset.padded_string_to_integer(tgt_tokens, max_seq_length + 1, dictionary_target)
    batched_object_data = TransformerBatch(torch.LongTensor(src_padded), torch.LongTensor(tgt_padded))

    # test dimensions
    assert batched_object_data.src.size() == torch.Size([batch_size, max_seq_length])
    assert batched_object_data.src_mask.size() == torch.Size([batch_size, 1, max_seq_length])
    assert batched_object_data.tgt.size() == torch.Size([batch_size, max_seq_length])
    assert batched_object_data.tgt_y.size() == torch.Size([batch_size, max_seq_length])
    assert batched_object_data.tgt_mask.size() == torch.Size([batch_size, max_seq_length, max_seq_length])


def test_transformer_batch_src_masking():
    src_tokens = [["the", "cow", "jumped", "over", "the", "moon"], ["the", "british", "are", "coming"]]
    tgt_tokens = [["la", "vache", "a", "sauté", "sur", "la", "lune"], ["les", "britanniques", "arrivent"]]
    batch_size = len(src_tokens)
    dictionary_source = NLPVocabulary.build_vocabulary(src_tokens)
    dictionary_target = NLPVocabulary.build_vocabulary(tgt_tokens)
    max_seq_length = 10
    src_padded = TransformerDataset.padded_string_to_integer(src_tokens, max_seq_length, dictionary_source)
    tgt_padded = TransformerDataset.padded_string_to_integer(tgt_tokens, max_seq_length + 1, dictionary_target)
    batched_object_data = TransformerBatch(torch.LongTensor(src_padded), torch.LongTensor(tgt_padded))

    # test masking
    # include EOS token
    assert torch.equal(batched_object_data.src_mask[0, :],
                       torch.BoolTensor([[True, True, True, True, True, True, True, False, False, False]]))
    assert torch.equal(batched_object_data.src_mask[1, :],
                       torch.BoolTensor([[True, True, True, True, True, False, False, False, False, False]]))


def test_transformer_batch_tgt_masking():
    src_tokens = [["the", "cow", "jumped", "over", "the", "moon"], ["the", "british", "are", "coming"]]
    tgt_tokens = [["la", "vache", "a", "sauté", "sur", "la", "lune"], ["les", "britanniques", "arrivent"]]
    batch_size = len(src_tokens)
    dictionary_source = NLPVocabulary.build_vocabulary(src_tokens)
    dictionary_target = NLPVocabulary.build_vocabulary(tgt_tokens)
    max_seq_length = 10
    src_padded = TransformerDataset.padded_string_to_integer(src_tokens, max_seq_length, dictionary_source)
    tgt_padded = TransformerDataset.padded_string_to_integer(tgt_tokens, max_seq_length + 1, dictionary_target)
    batched_object_data = TransformerBatch(torch.LongTensor(src_padded), torch.LongTensor(tgt_padded))

    # test masking of target: blend of padding and auto-regressive masking

    # full mask at last part of sequence
    assert torch.equal(batched_object_data.tgt_mask[0, 9, :],
                       torch.BoolTensor([True, True, True, True, True, True, True, True, False, False]))
    # should be just the very first item
    assert torch.equal(batched_object_data.tgt_mask[0, 0, :],
                       torch.BoolTensor([True, False, False, False, False, False, False, False, False, False]))

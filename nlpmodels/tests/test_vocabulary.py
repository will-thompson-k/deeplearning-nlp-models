from nlpmodels.utils import vocabulary


def test_vocabulary_expected_tokens():
    data = [
        ['wall', 'st', 'bears', 'claw', 'back', 'into', 'the', 'black', 'reuters', 'reuters', 'short', 'sellers',
         'wall', 'street', 'dwindling', 'band', 'of', 'ultra', 'cynics', 'are', 'seeing', 'green', 'again'],
        ['carlyle', 'looks', 'toward', 'commercial', 'aerospace', 'reuters', 'reuters', 'private', 'investment', 'firm',
         'carlyle', 'group', 'which', 'has', 'reputation', 'for', 'making', 'well', 'timed', 'and', 'occasionally',
         'controversial', 'plays', 'in', 'the', 'defense', 'industry', 'has', 'quietly', 'placed', 'its', 'bets', 'on',
         'another', 'part', 'of', 'the', 'market']
    ]
    vocab = vocabulary.NLPVocabulary.build_vocabulary(data)

    # should be 52 tokens in the dictionary + <MASK> + <UNK> + <EOS>
    assert len(vocab) == 55


def test_vocabulary_get_unk_token():
    vocab = vocabulary.NLPVocabulary()
    unk_index = vocab.lookup_token(vocab.unk_token)

    assert unk_index == vocab.unk_index


def test_vocabulary_token_return_unknown_token():
    vocab = vocabulary.NLPVocabulary()
    x = vocab.lookup_token("hey")

    assert x == vocab.unk_index

from nlpmodels.utils import tokenizer


def test_tokenizer_removes_punctuation():
    text = [
        'Hi. How2 are you doing??    ',
        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    ]
    exp_text = [['hi', 'how', 'are', 'you', 'doing']]
    tokenized_text = tokenizer.tokenize_corpus_basic(text)

    assert tokenized_text == exp_text

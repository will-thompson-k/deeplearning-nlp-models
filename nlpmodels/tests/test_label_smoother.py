import torch

from nlpmodels.utils.label_smoother import LabelSmoothingLossFunction


def test_label_smoothing():
    vocab_size = 5
    label_smoothing_loss_function = LabelSmoothingLossFunction(vocab_size=vocab_size, padding_idx=0, smoothing=0.1)

    # pass it a set of vocab probabilities in vector (batch_size,vocab_size) size of probabilities
    yhat = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],  # very confident in y at index = 2
                              [0, 0.9, 0.0, 0.1, 0],  # very confident in y at index = 1
                              [0, 0.3, 0.3, 0.4, 0]])  # uniform distribution, y at index = 3

    # this is the target_index in each batch
    y = torch.LongTensor([2, 1, 3])

    # returns a (batch_size,vocab_size) size with peak probability where target value is,
    # surrounded by non-zero probabilities for others
    y_smooth = label_smoothing_loss_function._compute_label_smoothing(y, yhat)

    # test they match in size
    assert y_smooth.size() == yhat.size()

    # test that the peak of the distribution is the actual target (y) for each batch
    assert y_smooth.argmax(dim=1).equal(y)

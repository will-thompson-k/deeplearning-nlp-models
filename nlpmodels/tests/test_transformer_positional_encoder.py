from nlpmodels.models.transformer_blocks import sublayers
import torch
import numpy as np
import matplotlib.pyplot as plt


def test_transformer_positional_encoding():

    pos_encoder = sublayers.PositionalEncodingLayer(dim_model=20, dropout=0, max_length=100)
    # passing in (batch_size=1, max_seq_length=100, dim_model=20) size tensor
    yhat = pos_encoder.forward((torch.zeros(1, 100, 20)))

    # plot along max_length, wave numbers 1:10
    # plt.plot(np.arange(100), yhat[0, :, 1:10].data.numpy())
    # plt.legend(["wave numbers %d" %w for w in [1,2,3,4,5,6,7,8,9]])
    # plt.show()

    # test that these are sin, cos pairs through identity (sin^2 + cos^2==1)
    # i.e.,     all(yhat[0, :, 1]**2 + yhat[0, :, 0]**2) == 1.0
    wave_number = range(0,20,2)
    for i in wave_number:
        assert all(yhat[0, :, i]**2 + yhat[0, :, i + 1]**2)==1.0
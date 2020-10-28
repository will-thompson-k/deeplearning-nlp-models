from nlpmodels.utils.optims import NoamOptimizer
# import matplotlib.pyplot as plt
import numpy as np


def test_noam_optimization():
    optimizer = NoamOptimizer(dim_model = 512, factor =1, warm_up = 4000,optimizer=None)
    optimizer_learning_rates = [optimizer.calc_lr(i) for i in range(1, 10000)]

    # to plot the LR changes
    # plt.plot(np.arange(1, 10000),optimizer_learning_rates)
    # plt.show()

    # test monotonic increase in LR up until warm_up followed by decay.
    optimizer_learning_rates = np.array(optimizer_learning_rates)
    diffs = np.diff(optimizer_learning_rates)
    assert all(diffs[0:4000-1]> 0) == True and all(diffs[4000:]< 0) == True

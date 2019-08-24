"""
This code is copied from uber atari-model-zoo projects. Link:

https://github.com/uber-research/atari-model-zoo/blob/master/
dimensionality_reduction/process_helper.py
"""

# from collections import OrderedDict
import numpy as np
from sklearn import decomposition, manifold
# import atari_zoo
# from atari_zoo import MakeAtariModel
# import urllib.request
# import os

# method= {
#     "name": "pca_tsne",
#     "pca_dim": 50,
#     "perplexity": 30,
#     "n_iter": 3000
# }

import seaborn as sns

DEFAULT_METHOD = {
    "name": "pca_tsne",
    "pca_dim": 50,
    "perplexity": 30,
    "n_iter": 1000
}


def reduce_dim(data, method=DEFAULT_METHOD):
    print(method)

    print("Reducing ...")

    perplexity = 30
    n_iter = 1000
    pca_dim = 50
    if 'tsne' in method['name']:
        if 'perplexity' in method.keys():
            perplexity = method['perplexity']
        if 'n_iter' in method.keys():
            n_iter = method['n_iter']
        if 'pca_dim' in method.keys():
            pca_dim = method['pca_dim']

    # print(perplexity, n_iter, pca_dim)
    # if method['name'] == 'tsne':
    #     print('running tsne')
    #     result = manifold.TSNE(
    #         n_components=2,
    #         perplexity=perplexity,
    #         verbose=2,
    #         random_state=0,
    #         n_iter=n_iter
    #     ).fit_transform(data)
    # elif method['name'] == 'pca':
    #     print('running pca')
    #     result = decomposition.PCA(2).fit_transform(data)
    if method['name'] == 'pca_tsne':
        print('running pca')
        pca_result = decomposition.PCA(pca_dim).fit_transform(data)
        print('running tsne')
        result = manifold.TSNE(
            n_components=2,
            perplexity=perplexity,
            verbose=2,
            random_state=0,
            n_iter=n_iter
        ).fit_transform(pca_result)
    # elif method['name'] == 'debug':
    #     print('running debug')
    #     nrow, ncol = data.shape
    #     idx_last_x, idx_last_y = int(ncol / 2 - 1), -1
    #     result = np.hstack(
    #         (
    #             data[:, idx_last_x].reshape(nrow, 1),
    #             data[:, idx_last_y].reshape(nrow, 1)
    #         )
    #     )
    else:
        raise NotImplementedError

    print(
        'Reduction Completed! data.shape={} result.shape={}'.format(
            data.shape, result.shape
        )
    )
    return result

def display(result):

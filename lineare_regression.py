"""
Linear Regression Example

Implement Linear Regression for Beijing House Price Problem
"""
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_boston
from matplotlib.animation import FuncAnimation
import re

"""
Part-01: Linear Regression
"""

housing_price = load_boston()
dataframe = pd.DataFrame(housing_price['data'])
dataframe.columns = housing_price['feature_names']
dataframe['price'] = housing_price['target']

# sns.heatmap(dataframe.corr(), annot=True, fmt='.1f')
# plt.show()

print(dataframe.columns)

rm = dataframe['RM']
lst = dataframe['LSTAT']
target = dataframe['price']


def model(x, w, b):
    return np.dot(x, w.T) + b


def loss(yhat, y):
    return np.mean( (yhat - y) ** 2)


def partial_w(x1, x2, y, yhat):
    return np.array([2 *np.mean((yhat - y) * x1), 2 * np.mean((yhat - y)  * x2)])


def partial_b(x1, x2, y, yhat):
    return 2 * np.mean((yhat - y))


w = np.random.random_sample((1, 2))
print(w)
b = 0
alpha = 1e-5

epoch = 200
history = []

history_k_b_loss = []

for e in range(epoch):
    losses = []
    for batch in range(len(rm)):
        random_index = random.choice(range(len(rm)))

        x1, x2 = rm[random_index], lst[random_index]
        y = target[random_index]

        yhat = model(np.array([x1, x2]), w, b)
        loss_v = loss(yhat, y)

        w = w - partial_w(x1, x2, y, yhat) * alpha
        b = b - partial_b(x1, x2, y, yhat) * alpha

        losses.append(loss_v)

        history_k_b_loss.append((w, b, loss_v))

        if batch % 100 == 0:
            print('Epoch: {}, Batch: {}, loss: {}'.format(e, batch, np.mean(losses)))

    history.append(np.mean(losses))



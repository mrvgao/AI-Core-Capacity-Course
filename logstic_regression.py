import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_boston
from matplotlib.animation import FuncAnimation
import re

housing_price = load_boston()
dataframe = pd.DataFrame(housing_price['data'])
dataframe.columns = housing_price['feature_names']
dataframe['price'] = housing_price['target']

rm = dataframe['RM']
lst = dataframe['LSTAT']
price = dataframe['price']
print(np.percentile(price, 66))

# plt.hist(target)
# plt.show()

dataframe['expensive'] = dataframe['price'].apply(lambda p: int(p > np.percentile(price, 66)))
expensive = dataframe['expensive']

# print(dataframe.head())
print(dataframe['expensive'])


def logistic(x):
    return 1 / (1 + np.exp(-x))


def model(x, w, b):
    return logistic(np.dot(x, w.T) + b)


def loss(yhat, y):
    return -1 * np.sum(y*np.log(yhat) + (1 - y) * np.log(1 - yhat))


def partial_w(x1, x2, y, yhat):
    return np.array([np.sum((yhat - y) * x1), np.sum((yhat - y) * x2)])


def partial_b(x1, x2, y, yhat):
    return np.sum(yhat - y)


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
        y = expensive[random_index]

        yhat = model(np.array([x1, x2]), w, b)
        loss_v = loss(yhat, y)

        w = w - partial_w(x1, x2, y, yhat) * alpha
        b = b - partial_b(x1, x2, y, yhat) * alpha

        losses.append(loss_v)

        history_k_b_loss.append((w, b, loss_v))

        if batch % 100 == 0:
            print('Epoch: {}, Batch: {}, loss: {}'.format(e, batch, np.mean(losses)))

    history.append(np.mean(losses))

predicated = [model(np.array([x1, x2]), w, b) for x1, x2 in zip(rm, lst)]
true = expensive


def accuracy(y, yhat):
    return sum(1 if i == j else 0 for i, j in zip(y, yhat)) / len(y)


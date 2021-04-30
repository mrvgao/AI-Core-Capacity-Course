"""
Linear Regression: 实现了回归，其中包括线性函数的定义，为什么要用线性函数，loss的意义，梯度下降的意义，stochastic gradient descent
Use Boston house price dataset.
北京2020年房价的数据集，为什么我没有用北京房价的数据集呢？
Boston: room size, subway, highway, crime rate 有一个比较明显的关系，所以就观察关系比较容易
北京的房价：！远近，！房况 ==》 学区！！！！ => 非常贵 海淀区
Harder than deep learning:
    1. compiler
    2. programming language & automata
    3. computer graphic
    4. complexity system
    5. computing complexity
    6. operating system
"""


from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from linear_regression_course import train


dataset = load_boston()
data = dataset['data']
target = dataset['target']
columns = dataset['feature_names']

dataframe = pd.DataFrame(data)
dataframe.columns = columns
dataframe['price'] = target

# print(dataframe.corr()) # show the correlation of dataframe variables
# correlation => 如果一个值的增大，会引起另外一个值一定增大，而且是定比例增大 相关系数就越接近于1
# correlation => 0 就是两者之间没有任何关系
# correlation => -1 一个值增大 另外一个值一定减小 而且减小是成相等比例的

# sns.heatmap(dataframe.corr())
# plt.show()

# RM：小区平均的卧室个数
# LSTAT: 低收入人群在周围的比例

rm = dataframe['RM']
lstat = dataframe['LSTAT']
price = dataframe['price']
greater_then_most = np.percentile(price, 66)
dataframe['expensive'] = dataframe['price'].apply(lambda p: int(p > greater_then_most))
target = dataframe['expensive']

print(dataframe[:20])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def model(x, w, b):
    return sigmoid(np.dot(x, w.T) + b)


def loss(yhat, y):
    return -np.sum(y*np.log(yhat) + (1 - y)*np.log(1 - yhat))


def partial_w(x, y, yhat):
    return np.array([np.sum((yhat - y) * x[0]), np.sum((yhat - y) * x[1])])


def partial_b(x, y, yhat):
    return np.sum((yhat - y))


model, w, b, losses = train(model, target,loss, partial_w, partial_b)

random_test_indices = np.random.choice(range(len(rm)), size=100)
decision_boundary = 0.5

for i in random_test_indices:
    x1, x2, y = rm[i], lstat[i], target[i]
    predicate = model(np.array([x1, x2]), w, b)
    predicate_label = int(predicate > decision_boundary)

    print('RM: {}, LSTAT: {}, EXPENSIVE: {}, Predicated: {}'.format(x1, x2, y, predicate_label))

# 剩下一件事情，就是要检查我们这个模型的准确度到底如何！！
"""
如何衡量模型的好坏：
1. accuracy 准确度
2. precision 精确度
3. recall 召回率
4. f1, f2 score
5. AUC-ROC 曲线
引出一个非常非常重要的概念： =》 过拟合 和 欠拟合 （over-fitting and under-fitting）
整个机器学习的过程，就是在不断的进行过拟合和欠拟合的调整！
"""


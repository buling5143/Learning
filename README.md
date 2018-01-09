# Learning

神经网络NN算法(应用篇)

# 代码
coding: utf-8

import numpy as np

 

def tanh(x):

    return np.tanh(x)

 

def tanh_deriv(x):

    """

    tanh的导数

    """

    return 1.0 - np.tanh(x) *
np.tanh(x)

 

def logistic(x):

    return 1.0 / (1 + np.exp(-x))

 

def logistic_deriv(x):

    """

    逻辑函数的导数

    """

    fx = logistic(x)

    return fx * (1 - fx)



设计神经网络的类结构

类的结构大致如下：



class
NeuralNetwork(object):

    def __init__(self, layers, activation='tanh'):

        pass

    def fit(self, X, Y,
learning_rate=0.2, epochs=10000):

        pass

    def predict(self, x);

        pass



构造函数

在构造函数中，需要确定神经网络的层数，每层的个数，从而确定单元间的权重规格和单元的偏向。



def __init__(self, layers, activation='logistic'):

    """

    :param layers: 层数，如[4, 3, 2] 表示两层len(list)-1,(因为第一层是输入层，，有4个单元)，

    第一层有3个单元，第二层有2个单元

    :param activation:

    """

    if activation == 'tanh':

        self.activation = tanh

        self.activation_deriv =
tanh_deriv

    elif activation == 'logistic':

        self.activation =
logistic

        self.activation_deriv =
logistic_deriv

 

    # 初始化随机权重

    self.weights = []

    for i in range(len(layers) - 1):

        tmp = (np.random.random([layers[i],
layers[i + 1]]) * 2 - 1) * 0.25

        self.weights.append(tmp)

 

    # 偏向随机初始化

    self.bias = []

    for i in range(1, len(layers)):

        self.bias.append((np.random.random(layers[i])* 2 - 1) * 0.25)



其中，layers 参数表示神经网络层数以及各个层单元的数量，例如下图这个模型：



这个模型就对应了 layers = [4, 3, 2] 。
这是一个 2 层，即 len(layers) - 1 层的神经网络。
输入层 4 个单元，其他依次是 3 ，2 。
权重是表示层之间单元与单元之间的连接。
因此权重也有 2 层。第一层是一个 4 x 3 的矩阵，即 layers[0] x layers[1]。同理，偏向也有 2 层，第一层个数 3 ，即leyers[1] 。

利用 np.random.random([m, n]) 来创建一个 m x n 的矩阵。在这个神经网络的类中，初始化都随机 -0.25 到 0.25 之间的数。

训练函数

在神经网络的训练中，需要先设定一个训练的终止条件，在理论篇介绍了3种停止条件，这边使用的是 达到预设一定的循环次数 。
每次训练是从样本中随机挑选一个实例进行训练，将这个实例的预测结果和真实结果进行对比，再进行反向传播得到各层的误差，然后再更新权重和偏向：



def fit(self, X, y,
learning_rate=0.2, epochs=10000):

    X = np.atleast_2d(X)

    y = np.array(y)

    # 随即梯度

    for k in range(epochs):

        i = np.random.randint(X.shape[0])

        a = [X[i]]   # 随即取某一条实例

        for j in range(len(self.weights)):

            a.append(self.activation(np.dot(a[j],
self.weights[j]) + self.bias[j] ))

        errors = y[i] - a[-1]

        deltas = [errors * self.activation_deriv(a[-1]) ,]  # 输出层的误差

        # 反向传播，对于隐藏层的误差

        for j in range(len(a) - 2, 0, -1):

            tmp = np.dot(deltas[-1], self.weights[j].T) * self.activation_deriv(a[j])

            deltas.append(tmp)

        deltas.reverse()

 

        # 更新权重

        for j in range(len(self.weights)):

            layer = np.atleast_2d(a[j])

            delta = np.atleast_2d(deltas[j])

            self.weights[j] +=learning_rate * np.dot(layer.T, delta)

 

        # 更新偏向

        for j in range(len(self.bias)):

            self.bias[j] +=learning_rate * deltas[j]



参数 learning_rate 表示学习率， epochs 表示设定的循环次数。

预测

预测就是将测试实例从输入层传入，通过正向传播，最后返回输出层的值即可：



def predict(self, row):

    a = np.array(row) # 确保是 ndarray 对象

    for i in range(len(self.weights)):

        a = self.activation(np.dot(a,self.weights[i]) + self.bias[i])

    return a

手写数字识别

手写数字数据集来自 sklearn ，其中由1797个图像组成，其中每个图像是表示手写数字的 8x8 像素图像：


可以推出，这个神经网络的输入层将有 64 个输入单元，分类结果是 0~9 ，因此输出层有10个单元，构造为：

nn =NeuralNetwork(layers=[64, 100, 10])



载入数据集


from sklearn import datasets

digits =datasets.load_digits()

X = digits.data

y = digits.target



拆分成训练集和数据集，分类结果离散化：



from sklearn.model_selection import
train_test_split

from sklearn.preprocessing
import LabelBinarizer

# 拆分为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y)

 

# 分类结果离散化

labels_train =
LabelBinarizer().fit_transform(y_train)

labels_test =
LabelBinarizer().fit_transform(y_test)



训练

取训练集进行训练：



nn.fit(X_train,
labels_train)



测试模型



# 收集测试结果

predictions = []

for i in range(X_test.shape[0]):

    o = nn.predict(X_test[i] )

    predictions.append(np.argmax(o))

 

# 打印对比结果

from sklearn.metrics import confusion_matrix,
classification_report

print
(confusion_matrix(y_test, predictions) )

print
(classification_report(y_test, predictions))



利用测试集对模型进行测试，得到 confusion_matrix 的打印结果：

[[51  0  0  0  1  0  0  0  0  0]

 [ 0 44  0  1  0  0  0  0  0  1]

 [ 0  0 34  0  0  0  0  0  0  0]

 [ 0  0  0 41  0  0  0  0  0  0]

 [ 0  0  0  0 39  0  0  0  0  0]

 [ 0  0  0  0  0 50  0  0  0  0]

 [ 0  1  0  0  0  0 42  0  1  0]

 [ 0  0  0  0  0  0  0 59  0  0]

 [ 0  4  1  3  0  0  0  0 31  0]

 [ 1  0  0  0  0  1  0  0  0 44]]



行列中，表示预测值与真实值的情况，比方预测值为0，真实值也为0，那么就在 [0][0] 计数 1。因此这个对角线计数越大表示预测越准确。

另一个报告 classification_report :



             precision    recall   f1-score   support

 

          0       0.98      0.98      0.98        52

          1       0.90      0.96      0.93        46

          2       0.97      1.00      0.99        34

          3       0.91      1.00      0.95        41

          4       0.97      1.00      0.99        39

          5       0.98      1.00      0.99        50

          6       1.00      0.95      0.98        44

          7       1.00      1.00      1.00        59

          8       0.97      0.79      0.87        39

          9       0.98      0.96      0.97        46

 

avg / total       0.97      0.97      0.97       450



正确率在 97% ，相当不错的。


 


##神经网络
* 机器学习的目的是为了训练出一组参数使得训练模型最好地学习到一种映射关系，使得训练模型通过这种关系能很好地去预测新的样本
###感知机（Perceptron）
* **感知机就是一种线性模型**，这是早期的网络模型，通过线性映射可以解决一些简单的视觉分类任务，比如区分三角形，圆形，矩形等。

感知机模型如下：
![](技术学习图片\21.png)
它接受长度为$n$的一维向量$\boldsymbol{x}=\left[x_{1}, x_{2}, \ldots, x_{n}\right]$，每个输入节点通过权值为$\boldsymbol{w_i,i\in[1,n]}$连接相加得：$\boldsymbol{z=w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{n} x_{n}+b}$，其中$b$称为感知机的偏置（bias），$\boldsymbol{z}$称为感知机的==净活性值==，写成向量形式：$$\boldsymbol{z=\boldsymbol{w}^{T} \boldsymbol{x}+b}$$
感知机并不能处理非线性的问题，因此要加入激活函数后得到==活性值==：$$\boldsymbol{a=\sigma(z)=\sigma\left(\boldsymbol{w}^{T} \boldsymbol{x}+b\right)}$$
此时的激活函数还是一些像是阶跃函数，符号函数等不连续的函数
###全连接层
* **全连接层在感知机的基础上，通过将不连续的阶跃函数换成其他平滑连续的激活函数，并通过堆叠多层网络层来增强网络的表达能力**

全连接层模型如下：
![](技术学习图片\22.png)
假设现有两个样本，$\boldsymbol{x^{1}=\left[x_{1}^{1}, x_{2}^{1}, x_{3}^{1}\right], \quad x^{2}=\left[x_{1}^{2}, x_{2}^{2}, x_{3}^{2}\right]}$,通过权重矩阵相乘连接，再与偏置相加，得到：$$\boldsymbol{\left[\begin{array}{cc}
o_{1}^{1} & o_{2}^{1} \\
o_{1}^{2} & o_{2}^{2}
\end{array}\right]=\left[\begin{array}{ccc}
x_{1}^{1} & x_{2}^{1} & x_{3}^{1} \\
x_{1}^{2} & x_{2}^{2} & x_{3}^{2}
\end{array}\right] @\left[\begin{array}{cc}
W_{11} & W_{12} \\
W_{21} & W_{22} \\
W_{31} & W_{32}
\end{array}\right]+\left[\begin{array}{cc}
b_{0} & b_{1}
\end{array}\right]}$$
**可以看出输出矩阵中的每个输出节点$\boldsymbol{o_i^j}$都与全部输入节点连接，这种网络层叫做全连接层**
####张量方式实现
~~~
import tensorflow as tf
X = tf.random.normal([2,784])
W1 = tf.Variable(tf.random.truncated_normal([784,256]))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(X,W1) + b1
o1 = tf.nn.relu(o1)
print(o1)
~~~
所用到的函数：
1. **tf.Variable(initializer，name)**
==参数initializer是初始化参数，name是可自定义的变量名称==
2. **tf.matmul(tensor_a,tensor_b)**
==批量矩阵相乘==
3. **tf.nn.relu(tensor，name=None)**
==Relu激活函数==
####层方式实现
全连接层本质上就是矩阵的相乘相加运算，可以直接通过张量来实现，但是TensorFlow中有更加高层，使用更方便的层实现方式：**layer.Dense(units，activation)**==只需要指定输出节点数units和激活函数类型即可。==

**运行“黑匣子”**：输入节点数将根据第一次运算时的输入shape确定，同时根据输入、输出节点数自动创建并初始化权值矩阵$\boldsymbol{W}$和偏置向量$\boldsymbol{b}$，使用非常方便。

运行代码：
~~~
import tensorflow as tf
from tensorflow.keras import layers
x = tf.random.normal([4,28*28])
fc = layers.Dense(512,activation=tf.nn.relu)
创建全连接层，指定输出节点数和激活函数
h1 = fc(x)
通过fc类完成一次全连接层的计算
print(h1)

OUT：
tf.Tensor(
[[0.4445603  0.         0.         ... 0.         0.         1.9244801 ]
 [0.37167704 0.5069377  0.03083919 ... 0.         2.0557394  0.        ]
 [0.         0.9400093  0.         ... 0.         1.136771   0.02230924]
 [2.4392984  0.14651556 0.         ... 1.9309815  0.         0.8839622 ]], shape=(4, 512), dtype=float32)
~~~
**layer.Dense函数**创建一层全连接层，返回一个类，向这个类传入输入数据x，输入节点数在传入时获取，接着在类的内部创建权值矩阵$\boldsymbol{W}$和$\boldsymbol{b}$，可以通过类内部的成员名**kernel**和**bias**来获取权值矩阵$\boldsymbol{W}$和$\boldsymbol{b}$，在优化参数时，需要获得网络的所有待优化的参数张量列表，可以通过类的**trainable_variables**来返回待优化参数列表，**non_trainable_variables**成员返回所有不需要优化的参数列表，**variable**返回所有内部张量列表
~~~
import tensorflow as tf
from tensorflow.keras import layers
x = tf.random.normal([4,28*28])
fc = layers.Dense(512,activation=tf.nn.relu)
h1 = fc(x)
print(h1,fc.kernel,fc.bias,fc.trainable_variables)
~~~
###神经网络
* ==前一层的输出节点数与当前层的输入节点数匹配，即可堆叠出任意层数的网络，这种由神经元构成的网络叫做神经网络==

全连接神经网络如下图：
![](技术学习图片\23.png)
####张量方式实现
~~~
import tensorflow as tf
from tensorflow.keras import layers
layer1

w1 = tf.Variable(tf.random.normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

layer2

w2 = tf.Variable(tf.random.normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

layer3

w3 = tf.Variable(tf.random.normal([128,64],stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))

output_layer

w4 = tf.Variable(tf.random.normal([64,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

x = tf.random.normal([10,784])
with tf.GradientTape() as tape:
    "梯度记录器"
    h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],256])
    h1 = tf.nn.relu(h1)
    h2 = h1@w2 + tf.broadcast_to(b2,[x.shape[0],128])
    h2 = tf.nn.relu(h2)
    h3 = h2@w3 + tf.broadcast_to(b3, [x.shape[0], 64])
    h3 = tf.nn.relu(h3)
    h4 = h3@w4 + tf.broadcast_to(b4,[x.shape[0],10])
gradients = tape.gradient([h1,h2,h3,h4],[w1,w2,w3,w4])
print(gradients)
~~~
将待优化张量用**tf.Variable**包裹起来，方便跟踪梯度的变化，而且==GradientTape==默认只监控由**tf.Variable创建的trainable=True属性（默认）的变量**，因此有时需要使用**watch**函数来引入监控梯度的变化，在使用TensorFlow自动求导功能计算梯度时，需要将前向计算过程放在**tf.GradientTape()环境**中，==从而利用GradientTape对象的gradient()方法自动求解参数的梯度，并用optimizers对象更新参数，**另外，默认情况下GradientTape的资源在调用gradient函数后就被释放，再次调用就无法计算了。所以如果需要多次计算梯度，需要开启GradientTape（persistent=True）属性**==
####层方式实现
~~~
import tensorflow as tf
from tensorflow import keras
x = tf.random.normal([10,728])
model = keras.Sequential([
    keras.layers.Dense(256,activation=tf.nn.relu),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(64,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=None)])
out = model(x)
print(out)
~~~
==通过Sequential容器封装成一个网络大类对象，调用大类的前向计算函数即可完成所有层的前向计算==

**从另一个角度来理解神经网络，它完成的是特征的维度变换的功能，比如4层的MNIST手写数字图片识别的全连接神经网络，它依次完成了$\boldsymbol{784 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 10}$的特征降维过程。原始的特征通常具有较高的维度，包含了很多底层特征及无用信息，通过神经网络的层层特征变换，将较高的维度降维到较低的维度，此时的特征一般包含了与任务强相关的高层特征信息，通过对这些特征进行简单的逻辑判定即可完成特定的任务**
####优化目标
* **神经网络从输入到输出的计算过程叫做前向传播过程，也是数据张量（tensor）从第一层，途径每个隐藏层（flow），直至输出层的过程，这也是TensorFlow框架名字意义所在，前向传播的最后一步就是要优化参数，通过目标函数得到误差，利用反向传播算法向后传播参数**

###输出层设计
* **网络的最后一层除了和所有的隐藏层一样，完成维度变换、特征提取的功能，还作为输出层使用，需要根据具体的任务场景来决定是否使用激活函数，以及使用什么类型的激活函数**

==根据任务场景的不同输出值也会不同，我们根据输出值二点区间范围来分类讨论：==
1. **$\boldsymbol{o \in R^d}$输出属于整个实数空间，或者某段普通的实数空间，比如函数值趋势的预测，年龄额预测问题**

2. **$\boldsymbol{o \in[0,1]}$输出值特别地落在$\boldsymbol{[0,1]}$的区间，如图片生成，图片像素值一般用$\boldsymbol{[0,1]}$表示；或者二分类问题的概率，如硬币正反面的概率预测问题**
3. **$\boldsymbol{o \in [0,1],\sum_io_i=1}$输出值落在$\boldsymbol{[0,1]}$的区间，并且所有输出值之和为1，常见的如多分类问题，如MNIST手写数字图片识别，图片属于10个类别的概率之和为1**
4. **$\boldsymbol{o \in [-1,1]}$输出值在$\boldsymbol{[-1,1]}$之间**
###误差计算
* 常见的误差计算函数有均方误差、交叉熵、KL散度、Hinge Loss函数等。

####均方差
将输出向量（预测）和真实向量映射到直角坐标系的两个点上，计算两个点之间的欧氏距离来衡量两个向量之间的差距：$$\boldsymbol{\mathrm{MSE}:=\frac{1}{d_{\text {out}}} \sum_{i=1}^{d_{\text {out}}}\left(y_{i}-o_{i}\right)^{2}}$$MSE误差函数的值总是大于等于0，==当MSE误差函数达到最小值0时，输出等于真实标签，此时神经网络的参数达到最优状态。==

张量形式实现
~~~
import tensorflow as tf
from tensorflow import keras
o = tf.random.normal([2,10])
y_onehot = tf.constant([1,3])
y_onehot = tf.one_hot(y_onehot,depth = 10)
loss = keras.losses.MSE(y_onehot,o)
print(loss)

OUT：
tf.Tensor([0.87758476 0.5987398 ], shape=(2,), dtype=float32)
~~~
**keras.MSE函数返回的是每个样本的均方差，需要在样本数量上再次平均来获得batch的均方差**

层形式实现
~~~
import tensorflow as tf
from tensorflow import keras
o = tf.random.normal([2,10])
y_onehot = tf.constant([1,3])
y_onehot = tf.one_hot(y_onehot,depth = 10)
criteon = keras.losses.MeanSquaredError()
loss = criteon(y_onehot,o)
print(loss)

OUT：
tf.Tensor(0.907153, shape=(), dtype=float32)
~~~
MSE函数对应的类**keras.losses.MeanSquaredError(pred,label)**
####交叉熵

**熵的概念：熵越大，代表不确定性越大，信息量也就越大**

**熵计算：
$$\boldsymbol{H(P):=-\sum_{i} P(i) \log _{2} P(i)}$$
交叉熵的计算：
$$\boldsymbol{H(p, q):=-\sum_{i=n} p(i) \log _{2} q(i)}$$
通过变换，交叉熵可以分解为p的熵$\boldsymbol{H(p)}$与$\boldsymbol{p,q}$的KL散度的和：
$$\boldsymbol{H(p, q)=H(p)+D_{K L}(p | q)}$$**
**其中KL的定义为：
$$\boldsymbol{D_{K L}(p | q)=\sum_{x \in \mathcal{X}} p(x) \log \left(\frac{p(x)}{q(x)}\right)}$$
KL散度是用于衡量两个分布之间距离的指标，$\boldsymbol{p=q}$时，$\boldsymbol{D_{KL}(p|q)}$取得最小值0.需要注意的是，交叉熵和KL散度都不是对称的：$$\boldsymbol{\begin{aligned}
H(p, q) & \neq H(q, p) \\
D_{K L}(p | q) & \neq D_{K L}(q | p)
\end{aligned}}$$**
**交叉熵可以很好地衡量2个分布之间的差别，特别地，当分类问题中y的编码分布p采用one_hot编码时，$\boldsymbol{H(y)=0}$,此时$$\boldsymbol{H(\boldsymbol{y}, \boldsymbol{o})=H(\boldsymbol{y})+D_{K L}(\boldsymbol{y} | \boldsymbol{o})=D_{K L}(\boldsymbol{y} | \boldsymbol{o})}$$退化到真实标签分布$\boldsymbol{y}$与输出概率分布$\boldsymbol{o}$之间的KL散度上，根据KL散度的定义，我们推导分类问题中交叉熵的计算表达式：$$\boldsymbol{H(y,o)=D_{KL}(y|o)=\sum_jy_j\log(\frac{y_j}{o_j})=1*\log\frac{1}{o_i}+\sum_{j\ne i}0*\log(\frac{0}{o_j})=-\log o_i}$$**
**其中$\boldsymbol{i}$为one-hot编码中为1的索引号，也是当前输入的真实类别。可以看到，交叉熵损失函数只与真实类别$\boldsymbol{i}$上的概率$\boldsymbol{o_i}$有关，对应概率$\boldsymbol{o_i}$越大，$\boldsymbol{H(y,o)}$越小，当对应概率为1时，交叉熵$\boldsymbol{H(y,o)}$取得最小值0，此时网络输出$\boldsymbol{o}$与真实标签$\boldsymbol{y}$完全一致，神经网络取得最优状态。最小化交叉熵的过程也是最大化正确类别的预测概率的过程。**
###神经网络类型
* ==全连接层是神经网络最基本的网络类型，对后续神经网络类型的研究有巨大的贡献，全连接层前向计算简单，梯度求导也较简单，但是在处理较大特征长度的数据时，全连接层的参数量往往较大，使得训练深层数的全连接层网络比较困难。下面是一系列神经网络变种类型==
1. 卷积神经网络
2. 循环神经网络
3. 注意力（机制）网络
4. 图神经网络

###油耗预测实战
~~~
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from tensorflow.keras import layers,losses,optimizers,Model
import matplotlib.pyplot as plt
dataset_path = keras.utils.get_file("auto-mpg.data","http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values="?",comment='\t',sep=" ",skipinitialspace=True)
"""
read_csv()
必填参数：
filepath_or_buffer读取文件的路径
常用参数：
sep：指定分隔符
header：指定行数来作为列名（数据开始行数）。如果文件中没有列名。则默认为0，否则设置为None。对于数据读取有表头和没表头的情况很实用
names：列表，用于对列名的重命名，即添加表头，如果数据有表头，但想用新的表头，可以设置header=0，names=['a','b']实现表头的定制
index_col:整形或者列表，用作行索引的列编号或者列名，如果给定一个列表则说明有多个行索引，如：index_col=[0,1]来指定文件中的第一和第二列为索引列
usecols：列表，返回一个数据子集，即选取某几列，不读取整个文件的内容，有助于加快速度和降低内容
squeeze:如果文件值包含一列，则返回一个series
prefix：str，在没有列标题时，给列添加前缀
mangle_dupe_cols:重复的列，如果设定为False则会将所有重名列覆盖
"""
dataset = raw_dataset.copy()
dataset.head()
dataset.isna().sum()
统计空白数据，原始数据中的数据可能含有空字段（缺失值）的数据项，需要清除这些记录项
dataset = dataset.dropna()
删除空白数据项
dataset.isna().sum()
再次统计空白数据
"""
由于origin字段为类别数据，我们要将其移动出来，并转换为新的3个字段：USA，Europe和Japan，分别代表是否来自此产地
"""
处理类别数据，其中origin列代表了1，2，3，分布代表产地：美国、欧洲、日本
先弹出（删除并返回）origin这一列
origin = dataset.pop('Origin')
根据origin列来写入新的3个列
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()
按着8：2的比例切分训练集和测试集
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#移动MPG这一列为真实标签Y
train_stats = train_dataset.describe()
train_labels = train_dataset.pop('MPG')
test_label = test_dataset.pop('MPG')

查看训练集的输入X的统计数据（均值，标准差），并完成数据的标准化

train_stats.pop('MPG')
train_stats = train_stats.transpose()
标准化数据
def norm(x):
  return (x - train_stats['mean'])/train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
打印出训练集和测试集的大小
"""print(normed_train_data.shape,train_labels.shape)
print(normed_test_data.shape,test_label.label.shape)"""
创建数据集对象
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values,train_labels.values))
随机打散，批量化
train_db = train_db.shuffle(100).batch(32)
创建网络
class Network(Model):
    def __init__(self):
        #回归网络
        super(Network,self).__init__()
        #创建3个全连接层
        self.fc1 = layers.Dense(64,activation='relu')
        self.fc2 = layers.Dense(64,activation='relu')
        self.fc3 = layers.Dense(1)
    def call(self,inputs,training=None,mask=None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
"""
实例化网络对象和创建优化器
"""
创建网络类实例
model = Network()
model.build(input_shape=(4,9))
打印网络信息
model.summary()
创建优化器，指定学习率
optimizer = tf.keras.optimizers.RMSprop(0.001)
losslist=[]
for epoch in range(200):
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            计算MSE
            loss = tf.reduce_mean(losses.MSE(y,out))
            计算MAE，测试模型的性能
            mae_loss = tf.reduce_mean(losses.MAE(y,out))
            打印误差
            if step % 10 ==0:
                print(epoch,step,float(loss))
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
    losslist.append(loss)
plt.plot(X,losslist,label = 'loss')
plt.legend()
plt.show()
~~~
![](技术学习图片\26.png)
#Tensorflow2.0
* Tensorflow中的计算可以表示为一个有向图，或称为计算图，其中的每一个运算操作将作为一个节点，节点与节点之间的连接称为边，这个计算图描述了数据的计算流程，它也负责维护和更新状态，==计算图中每一个节点可以有任意多个输入和输出，在计算图的边中流动（flow）的数据称为张量（tensor），因此这个框架叫做Tensorflow。==
* Tensorflow2和Tensorflow1.X的区别在于，Tensorflow1.X采用的方式叫做符号式编程，Tensorflow2采用的是命令式编程，也称为动态图优先模式，不像Tensorflow1.X要先创建计算图之后才能进行运算，Tensorflow2能同时得到计算图和数值结果
* **Tensorflow深度学习框架在构建网络的功能**：
1.加速计算
Tensorflow可以利用GPU实现大量矩阵之间的并行计算。
2.自动梯度
Tensorflow提供了自动求导的功能，可以不需要手动推导，即可计算出输出对网络的偏导数。
~~~
autograd

x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)
"创建4个张量"
with tf.GradientTape() as tape:
	"构建梯度环境"
	tape.watch([a, b, c])
	y=a**2 * x + b * x + c
[dy_da,dy_db,dy_dc] = tape.gradient(y,[a,b,c])
print(dy_da,dy_db,dy_dc)
~~~
3.常用神经网络接口
内建了常用网络运算函数，常用网络层，网络训练，网络保存和加载，网络部署等一系列深度学习系统的便捷功能。
~~~
linear_regression

import numpy as np
import matplotlib.pyplot as plt
def gradient(b_current,w_current,n,data,learningrate):
    b_gradient=0.
    w_gradient=0.
    z=learningrate
    for i in range(n):
        x = data[i,0]
        y = data[i,1]
        b_gradient += (2/n)*((w_current*x+b_gradient)-y)
        w_gradient += (2/n)*((w_current*x+b_current)-y)
    new_b = b_current - (b_gradient*z)
    new_w = w_current - (w_gradient*z)
    return new_b,new_w
def loss(n,b,w,data):
    totalloss = 0
    for i in range(0,n):
        x = data[i,0]
        y = data[i,1]
        loss = (y - (w*x + b))**2
        totalloss += loss
    result = totalloss/float(n)
    return result
def updata_gradient(data,num,learningrata):
    b = 0
    w = 0
    n = data.shape[0]
    totalloss = np.zeros([num])
    for i in range(num):
        b,w = gradient(b,w,n,data,learningrata)
        totalloss[i] = loss(n,b,w,data)
    return b,w,totalloss
def plot(data,w,b,num1,total_loss):
    x = np.zeros(data.shape[0])
    y = np.zeros(data.shape[0])
    y1 = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x[i] = data[i,0]
        y[i] = data[i,1]
    for j in range(data.shape[0]):
        num = x[j]
        y1[j] = w*num + b
    fig1 = plt.figure()
    ax=fig1.add_subplot(211)
    ax.plot(x,y,'ro',label='Actually')
    ax.plot(x,y1,'b-',label='predict')
    ax.legend()
    ax1 = fig1.add_subplot(212)
    x1 = np.arange(0,num1,1)
    ax1.plot(x1,total_loss,'r-',label='loss')
    ax1.legend()
    plt.tight_layout()
    fig1.show()

def run():
    b = 0
    w = 0
    learningrata = 0.0001
    data = np.genfromtxt(
        'D:\刘柏良\TensorFlow资料\TensorFlow-2.x-Tutorials-master\TensorFlow-2.x-Tutorials-master\深度学习与TensorFlow入门实战-源码和PPT\lesson04-回归问题实战\data.csv',
        delimiter=",")
    n=data.shape[0]
    num=1000
    totalloss = loss(n,b,w,data)
    print("start at w={0},b={1},loss={2}".format(w,b,totalloss))
    print("runing..")
    b,w,total_loss= updata_gradient(data,num,learningrata)
    totalloss = loss(n, b, w, data)
    print("end at w={0},b={1},loss={2}".format(w,b,totalloss))
    plot(data,w,b,num,total_loss)
if __name__ == '__main__':
    run()
~~~
~~~
手写图片识别
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets
(x, y), (x_val, y_val) = datasets.mnist.load_data()
"获取数据集中的训练集和测试集，第一个元组返回的是训练集，第二个元组返回的是测试集，x表示训练的图片，y表示训练集的标签"
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
"将图像矩阵转换为Tensorflow中能运算的tensor，并将图片的像素点归一化"
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
"将给定元组，列表的tensor或单个tensor沿其第一个维度切片，返回一个含有N个样本的数据集（假设tensor的第一个维度为N）"
train_dataset = train_dataset.batch(200)


model = keras.Sequential([ 
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)
"Keras.optimizers.SGD(lr:学习率,momentum:动量参数,decay:每次更新后的学习率减值,nesterov:布尔值，能否使用Nesterov动量)"



def train_epoch(epoch):

    # Step4.loop
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            """
            在Tensorflow1.X静态图的时代，每张计算图都有两部分，一部分是前向图，一部分是反向图。反向图用来计算梯度
            但是在Tensorflow2.0利用的是动态图，自动梯度是它的特点
            每行代码顺序执行，没有构建图的过程，但是不能每行都执行梯度计算
            因此需要一个上下文管理器，来告诉程序这里要进行梯度计算
            """
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        grads = tape.gradient(loss, model.trainable_variables)
        """
        计算loss关于model.trainable_variable的梯度
        默认情况下，GradientTape的资源在调用gradient函数后就被释放，再调用就无法计算了
        如果需要多次计算梯度，就需要开启persistent=True属性
        """
        # w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())


def train():
    for epoch in range(30):
        train_epoch(epoch)



if __name__ == '__main__':
    train()
~~~
##Tensorflow基础
* **数据类型**：张量是Tensorflow的主要数据载体，分为：
1.标量（Scalar）：单个的实数，维度数为0，shape为[]
2.向量（Vector）：n个实数的有序集合，通过中括号包裹，如：[1.2],[1.2,3.4]等，维度数为1
3.矩阵（Matrix）：n行m列实数的有序集合，如$\begin{bmatrix}1&2\\3&4\end{bmatrix}$或$[[1,2][3,4]]$,每个维度上的长度不定，shape为[n,m]
4.张量（Tensor）：所有维度数dim>2的数组统称为张量。张量的每个维度也叫做轴（axis），一般维度代表了具体的物理含义，比如shape为[2，32，32，3]共有四个维度，如果表示图片数据的话，每个维度代表的含义分别是图片数量、图片高度、图片宽度、图片通道数，当然张量的维度数以及每个维度所代表的具体物理含义需要用户自行定义

在Tensorflow中创建标量：
~~~
不像python动态声明变量一样，在Tensorflow创建标量要使用函数constant创建标量
import tensorflow as tf
a = 1.2
aa = tf.constant(1.2)#创建标量
print(type(a),type(aa),tf.is_tensor(aa))

OUt:<class 'float'> <class 'tensorflow.python.framework.ops.EagerTensor'> True
~~~
在Tensorflow中创建张量：
~~~
import tensorflow as tf
x = tf.constant([1,2.,3.3])
print(x)

OUT：tf.Tensor([1.  2.  3.3], shape=(3,), dtype=float32)
~~~
在Tensorflow中创建矩阵：
~~~
import tensorflow as tf
a = tf.constant([[[1,2],[3,4]],[[4,5],[6,7]]])
print(a)
~~~
* **字符串类型**
在Tensorflow中创建字符串类型的张量,**tf.string**模块中提供了常见字符串型的工具函数，但字符型类型在Tensorflow中使用不多
~~~
import tensorflow as tf
a = tf.constant("Hello world!")
print(a)
OUT:tf.Tensor(b'Hello world!', shape=(), dtype=string)
~~~
* **布尔类型**
需要注意的是，Tensorflow的布尔类型和python语言的布尔类型并不对等
~~~
import tensorflow as tf
a = tf.constant(True)
print(a)

OUT:tf.Tensor(True, shape=(), dtype=bool)

传入布尔类型的向量
import tensorflow as tf
a = tf.constant([True,False])
print(a)
OUT：tf.Tensor([ True False], shape=(2,), dtype=bool)
~~~
* **数值类型**
对于数值类型的张量，可以保持为不同字节长度的精度，如浮点数3.14既可以保存为16—bit长度，也可以保存为32—bit甚至64—bit的精度。bit越长，精度越高，同时占用的内存空间也就越大。常用的精度类型有tf.int16，tf.int32，tf.int64，tf.float16，tf.float32
~~~
import tensorflow as tf

print(tf.constant(123456789,dtype=tf.int16),tf.constant(123456789,dtype=tf.int32))

OUt:tf.Tensor(-13035, shape=(), dtype=int16) tf.Tensor(123456789, shape=(), dtype=int32)
第一个保存精度过低导致溢出，对于浮点数，高精度的张量可以表示更精准的数据
~~~
读取精度，通过访问张量得到dtype成员属性可以判断张量的保存精度，对于不符合要求的张量的类型及精度，需要通过tf.cast函数进行转换
~~~
import tensorflow as tf
a = tf.constant(1,dtype=tf.float16)
print(a.dtype)
if a.dtype != tf.float32:
    a = tf.cast(a,tf.float32)
print(a.dtype)

OUT：
<dtype: 'float16'>
<dtype: 'float32'>

布尔型与整形之间相互转换
import tensorflow as tf
a = tf.constant([True,False])
tf.cast(a,tf.int16)
print(a)

tf.Tensor([1 0], shape=(2,), dtype=int16)
~~~

* **待优化张量**
为了区分需要计算梯度信息的张量与不需要计算梯度信息的张量，Tensorflow增加了一种专门的数据类型来支持梯度信息的记录：**tf.Variable**，tf.Variable类型**在普通的张量类型基础上添加了name，trainable等属性来支持计算图的构建**。由于梯度运算会消耗大量的计算资源，而且会自动更新相关参数，对于不需要的优化的张量，如神经网络的输入X，不需要通过tf.Variable封装；相反，==对于需要计算梯度并优化的张量，如神经网络层的W和b，需要通过tf.Variable包裹以便Tensorflow跟踪相关梯度信息。name属性用于命名计算图中的变量，这套命名体系是TensorFlow内部维护的，trainable表征当前张量是否需要被优化，创建Variable对象是默认启用优化标志，可以设置trainable=False来设置张量不需要优化==

~~~
import tensorflow as tf
a = tf.constant([-1,0,1,2])
aa = tf.Variable(a)
print(aa.name,aa.trainable)

OUT：Variable:0 True
~~~


除了通过普通张量的方式创建Variable，也可以直接创建
~~~
import tensorflow as tf
a = tf.Variable([[1,2],[3,4]])
print(a)

OUT：
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
array([[1, 2],
       [3, 4]])>
~~~
###从Numpy，List创建张量
* Numpy Array数组和Python List是Python程序中非常重要的数据载体容器，很多数据都是通过Python语言将数据加载至Array或者List容器中，再转换为Tensor类型，通过TensorFlow运算处理后导出到Array或者List容器中，方便其他模块使用。

**tf.convert_to_tensor**可以创建新Tensor，并将保存在Python List对象或Numpy Array对象中的数据导入代新的Tensor中，实际上，tf.constant和tf.convert_to_tensor都可以自动二点把Numpy数组或Python List数据类型转化为Tensor类型，使用其一即可。
~~~
import tensorflow as tf
a = tf.convert_to_tensor([1,2.])
print(a)

OUT:tf.Tensor([1. 2.], shape=(2,), dtype=float32)
~~~
1. **创建全为0或全为1的张量**
   考虑到有一些参数矩阵需要通过训练得出，例如线性变换$y=wx+b$中的$w$和$b$就需要反向传播更新参数，那么一开始应该将它们初始化，将$w$初始化为1，$b$初始化为0，使得该线性变化层输出$y=x$，这是一种比较好的层初始化状态。通过**tf.zeros和tf.ones**即可创建任意形状全为0或1的张量
~~~
创建全1，全0的向量
import tensorflow as tf
print(tf.zeros([1]),tf.ones([1]))

OUT：
tf.Tensor([0.], shape=(1,), dtype=float32) tf.Tensor([1.], shape=(1,), dtype=float32)

创建全0，全1的矩阵
import tensorflow as tf
print(tf.zeros([2,2]),tf.ones([2,2]))

OUT：
tf.Tensor(
[[0. 0.]
 [0. 0.]], shape=(2, 2), dtype=float32) tf.Tensor(
[[1. 1.]
 [1. 1.]], shape=(2, 2), dtype=float32)

~~~
有时候要求初始化的矩阵与某个矩阵的大小相等，这时可以通过**tf.zeros_like,tf.ones_like**可以方便地创建与某个张量shape一致，内容全为0或全为1的张量



2. **创建自定义数值张量**
    有时候需要全部初始化为某个自定义数值的张量，可以通过**tf.fill**可以创建全为自定义数值Value的张量。
~~~
tf.fill(shape,value)
import tensorflow as tf
print(tf.fill([2,2],100))

OUT：
tf.Tensor(
[[100 100]
 [100 100]], shape=(2, 2), dtype=int32)
~~~ 


3. 创建已知分布的张量
    正态分布和均匀分布是最常见的分布之一，当需要数据随机性采样时，就需要创建采集遵循这些分布的数据样本，可通过$\boldsymbol{tf.random.normal(shape,mean=0.0,stddev^2})$ shape表示创建的形状，均值为mean，标准差为stddev的正态分布
    
创建正态分布张量
~~~
import tensorflow as tf
print(tf.random.normal([2,2],mean=1,stddev=2))

OUT：tf.Tensor(
[[-0.2810527  -1.4531558 ]
 [-0.59204316  0.29616654]], shape=(2, 2), dtype=float32)

~~~
**tf.random.uniform(shape,minval=0,maxval=None,dtype=float32)** 可以创建采样自[minval,maxval]区间的均匀分布的张量，==如果需要均匀采样整形类型的数据，必须指定采样取间的最大值maxval参数，同时制定数据类型为tf.int*型==
~~~
import tensorflow as tf
print(tf.random.uniform([2,2],maxval=100))

OUT：
tf.Tensor(
[[55.972492 20.881033]
 [31.853653 41.91586 ]], shape=(2, 2), dtype=float32)

采样整形类型的数据
import tensorflow as tf
print(tf.random.uniform([2,2],maxval=100,dtype=tf.int32))

OUT：
tf.Tensor(
[[41 68]
 [25 91]], shape=(2, 2), dtype=int32)
~~~
###创建序列
* 在循环计算或者对张量进行索引时，经常需要创建一段连续的整形序列，可以通过tf.range()函数来实现，tf.range(limit,delta=1)可以创建[0，limit]之间，步长为delta的整形序列，不包含limit本身

创建0~9，步长为1的整形序列
~~~
import tensorflow as tf
print(tf,range(10))

OUt:
tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
~~~
通过tf.range(start,limit,delta=1)可以创建[start,limit]，步长为delta的序列，不包含limit本身
~~~
import tensorflow as tf
print(tf.range(1,10,delta=2))

OUT：tf.Tensor([1 3 5 7 9], shape=(5,), dtype=int32)
~~~
##张量的应用
* 介绍完张量的相关属性和创建方式后，接下来要介绍每种维度下张量的典型应用，方便更直观地联想到张量的主要物理意义和用途，为后续张量的维度变换等一系列抽象操作的学习打下基础
###标量
标量就是一个简单的数字，维度数为0，==标量的典型用途之一是误差值的表示，各种测量指标的表示，比如准确度（Accuracy，acc）精度（Precision）和召回率（Recall）==
###向量
向量是一种非常常见的数据载体，如在全连接层和卷积神经网络层中，偏置张量$b$就是使用向量
###矩阵
矩阵也是非常常见的张量类型，比如全连接层的批量输入$X=\begin{bmatrix}b,d_{in}\end{bmatrix}$，其中$b$表示输入样本的个数，即batch size，$d_{in}$表示输入特征的长度。
###3维张量
三维的张量一个典型应用是表示序列信号，它的格式是$$X=[b,sequence len,feature len]$$
其中$b$表示序列信号的数量，$sequence$ $len$表示序列信号在时间维度上的采样点数，$feature$ $len$表示每个点的特征长度，例如自然语言处理中句子的表示，如评价句子的是否为正面情绪的情感分类任务网络，为了能够方便字符串被神经网络处理，一般将单词通过嵌入层（Embedding Layer）编码为固定长度的向量，比如两个等长（单词数为5）的句子序列可以表示为shape为[2,5,3]的3维张量，其中2个表示句子个数，5表示单词数量，3表示单词向量的长度。
###4维张量
4维张量在卷积神经网络中应用的非常广泛，它用于保存特征图（Feature Maps）数据，一般定义为$\begin{bmatrix}b,h,w,c\end{bmatrix}$,$b$表示输入的数量，$w,h$分布表示特征图的高宽，$c$表示特征图的通道数
##索引与切片
###索引
在TensorFlow中，支持基本的$[i][j]$标准索引方式，也支持通过逗号分隔索引号的索引方式。
~~~
import tensorflow as tf
假设是生成4张32×32大小的彩色图片
x = tf.random.normal([4,32,32,3])
print(x)
取第一张图的数据
print(x[0])
取第一张图的第一行的像素
print(x[0][0])
取第一张图的第一行第一列的像素
print(x[0][0][0])
去第一张图的第一行第一列的第一个通道的像素
print(x[0][0][0][0])
~~~
当张量的维度数较高时，使用$[i][j]……[k]$的书写不方便，可以采用$[i,j,……,k]$的方式索引，它们是等价的
###切片
通过$start:end:step$切片方式可以方便地提取一段数据，其中$start$为开始读取位置的索引，$end$为结束读取位置的索引（不包括$end$位），$step$为读取步长，这三个参数可以根据需要选择性地省略，全部省略即$::$，表示从最开始读取到最末尾，步长为1，即不跳过任何元素。特别地，$step$可以为负数，$step=-1$表示从$start$开始，逆序读取至$end$结束（不包括$end$），且索引号$end \le start$,

Tensorflow切片方法
![](技术学习图片\18.png)
##维度变换
* 在神经网络运算过程中，维度变换是最核心的张量操作，通过维度变换可以将数据任意地切换形式，满足不同场合的运算需求
**为什么需要维度变换？**
在线性层中$\boldsymbol{\mathrm{Y}=\mathrm{X} @ \mathrm{W}+b}$其中$X$包含了2个样本，每个样本的特征长度为4，$X$的shape为$[2,4]$。线性层的输出为3个节点，即$W$的shape定义为$[4,3]$，那么$\boldsymbol{\mathrm{X}@\mathrm{W}}$运算张量shape为$[2,3]$，需要叠加上shape为$[3]$的偏置$\boldsymbol{b}$。==不同的shape的2个张量怎么直接相加呢？==
回到我们设置偏置的初衷，我们给每个层的每个输出节点添加一个偏置，这个偏置数据是对所有的样本都是共享的，换言之，每个样本都应该累加上同样的偏置向量$\boldsymbol{b}$。
因此对于两个样本的输入X，我们需要将shape为$[3]$的偏置$\boldsymbol{b}$$$\begin{bmatrix}b_0\\b_1\\b_2\\\end{bmatrix}$$
按样本数量复制1份，变成矩阵形式$B'$：
$$\boldsymbol{B'=\begin{bmatrix}b_0&b_1&b_2\\b_0&b_1&b_2\end{bmatrix}}$$
$\mathrm{X}^{\prime}=\mathrm{X} @ \mathrm{W}$
$$X^{\prime}=\left[\begin{array}{ccc}
x_{00}^{\prime} & x_{01}^{\prime} & x_{02}^{\prime} \\
x_{10}^{\prime} & x_{11}^{\prime} & x_{12}^{\prime}
\end{array}\right]$$
此时$\boldsymbol{\mathrm{Y}=\mathrm{X} @ \mathrm{W}+b=\mathrm{Y}=\mathrm{X}^{\prime}+\mathrm{B}^{\prime}}$
$$Y=X^{\prime}+B^{\prime}=\left[\begin{array}{lll}
x_{00}^{\prime} & x_{01}^{\prime} & x_{02}^{\prime} \\
x_{10}^{\prime} & x_{11}^{\prime} & x_{12}^{\prime}
\end{array}\right]+\left[\begin{array}{lll}
b_{0} & b_{1} & b_{2} \\
b_{0} & b_{1} & b_{2}
\end{array}\right]$$
为了实现矩阵相加的这种形式，将$\boldsymbol{b}$插入一个新的维度，把它定义为batch（样本）维度。
==算法的每个模块对于数据张量的格式有不同的逻辑要求，当现有的数据格式不满足算法要求时，需要通过维度变换将数据调整为正确的格式，这就是维度变换的功能==
**基本的维度变换包含了改变视图的reshape，插入新维度expand_dims，删除维度squeeze，交换维度transpose，复制数据tile等**
###Reshape
####**张量的存储和视图的概念**：
==张量的视图就是我们理解张量的方式==，比如shape为$[2,4,4,3]$的张量A，我们从逻辑上可以理解为2张图片，每张图片4行4列，每个位置有RGB3个通道的数据，==张量的存储体现在张量在内存上保存为一段连续的内存区域，对于同样的存储，我们可以有不同的理解方式==，比如同样是shape为$[2,4,4,3]$，在不改变张量的存储方式的前提下，可以理解为2个样本，每个样本的特征的长度为48的向量
shape为$[2,4,4,3]$的数据在内存中存储的格式：
$$\begin{array}{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & \dots & \dots & \dots & 93 & 94 & 95 \\
\hline
\end{array}$$
由图可见，==内存并不支持这个维度层级概念，只能以平铺方式按序写入内存，因此这种层级关系需要人为管理，也就是说，每个张量的存储顺序需要人为跟踪==，为了方便表达，我们把shape中相对靠左侧的维度叫做**大维度**，相对靠右侧的维度叫做**小维度**，改变张量的视图仅仅是改变了张量的理解方式，并不会改变张量的存储方式，==改变视图实际上是要告诉计算机应该怎样去读取平铺排列的数据==，但是改变视图操作在提供便捷性的同时，也会带来很多逻辑隐患，主要原因是张量的视图与存储不同步造成的。
####Reshape视图
* **在TensorFlow中，可以通过张量的ndim和shape成员属性获得张量的维度数和形状**

假设初始视图为[b,h,w,c],下面改变初始视图的理解
**合法的视图变换操作**
$$[b, h * w, c] 张量理解为b 张图片，h*w 个像素点，c个通道$$
$$[b, h, w * c] 张量理解为b 张图片，h行，每行的特征长度为 \mathrm{w}^{*} \mathrm{c}$$
$$[b, h * w * c] 张量理解为 b 张图片，每张图片的特征长度为 \mathrm{h}^{*} \mathrm{w}^{*} \mathrm{c}$$
**不合法的视图变换操作**
如果定义的新视图为$[b,w,h,c],[b,c,h*w]$或者$[b,c,h,w]$等时，与初始张量的存储顺序相悖，果果不同步更新张量的存储顺序，那么恢复出的数据将与新视图不一致，从而导致数据错乱，改变视图是神经网络中非常常见的操作，==可以通过串联多个Reshape操作来实现复杂逻辑，但是在通过Reshape改变视图时，必须始终记住张量的存储顺序，新视图的维度顺序不能与存储顺序相悖，否则需要通过交换维度操作将存储顺序同步过来==。例如：对于shape为$[4,32,32,3]$的图片数据，通过Reshape操作将shape调整为$[4,1024,3]$，此时视图的维度顺序为$\boldsymbol{b-pixel-c}$的图片数据，张量的存储顺序为$[b,h,w,c]$，可以将$[4,1024,3]$恢复为：
$[b, h, w, c]=[4,32,32,3]$时，新视图的维度顺序与存储顺序无冲突，可以恢复出无逻辑问题的数据
$[b, w, h, c]=[4,32,32,3]$时，新视图的维度顺序与存储顺序冲突
$[h * w * c, b]=[3072,4]$时，新视图的维度顺序与存储顺序冲突
###增删维度
####增加维度
增加一个长度为1的维度相当于给原有的数据增加一个新维度的概念，维度长度为1，故数据并不需要改变，仅仅是改变数据的理解方式，因此它其实可以理解为改变视图的一种特殊方式。

**通过tf.expand_dims(x,axis)可以在指定axis轴前插入一个新的维度**
~~~
import tensorflow as tf
x = tf.random.uniform([5,5],maxval=10,dtype=tf.int32)
print(x)
x = tf.expand_dims(x,axis=2)
在尾部给张量增加一个新的维度
print(x)
x = tf.expand_dims(x,axis=0)
在张量最前面增加一个新的维度
print(x)

OUT:tf.Tensor(
[[4 2 2 5 5]
 [8 6 7 6 9]
 [6 1 5 2 1]
 [8 6 8 4 2]
 [2 6 1 1 1]], shape=(5, 5), dtype=int32)
tf.Tensor(
[[[4]
  [2]
  [2]
  [5]
  [5]]

 [[8]
  [6]
  [7]
  [6]
  [9]]

 [[6]
  [1]
  [5]
  [2]
  [1]]

 [[8]
  [6]
  [8]
  [4]
  [2]]

 [[2]
  [6]
  [1]
  [1]
  [1]]], shape=(5, 5, 1), dtype=int32)
tf.Tensor(
[[[[4]
   [2]
   [2]
   [5]
   [5]]

  [[8]
   [6]
   [7]
   [6]
   [9]]

  [[6]
   [1]
   [5]
   [2]
   [1]]

  [[8]
   [6]
   [8]
   [4]
   [2]]

  [[2]
   [6]
   [1]
   [1]
   [1]]]], shape=(1, 5, 5, 1), dtype=int32)
~~~
可以看到，插入一个新维度后，数据的存储顺序并没有改变，，仅仅是在插入一个新的维度后，改变了数据的视图
####删除维度
删除维度只能删除长度为1的维度，也不会改变张量的存储，**可以通过tf.squeeze(x,axis)函数，axis参数为待删除的维度的索引号**
~~~
import tensorflow as tf
x = tf.random.uniform([5,5],maxval=10,dtype=tf.int32)
x = tf.expand_dims(x,axis=2)
x = tf.expand_dims(x,axis=0)
print(x)
x = tf.squeeze(x,axis=0)
print(x)

OUT：tf.Tensor(
[[[[3]
   [1]
   [9]
   [6]
   [4]]

  [[4]
   [0]
   [5]
   [5]
   [3]]

  [[4]
   [2]
   [3]
   [6]
   [9]]

  [[1]
   [4]
   [9]
   [4]
   [9]]

  [[1]
   [7]
   [4]
   [7]
   [1]]]], shape=(1, 5, 5, 1), dtype=int32)
tf.Tensor(
[[[3]
  [1]
  [9]
  [6]
  [4]]

 [[4]
  [0]
  [5]
  [5]
  [3]]

 [[4]
  [2]
  [3]
  [6]
  [9]]

 [[1]
  [4]
  [9]
  [4]
  [9]]

 [[1]
  [7]
  [4]
  [7]
  [1]]], shape=(5, 5, 1), dtype=int32)
~~~
####交换维度
在实现算法逻辑时，在保持维度顺序不变的条件下，仅仅改变张量的理解方式是不够的，有时需要直接调整数据的存储顺序即交换维度，==这改变了张量的存储顺序，也改变了张量的视图。==

**交换维度的应用**：在TensorFlow中图片张量的默认存储格式是通道后行格式，：$[b,h,w,c]$，但是部分库的图片格式是通道先行：$[b,c,h,w]$，因此需要完成$[b,h,w,c]$到$[b,c,h,w]$维度的交换运算

**可以通过tf.transpose(x,perm)函数完成维度交换运算**，perm表示新维度的顺序
~~~
import tensorflow as tf
x = tf.random.normal([2,3,4,5],mean=1,stddev=2)
print(x)
x = tf.transpose(x,[0,3,1,2])
print(x)
~~~
####数据复制
通过增加维度操作插入新的维度后，可能希望在新的维度上面复制若干份数据，以满足后续算法的格式要求，类比$\boldsymbol{\mathrm{Y}=\mathrm{X} @ \mathrm{W}+b}$中的$\boldsymbol{b}$

**可以通过tf.tile(tensor,multiples)函数完成数据在指定维度上的复制操作，multiple分别指定了每个维度上面的复制倍数，1表示不复制，2表示复制原来长度的2倍，即数据复制一份，以此类推。**
###Broadcasting（广播机制）
它是一种轻量级张量复制手段，在逻辑上扩展张量数据的形状，但是只在需要的时候才会执行实际存储复制操作。对于大部分场景，Broadcasting机制都能通过优化手段避免实际复制数据而完成逻辑运算，从而相对于tf.tile函数，减少了大量计算代价。==它的最终效果和tf.tile复制相同，但是Broadcasting机制节省了大量计算资源，建议在运算过程中尽可能地利用Broadcasting提高计算效率。==

上述$\boldsymbol{\mathrm{Y}=\mathrm{X} @ \mathrm{W}+b}$的例子，我们通过tf.expand_dims和tf.tile完成实际复制数据运算，将$\boldsymbol{b}$变换为$[2,3]$，然后再与$\boldsymbol{\mathrm{X} @ \mathrm{W}}$进行矩阵相加，但是==实际上，我们可以直接将两个shape不相等的矩阵进行相加，这是为什么呢？==

因为在进行相加的时候，它自动调用了Broadcasting函数**tf.broadcast_to(x,new_shape)**，将两者shape扩张为相同的shape，上式可以等价为：$$y=x\left(w+t f . b \text { roadcast }_{-} \operatorname{to}(b,[2,3])\right.$$

**Broadcasting并不是对所有shape不同的张量的运算都有相同的效果，所有运算都要在正确的逻辑下运行，Broadcasting机制不会扰乱正常的计算逻辑，它只会针对最常见的场景自动完成增加维度并复制数据的功能，提高开发效率和运行效率**

####最常见的场景是什么？
Broadcasting机制的核心思想是==普适性，即同一份数据能普遍适合于其他位置==

**普适性的判断**：
在验证普适性之前要对维度进行一些操作：==将张量shape靠右对齐==
对于长度为1的维度，默认这个数据普遍适合于当前维度的其他位置；对于不存在的维度，则在增加新维度后默认当前数据也是普适性与新维度，从而可以扩展为更多维度数，其他长度的张量形状。

考虑shape为$[w,1]$的张量A，需要扩展为shape：$[b,h,w,c]$
$$\begin{array}{|l|l|l|l|}
\hline b & h & w & c \\
\hline & & w & 1 \\
\hline
\end{array}$$
将维度靠右对齐后，对于通道维度$c$，张量现长度为1，则默认此数据同样适合当前维度的其他位置，将数据逻辑上复制$c-1$份，长度变为$c$，对于不存在的$b$和$h$维度，则自动插入新维度，新维度长度为1，同时默认当前的数据普适于新维度的其他位置，即对于其他的图片 、其他的行来说，与当前的这一行的数据完全一致，这样将数据$b,h$维度的长度自动扩展为$b,h$
$$\begin{array}{|l|l|l|l|}
\hline b & h & w & c \\
\hline & & w & 1 \\
\hline
\end{array}\Rightarrow\begin{array}{|l|l|l|l|}
\hline b & h & w & c \\
\hline 1 & 1 & w & 1 \\
\hline
\end{array}\Rightarrow\begin{array}{|l|l|l|l|}
\hline b & h & w & c \\
\hline b & h & w & c \\
\hline
\end{array}$$
###数学运算
**加：tf.add
减：tf.subtract
乘：tf.multiply
除：tf.divide
乘方：tf.pow(x,a)，相当于$x^a$
平方:tf.square(x)
平方根：tf.sqrt(x)
指数：tf.exp()
对数：tf.math.log(x)
矩阵相乘:tf.matmul(a,b)**
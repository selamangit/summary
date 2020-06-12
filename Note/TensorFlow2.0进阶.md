##TensorFlow2.0进阶
###合并与分割
####合并
* 合并是指将多个张量在某个维度上合并为一个张量。以某学校班级成绩册数据为例，设张量$A$保存了1~4号班机的成绩册，每个班级35个学生，共8门科目，则张量A的shape为$[4,35,8]$；同样的方式，张量$B$保存了剩下的6个班级的成绩册，shape为$[6,35,8]$，为了得到全校的成绩册，需要合并这两个张量，便可以得到包含全校学生成绩的成绩册张量$C$，shape为$[10,35,8]$
**张量的合并可以通过拼接和堆叠操作来实现，拼接不会产生新的维度，堆叠会创建新的维度，选择使用拼接还是堆叠操作来合并张量，取决于具体的场景是否需要创建新的维度。**

#####拼接
**拼接操作可以在任意的维度上进行，唯一的约束是非合并维度的长度必须一致。通过tf.concat(tensors,axis)来实现**
~~~
import tensorflow as tf
a = tf.random.normal([4,35,8])
b = tf.random.normal([6,35,8])
c = tf.concat([a,b],axis=0)
tensor参数用方括号包起来，axis表示在哪个维度上合并
print(c.shape)

OUT:(10, 35, 8)
~~~
#####堆叠
* 继续以学生成绩册为例：假设张量$A$保存了某个班的成绩册，shape为$[35,8]$，张量$B$保存了另一个班级的成绩册，shape为$[35,8]$,合并这两个班的数据时，需要创建一个新维度，定义为班级数量维度，且新维度可以选择放置在任意位置，一般根据大小维度的经验法则，将较大概念的班级维度放置在学生维度之前，则合并后的张量的新shape为$[2,35,8]$

**如果在合并数据时，希望创建一个新的维度，则需要使用tf.stack(tensors,axis)来实现，tf.stack也需要满足张量堆叠合并条件，需要所有合并的张量的shape完全一致才可以合并**
~~~
import tensorflow as tf
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
c = tf.stack([a,b],axis=0)
print(c.shape)

OUT:(2, 35, 8)
~~~
####分割
* 分割是合并操作的逆操作，继续以学生成绩册为例：假设现在有全校的成绩册。shape为$[10,35,8]$，需要将每个班级的成绩册分离出来，保存在每个张量中

**通过tf.split(x,axis,num_or_size_splits)来实现
x表示待分割张量
axis表示分割的维度索引号
num_or_size_splits表示切割方案，若为单个数值时，表示切割的份数，当为list时，表示每个元素每份的长度，如$[2,4,2,2]$表示切割为4份，每份的长度为2，4，2，2**
###数据统计
* ==在神经网络的计算过程中，经常需要统计数据的各种属性，如最大值，均值，范数等。由于张量的shape比较大，直接观察数据很难获得有用信息，TensorFlow提供了快速获得这些数据的途径==
####向量范数
* 向量范数是表征向量“长度”的一种度量方法，常用来表示张量的权值大小，梯度大小等。

**$\boldsymbol{L1}$范数，定义为向量$\boldsymbol{x}$的所有元素绝对值之和$$\boldsymbol{\|x\|_{1}=\sum_{i}\left|x_{i}\right|}$$**
**$\boldsymbol{L2}$范数，定义为向量$\boldsymbol{x}$的所有元素的平方和，再开根号$$\boldsymbol{\|x\|_{2}=\sqrt{\sum_{i}\left|x_{i}\right|^{2}}}$$**
**$\boldsymbol{\infty-}$范数，定义为向量$\boldsymbol{x}$的所有元素绝对值的最大值$$\boldsymbol{\|x\|_{\infty}=\max _{i}\left(\left|x_{i}\right|\right)}$$**

**对于矩阵、张量，同样可以利用向量范数的计算公式，等价于将矩阵、张量打平成向量后计算**

**可以通过tf.norm(x,ord)求解张量的$\boldsymbol{L1,L2,\infty}$等范数，其中参数ord指定为1，2时计算$L1,L2$范数，指定为np。inf时计算$\infty$范数**
~~~
import tensorflow as tf
import numpy as np
x = tf.ones([2,2])
print(tf.norm(x,ord=1))
print(tf.norm(x,ord=2))
print(tf.norm(x,ord=np.inf))

OUT：
tf.Tensor(4.0, shape=(), dtype=float32)
tf.Tensor(2.0, shape=(), dtype=float32)
tf.Tensor(1.0, shape=(), dtype=float32)
~~~
###最大最小值、均值、和
* **可以通过tf.reduce_max，tf.reduce_min，tf.reduce_mean，tf.reduce_sum可以求解张量在某个维度上的最大、最小、均值、和，也可以求全局最大、最小、均值、和信息。**
* **除了希望获取张量的最值信息，还希望获得最值所在的索引号，例如在分类任务的标签预测。考虑10分类问题，得到神经网络的输出张量out，shape为$[2,10]$，代表2个样本属于10个类别的概率，由于元素的位置索引代表了当前样本属于此类别的概率，预测时往往会选择概率值最大的元素所在的索引号作为样本类别的预测值，可以通过tf.argmax(x,axis),tf.argmin(x,axis)来求解在axis轴上，x的最大值、最小值所在的索引号**

**tf.reduce_max(tensor,axis)**
~~~
import tensorflow as tf
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.reduce_max(x,axis=1))
~~~
**tf.reduce_min(tensor,axis)**
~~~
import tensorflow as tf
import numpy as np
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.reduce_min(x,axis=1))
~~~
**tf.reduce_mean(tensor,axis)，在求解误差函数时，需要计算样本的平均误差，此时可以通过tf.reduce_mean在样本数维度上计算均值**
~~~
import tensorflow as tf
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.reduce_mean(x,axis=1))
~~~
**tf.reduce_sum(tensor,axis)，与均值函数相似，可以求解张量在axis轴上所有特征的和**
~~~
import tensorflow as tf
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.reduce_sum(x,axis=1))
~~~
**tf.argmax(tensor,axis)**
~~~
import tensorflow as tf
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.argmax(x,axis=1))
~~~
**tf.argmin(tensor,axis)**
~~~
import tensorflow as tf
x = tf.random.normal([5,5],mean=1,stddev=2)
print(tf.argmin(x,axis=1))
~~~
###张量比较
* 为了计算分类任务的准确率（acc）等指标，一般需要将预测结果和真实标签比较，统计比较结果中正确的数量来计算准确率。

**可以通过tf.equal(a,b)或tf.math.equal(a,b)来比较这2个张量是否相等**

**但是tf.equal()函数返回布尔型的张量比较结果，需要统计张量中True的个数，就知道预测正确的个数。为了达到这个目的，我们先将布尔型转换为整形张量（tf.cast）**

**常用比较函数**
函数|功能
---|:--:|
**tf.math.greater**|$a>b$
**tf.math.less**|$a<b$
**tf.math.greater_equal**|$a\ge b$
**tf.math.less_equal**|$a\le b$
**tf.math.not_equal**|$a\ne b$
**tf.math.is_nan**|$a=nan$
###填充与复制
####填充
* 对于图片数据的高和宽、序列信号的长度，维度长度可能各不相同。为了方便网络的并行计算，需要将不同长度的数据扩张为相同长度，在==TensorFlow基础中介绍的数据复制==可以增加数据的长度，但是重复复制数据会破坏原有的数据结构，并不适合于此处，==通常做法是，在需要补充长度的信号开始或结束处填充足够数量的特定数值，如0，使得填充后长度满足系统要求。==

例如2个句子张量，每个单词使用数字编码的方式，如1表示$I$,2表示$like$等。第一个句子为：
"I like the weather today."
假设句子编码为：[1,2,3,4,5,6]，第二个句子为：
"So do I."
它的编码为：[7,8,1,6]。为了能够保存在同一个张量中，我们需要将这两个句子的长度保持一致，也就是说，需要将第二个句子的长度扩充为6，常见的填充方案是在句子末尾填充若干数量的0，变成[7,8,1,6,0,0]，此时这两个句子堆叠合并shape为[2,6]的张量

**填充操作可以通过tf.pad(x,paddings)函数实现，paddings是包含了多个[left Padding,Right padding]的嵌套方案list，如$\begin{bmatrix}\begin{bmatrix}0,0\end{bmatrix},\begin{bmatrix}2,1\end{bmatrix},\begin{bmatrix}1,2\end{bmatrix}\end{bmatrix}$表示第一个维度不填充，第二个维度左边（起始处）填充两个单元，右边（结束处）填充一个单元，第三个维度左边填充一个单元，右边填充两个单元**
~~~
import tensorflow as tf
a = tf.constant([1,2,3,4,5,6])
b = tf.constant([7,8,1,6])
b = tf.pad(b,[[0,2]])
print(b)
c = tf.stack([a,b],axis=0)
print(c)

OUT：
tf.Tensor([7 8 1 6 0 0], shape=(6,), dtype=int32)
tf.Tensor(
[[1 2 3 4 5 6]
 [7 8 1 6 0 0]], shape=(2, 6), dtype=int32)
~~~
#####填充在NLP中的应用
在自然语言处理中，需要加载不同句子长度的数据集，有些句子长度较小 ，如10个单词左右，部分句子长度较长，如超过100个单词。为了能够保存在同一个张量中，一般会选取能够覆盖大部分句子长度的阈值，如80个单词：对于小于80个单词的句子，在末尾填充相应数量的0；对于大于80个单词的句子，截断超过规定长度的部分单词。
####复制
**在TensorFlow中已经介绍了数据复制的相关知识，在此再复习一遍，就是通过tf.tile(tensor，multiples)函数来实现，multiples为一个列表，分别指明了各维度上的数据复制多少次**
~~~
import tensorflow as tf
a = tf.constant([[[1,2,3],[4,5,6],[7,8,9]],[[9,9,9],[8,8,8],[7,7,7]],[[1,1,1],[2,2,2],[3,3,3]]])
print(a)
a = tf.tile(a,multiples=[1,2,3])
print(a)

tf.Tensor(
[[[1 2 3]
  [4 5 6]
  [7 8 9]]

 [[9 9 9]
  [8 8 8]
  [7 7 7]]

 [[1 1 1]
  [2 2 2]
  [3 3 3]]], shape=(3, 3, 3), dtype=int32)
tf.Tensor(
[[[1 2 3 1 2 3 1 2 3]
  [4 5 6 4 5 6 4 5 6]
  [7 8 9 7 8 9 7 8 9]
  [1 2 3 1 2 3 1 2 3]
  [4 5 6 4 5 6 4 5 6]
  [7 8 9 7 8 9 7 8 9]]

 [[9 9 9 9 9 9 9 9 9]
  [8 8 8 8 8 8 8 8 8]
  [7 7 7 7 7 7 7 7 7]
  [9 9 9 9 9 9 9 9 9]
  [8 8 8 8 8 8 8 8 8]
  [7 7 7 7 7 7 7 7 7]]

 [[1 1 1 1 1 1 1 1 1]
  [2 2 2 2 2 2 2 2 2]
  [3 3 3 3 3 3 3 3 3]
  [1 1 1 1 1 1 1 1 1]
  [2 2 2 2 2 2 2 2 2]
  [3 3 3 3 3 3 3 3 3]]], shape=(3, 6, 9), dtype=int32)
~~~
###数据限幅
* 非线性激活函数ReLU函数其实可以通过简单的数据限幅运算实现的，限制数据的范围$\boldsymbol{x\in[0,+\infty]}$即可

**tf.maximum(x,a)实现数据的下限幅$\boldsymbol{x\in[a,+\infty]}$；通过tf.minimum(x,a)实现数据上限幅$\boldsymbol{x\in[-\infty,a]}$**
~~~
import tensorflow as tf
x = tf.range(9)
print(tf.maximum(x,2))

OUT:
tf.Tensor([2 2 2 3 4 5 6 7 8], shape=(9,), dtype=int32)
~~~
**更方便的，我们可以使用tf.clip_by_value()函数来实现上下限幅**
###高级操作
* 上述操作都是比较简单的操作，接下来就要介绍一些复杂一点的功能函数
####tf.gather(tensor，[index]，axis)
**tf.gather**可以==实现根据索引号收集数据的目的==。考虑班级成绩册的例子，共有4个班级，每个班级35个学生，8门科目，保存成绩册的张量的shape为$[4,35,8]$，现在需要收集第1~2个班级的成绩册，可以给定需要收集班级的索引号：[0,1]，班级维度为axis=0
~~~
import tensorflow as tf
x = tf.random.normal([4,35,8])
a = tf.gather(x,[0,1],axis=0)
print(a.shape)

OUT：
(2, 35, 8)
~~~
**tf.gather非常适合索引没有规则的场合，其中索引号可以乱序排序，此时收集的数据也是对应顺序**
~~~
import tensorflow as tf
x = tf.range(9)
a = tf.reshape(x,[3,3])
print(a)
print(tf.gather(a,[0,2,1],axis=0))

OUT：
tf.Tensor(
[[0 1 2]
 [3 4 5]
 [6 7 8]], shape=(3, 3), dtype=int32)
tf.Tensor(
[[0 1 2]
 [6 7 8]
 [3 4 5]], shape=(3, 3), dtype=int32)
~~~
**tf.gather可以组合起来使用，来索引多维数据中的某个数据，继续以学生成绩册为例，抽查第2~3班的3，4，6，27号学生的成绩**
~~~
import tensorflow as tf
x = tf.random.uniform([4,35,8],minval=0,maxval=100,dtype=tf.int32)
class_1 = tf.gather(x,[1,2],axis=0)
print(class_1.shape)
students = tf.gather(class_1,[3,4,6,27],axis=1)
print(students)

OUT：
(2, 35, 8)
tf.Tensor(
[[[90 81 88 55 17 23  9 18]
  [36 61 39 16 46 84 76 21]
  [81 67  5 33 70 52 76 74]
  [99 67  3  1 47 18 35 20]]

 [[83 62  1 39 54 18 54 93]
  [83 28 47 84 69  5 88  8]
  [10 59 85  6 67 64 99 76]
  [52 84 60 44 33 26 80 23]]], shape=(2, 4, 8), dtype=int32)
~~~
####tf.gather_nd(tensor，[index_list])
**通过tf.gather_nd,可以通过指定每次采样的坐标来实现采样多个点的目的，实现多维度坐标收集数据** 继续以学生成绩册为例，我们希望抽查第2个班级的第2个同学的所有科目，第3个班级的第3个同学的所有科目，第4个班级的第4个同学的所有科目的成绩，这3个采样点的索引坐标可以记为：$[1,1]，[2,2]，[3,3]$，我们将这3个采样点的索引合并为一个List$[[1,1]，[2,2],[3,3]]$
~~~
import tensorflow as tf
x = tf.random.uniform([4,35,8],minval=0,maxval=100,dtype=tf.int32)
print(tf.gather_nd(x,[[1,1],[2,2],[3,3]]))

OUT:
tf.Tensor(
[[ 3 42 28 94 98 37 19 69]
 [70 56 89 85  6 26 95 41]
 [ 3 88 85 66 82 21 98 93]], shape=(3, 8), dtype=int32)
~~~
####tf.boolean_mask(tensor，[mask_list]，axis)
**除了可以通过给定索引号的方式采样，还可以通过给定掩码（mask）的方式采样，但是要注意掩码的长度必须与对应维度的长度一致**，继续以学生成绩册为例，在班级维度上采样
~~~
import tensorflow as tf
x = tf.random.uniform([4,35,8],minval=0,maxval=100,dtype=tf.int32)
print(tf.boolean_mask(x,mask=[True,False,False,True],axis=0).shape)

OUT：
(2, 35, 8)
~~~
**多维掩码**
~~~
import tensorflow as tf
x = tf.random.uniform([2,4],minval=0,maxval=100,dtype=tf.int32)
print(x)
print(tf.boolean_mask(x,mask=[[True,False,False,True],[True,False,False,True]]))

OUT：
tf.Tensor(
[[12 66  5 66]
 [59 56 56 55]], shape=(2, 4), dtype=int32)
tf.Tensor([12 66 59 55], shape=(4,), dtype=int32)
~~~
####tf.where(cond，tensor_a，tensor_b)
**可以通过cond条件的真假从tensor_a或tensor_b中读取数据，判断条件：**
$$\boldsymbol{o_{i}=\left\{\begin{array}{ll}a_{i} & \text { cond }_{i} \text { 为True } \\ b_{i} & \text { cond }_{i} \text { 为False }\end{array}\right.}$$
其中$i$为张量的索引，返回张量大小与$a,b$张量一致，相当于条件选择语句，如果$cond_i$为True，则选择从$a_i$中复制数据，否则从$b_i$中复制数据
~~~
import tensorflow as tf
a = tf.ones([3,3])
b = tf.zeros([3,3])
c = tf.where([[True,False,True],[False,True,True],[True,True,True]],a,b)
print(c)

OUT：
tf.Tensor(
[[1. 0. 1.]
 [0. 1. 1.]
 [1. 1. 1.]], shape=(3, 3), dtype=float32)
~~~
**当$a=b=None$时，即$a，b$参数不指定时，tf.where会返回cond张量中所有True的元素的索引坐标**
~~~
import tensorflow as tf
a = tf.ones([3,3])
b = tf.zeros([3,3])
c = tf.where([[True,False,True],[False,True,True],[True,True,True]])
print(c)

OUT：
tf.Tensor(
[[0 0]
 [0 2]
 [1 1]
 [1 2]
 [2 0]
 [2 1]
 [2 2]], shape=(7, 2), dtype=int64)
~~~
**那么返回索引值的坐标有什么用呢？假设我们需要提取张量中所有正数的数据和索引：**

==首先创建张量a，并通过比较运算（tf.equal）得到所有正数的位置掩码，然后将掩码输入tf.where不指定任何张量，tf.where返回掩码为True的索引位置列表，接着将这个列表输入tf.gather_nd来获取所有的正数，实际上，在得到mask索引后，也可以直接通过tf.boolean_mask获取对应元素==
~~~
import tensorflow as tf
a = tf.random.normal([3,3],mean=1,stddev=2)
zero = tf.zeros([3,3])
mask = tf.math.greater_equal(a,zero)
获取掩码
index = tf.where(mask)
num = tf.gather_nd(a,index)
print(num)

OUT：
tf.Tensor([4.1355724  1.3075178  0.77469975 0.83479714], shape=(4,), dtype=float32)
~~~
####scatter_nd(indices，updatas，shape)
**通过该函数可以高效地刷新张量的部分数据，但是只能在全0张量的白板上刷新，因此可能需要结合其他操作来实现现有张量的数据刷新功能**

**一维张量白板的刷新运算**
![](技术学习图片\19.png)
==白板的形状表示为shape参数，需要刷新的数据索引为indices，新数据为updates，其中每个需要刷新的数据对应在白板中的位置，根据indices给出的索引位置将updates中新的数据依次写入白板中并返回更新后的白板张量。==
~~~
import tensorflow as tf
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([4.4,3.3,1.1,7.7])
print(tf.scatter_nd(indices,updates,[8]))

OUT:
tf.Tensor([0.  1.1 0.  3.3 4.4 0.  0.  7.7], shape=(8,), dtype=float32)
~~~
**三维张量白板的刷新运算**
![](技术学习图片\20.png)
~~~
import tensorflow as tf
indices = tf.constant([[1],[3]])
updates = tf.constant([[[5,5,5,5,],[6,6,6,6],[7,7,7,7],[8,8,8,8]],[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]])
print(tf.scatter_nd(indices,updates,[4,4,4]))

OUT:
tf.Tensor(
[[[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[5 5 5 5]
  [6 6 6 6]
  [7 7 7 7]
  [8 8 8 8]]

 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]

 [[1 1 1 1]
  [2 2 2 2]
  [3 3 3 3]
  [4 4 4 4]]], shape=(4, 4, 4), dtype=int32)
~~~
####meshgrid(x,y)
**如果想要绘制3D图，通过该函数能方便地生成二维网络采样点坐标，方便可视化等应用场合。**
~~~
import tensorflow as tf
import numpy as np
x = np.linespace(-8,8,100)
设置x坐标的间隔
y = np.linespace(-8,8,100)
设置y坐标的间隔
x,y = tf.meshgrid(x,y)
生成网格点，并拆分后返回
~~~
###经典数据集加载
####随机打散
* **通过DataSet.shuffle(buffer_size)工具可以设置Dataset对象随机打散数据之间的顺序，防止每次训练时数据按固定顺序产生，从而使得模型尝试“记忆”住标签信息**
**buffer_size指定缓冲池的大小，一般设置为一个较大的参数即可。**
####批训练
* ==为了利用显卡的并行计算能力，一般在网络的计算过程中会同时计算多个样本，我们把这种训练方式称为批训练，其中样本的数量叫做batch size。为了一次能够从Dataset中产生batch size数量的样本，需要设置Dataset围殴批训练方式==
**train_db = train_db.batch(128)**
==其中128为batch size参数，即一次并行计算128个样本的数据。该参数一般根据用户的GPU显存资源来设置，显存不够，可以适量减少batch size来减少算法的显存使用量==
####预处理
* **当从dataset中加载的数据集额格式与实际不能匹配时，用户要根据自己的逻辑实现预处理函数，将数据集的格式转化为我们需要的类型**
####循环训练
~~~
for step,(x,y) in enumerate(train_db):
~~~
==通过上述方式进行迭代，每次返回的x，y对象即为批量样本和标签，当所有样本完成迭代后，for循环终止，我们一般把完成一个batch的数据训练，叫做一个step，通过多个step来完成整个训练集的一次迭代，叫做一个epoch，在实际训练时，通常需要对数据集迭代多个epoch才能取得较好额训练效果==

##Keras高层接口
* Keras是一个主要由python语言开发的开源神经网络计算库，它被设计为高度模块化和易扩展的高层神经网络接口，使得用户可以不需要过多的专业知识就可以简洁、快速地完成模型的搭建与训练。==Keras库分为前端和后端，其中后端可以基于现有的深度学习框架实现，如：Theano，CNTK，TensorFlow，前端接口即Keras抽象过的统一接口API==

###常见功能模块
**Keras提供了一系列高层的神经网络类和函数，如常见数据集加载函数，网络层类，模型容器，损失函数类，优化器类，经典模型类等等。对于常见数据集，可以在实现下载、管理、加载功能函数**
####常见网络层类
* ==对于常见的神经网络层，可以使用张量方式的底层接口函数来实现，这些接口函数一般在**tf.nn模块**==，更常用地，对于常见的网络层，我们一般直接使用层方式来完成模型的搭建，==在**tf.keras.layers**命名空间中提供了大量常见网络层类接口，如全连接层，激活含水层，池化层，卷积层，循环神经网络等等==。对于这些网络层类，只需要在创建时**指定网络层的相关参数**，并**调用__call__方法**即可完成前向计算。在**调用__call__方法**时，Keras会自动调用每个层的前向传播逻辑，这些逻辑一般实现在类的**call函数**中。

####网络容器
在搭建一个深层次的网络时，如果需要手动调用每一层的类实例来完成前向传播运算，这会让代码显得很繁琐。因此**Keras提供了网络容器Sequential将多个网络层封装成一个大网络模型，只需要调用网络模型的实例一次即可完成数据从第一层到最末层的顺序计算，很大程度上提高了代码的可读性。**
~~~
import tensorflow as tf
from tensorflow.keras import layers,Sequential
Network = Sequential([
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation=None)
])
X = tf.random.normal([5,784],mean=10,stddev=4)
print(Network(X))
~~~
**Sequential容器也可以通过add()方法继续追加新的网络层，实现动态创建网络的功能**
~~~
import tensorflow as tf
from tensorflow.keras import layers,Sequential
Network = Sequential([
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='relu')
])
X = tf.random.normal([5,784],mean=10,stddev=4)
print(Network(X))
Network.summary()
Network.add(layers.Dense(3,activation='relu'))
print(Network(X))
Network.summary()

OUT：
tf.Tensor(
[[ 0.31222105  2.324831    0.          0.          0.8691125   4.446599
   0.7532449   0.          0.          7.0183835 ]
 [ 0.          4.575545    0.          0.          0.          0.
   0.          0.          0.          4.863575  ]
 [ 0.          4.6534796   0.          0.          0.          0.
   4.869212    0.          0.          8.841119  ]
 [ 7.117663    4.2178197   1.489924    0.          0.          1.0914445
   0.          0.          0.         15.866627  ]
 [ 0.          0.          1.4162097   0.          0.          4.338018
   4.436591    0.          0.          4.843932  ]], shape=(5, 10), dtype=float32)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  100480    
_________________________________________________________________
dense_1 (Dense)              multiple                  8256      
_________________________________________________________________
dense_2 (Dense)              multiple                  650       
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
_________________________________________________________________
tf.Tensor(
[[ 5.488745   0.         3.7889113]
 [ 3.5730066  0.         1.3197422]
 [ 4.624317   0.         3.2175438]
 [11.70434    0.         6.9458585]
 [ 2.813189   0.         5.0816736]], shape=(5, 3), dtype=float32)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                multiple                  100480    
_________________________________________________________________
dense_1 (Dense)              multiple                  8256      
_________________________________________________________________
dense_2 (Dense)              multiple                  650       
_________________________________________________________________
dense_3 (Dense)              multiple                  33        
=================================================================
Total params: 109,419
Trainable params: 109,419
Non-trainable params: 0
_________________________________________________________________
~~~

~~~
import tensorflow as tf
from tensorflow.keras import layers,Sequential
layer_num = 2
创建空网络容器
Network = Sequential([])
for _ in range(layer_num):
    Network.add(layers.Dense(3,activation='relu'))
Network.build(input_shape=(2,4))
X = tf.random.normal([2,4])
print(Network(X))
Network.summary()
~~~
**Network.summary() 函数的功能是打印网络结构和参数量，如果在使用动态创建网络时，全连接层网络的参数不是自定义的，这时很多类内并没有创建内部权值张量等成员变量，因此需要通过调用类的build方法并指定输入大小，即可自动创建所有层的内部张量**

==可以看见Layer列为每层的名字，这个名字由TensorFlow内部维护，与python的对象名不一样，param#列为层的参数个数，Total Params统计出了总的参数量，Trainable params为待优化的参数量，Non-trainable params为不需要优化的参数量。==

==Squential容量封装多层网络层时，所有层的参数列表将会自动并入Sequential容器的参数列表中，不需要人为合并网络参数列表。Sequential对象的trainable_variables和variables包含了所有层的待优化张量列表和全部张量列表==
###模型装配、训练与测试
* **训练网络的一般步骤：**
通过前向计算获得网络的输出值，再通过损失函数计算网络误差，然后通过自动求导工具计算梯度并更新，同时间隔性地测试网络的性能。
这种常用的训练逻辑，可以直接通过Keras提供的模型装配与训练高层接口实现，简洁清晰。

==在Keras中，有2个比较特殊的类：keras.Model和keras.layers.Layer类。**Layer类是网络层的母类，定义了网络层的一些常见功能，如：添加权值，管理权值列表等。Model类是网络的母类，除了具有Layer类的功能，还添加了保存、加载模型，训练与测试模型等便捷功能，Sequential也是Model的子类，因此具有Model类的所有功能**==

用Sequential容器封装网络
~~~
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets
"预处理数据"
def Process(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    x = tf.reshape(x,[28*28])
    y = tf.cast(y,tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
"Label独热编码"
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
train_ds = train_ds.map(Process).shuffle(60000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_ds = test_ds.map(Process).batch(128)
print(train_ds,test_ds)

"模型装配"
Network = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)])
Network.build(input_shape=(None,28*28))
Network.summary()

"模型训练"
Network.compile(optimizer=optimizers.Adam(lr=0.001),loss = losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = Network.fit(train_ds,epochs=5,validation_data=test_ds,validation_freq=2)
print(history.history)

sample = next(iter(test_ds))
print(sample)
x = sample[0]
y = sample[1]
pred = Network.predict(x)
y = tf.argmax(y,axis=1)
pred = tf.argmax(pred,axis=1)
print(y)
print(pred)
~~~
解释上面用到的一些函数的作用
* map（）函数根据提供的函数对指定序列进行映射，不改变原有序列而是返回一个新的序列，上面的操作就是对from_tensor_slices函数返回的数据集元组进行Process映射
* Sequential容器用来封装网络结构，也可以自己创建一个图层（类）来封装网络层结构
* **需要特别介绍一下Layer中的__init__()，build()，call()方法，这三个方法是从layers.Layer处继承来的，
其中__init__():初始化一些成员变量。
build():需要知道输入张量的大小，当然创建网络时不一定要等到调用build来创建变量，也可以在__init__中创建变量，但是在build中创建变量的优点是它可以根据图层将要操作的输入的形状启用后期变量创建，而且在__init__中创建变量时需要指定创建的变量的形状。
call函数在类被调用时执行**
* **compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
optimizer：指定参数优化器，
loss参数指定所选取的目标函数的类型，
metrics参数为一个列表，包含评估模型在训练和测试时的性能，当需要在多输出模型中为不同的输出指定不同的指标，可以传入为该参数传入一个字典
sample_weight_mode：如果需要执行按时间步采样权重（2D权重），可以设置为==temporal==。默认为==None==，为采样1D权重，如果模型有多个输出，那么可以通过传递mode的字典或列表
loss_weights:可选的指定标量系数（Python 浮点数）的列表或字典， 用以衡量损失函数对不同的模型输出的贡献。 模型将最小化的误差值是由 loss_weights 系数加权的加权总和误差。 如果是列表，那么它应该是与模型输出相对应的 1:1 映射。 如果是张量，那么应该把输出的名称（字符串）映到标量系数
target_tensors:默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。 相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据）， 您可以通过 target_tensors 参数指定它们。 它可以是单个张量（单输出模型），张量列表，或一个映射输出名称到目标张量的字典。
==功能：用于配置训练模型。==**
==compile主要完成损失函数和优化器的一些配置，是为训练服务的，如果只是需要预测，可以不用compile函数==
* **fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
x: 训练数据的 Numpy 数组（如果模型只有一个输入）， 或者是 Numpy 数组的列表（如果模型有多个输入）。 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 None（默认）。
y: 目标（标签）数据的 Numpy 数组（如果模型只有一个输出）， 或者是 Numpy 数组的列表（如果模型有多个输出）。 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 None（默认）。
batch_size: 整数或 None。每次梯度更新的样本数。如果未指定，默认为 32。
epochs: 整数。训练模型迭代轮次。一个轮次是在整个 x 和 y 上的一轮迭代。 请注意，与 initial_epoch 一起，epochs 被理解为 「最终轮次」。模型并不是训练了 epochs 轮，而是到第 epochs 轮停止训练。
verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
callbacks: 一系列的 keras.callbacks.Callback 实例。一系列可以在训练时使用的回调函数。 详见 callbacks。
validation_split: 0 和 1 之间的浮点数。用作验证集的训练数据的比例。 模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。 验证数据是混洗之前 x 和y 数据的最后一部分样本中。
validation_data: 元组 (x_val，y_val) 或元组 (x_val，y_val，val_sample_weights)， 用来评估损失，以及在每轮结束时的任何模型度量指标。 模型将不会在这个数据上进行训练。这个参数会覆盖 validation_split。
shuffle: 布尔值（是否在每轮迭代之前混洗数据）或者 字符串 (batch)。 batch 是处理 HDF5 数据限制的特殊选项，它对一个 batch 内部的数据进行混洗。 当 steps_per_epoch 非 None 时，这个参数无效。
class_weight: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。 这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
sample_weight: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。 您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1:1 映射）， 或者在时序数据的情况下，可以传递尺寸为 (samples, sequence_length) 的 2D 数组，以对每个样本的每个时间步施加不同的权重。 在这种情况下，你应该确保在 compile() 中指定 sample_weight_mode="temporal"。
initial_epoch: 整数。开始训练的轮次（有助于恢复之前的训练）。
steps_per_epoch: 整数或 None。 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。 使用 TensorFlow 数据张量等输入张量进行训练时，默认值 None 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
validation_steps: 只有在指定了 steps_per_epoch 时才有用。停止前要验证的总步数（批次样本）。
==功能：以给定数量的轮次（数据集上的迭代）训练模型。
返回：一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。==**

* **predict(x, batch_size=None, verbose=0, steps=None)
x: 输入数据，Numpy 数组 （或者 Numpy 数组的列表，如果模型有多个输出）。
batch_size: 整数。如未指定，默认为 32。
verbose: 日志显示模式，0 或 1。
steps: 声明预测结束之前的总步数（批次样本）。默认值 None。
==功能：为输入样本生成输出预测。计算是分批进行的。
返回：预测的 Numpy 数组（或数组列表）。==**
* **evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
x: 测试数据的 Numpy 数组（如果模型只有一个输入）， 或者是 Numpy 数组的列表（如果模型有多个输入）。 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 None（默认）。
y: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 None（默认）。
batch_size: 整数或 None。每次评估的样本数。如果未指定，默认为 32。
verbose: 0 或 1。日志显示模式。 0 = 安静模式，1 = 进度条。
sample_weight: 测试样本的可选 Numpy 权重数组，用于对损失函数进行加权。 您可以传递与输入样本长度相同的扁平（1D）Numpy 数组（权重和样本之间的 1:1 映射）， 或者在时序数据的情况下，传递尺寸为 (samples, sequence_length) 的 2D 数组，以对每个样本的每个时间步施加不同的权重。 在这种情况下，你应该确保在 compile() 中指定 sample_weight_mode="temporal"。
steps: 整数或 None。 声明评估结束之前的总步数（批次样本）。默认值 None。
==功能：在测试模式下返回模型的误差值和评估标准值。计算是分批进行的。
返回：标量测试误差（如果模型只有一个输出且没有评估标准） 或标量列表（如果模型具有多个输出 和/或 评估指标）。 属性 model.metrics_names 将提供标量输出的显示标签。==**
###模型保存与加载
* ==模型训练完成后要保存到文件系统上，方便后续的模型测试与部署工作，事实上，在训练时间间隔性地保存模型状态也是非常好的习惯，这点在训练大规模的网络尤其重要，因为大规模的网络需要训练数天乃至数周的时长，一旦训练过程被中断或者发生死机等意外，之前训练的进度将全部丢失。如果能间断的保存模型状态到文件系统，即使发生意外，也可以从最近一次的网络状态文件中恢复，从而避免浪费大量的训练时间。因此模型的保存与加载非常重要。==
####张量方式
**网络的状态主要体现在网络的结构以及网络层内部张量参数上，直接保存网络张量参数到文件上是最轻量级的一种方式，通过调用==Model.save_weights(path)方法==即可将当前的网络参数保存到path文件上。保存好参数文件后，在需要时，只需先创建好网络对象，然后调用网络对象的==load_weights(path)方法==即可将指定的模型文件中保存的张量数值写入当前网络参数中去
这种保存与加载网络的方式最为轻量级，文件中保存的仅仅是参数张量的数值，并没有其他额外的结构参数，==但是它需要使用相同的网络结构才能够恢复网络状态，因此一般在拥有网络源文件的情况下使用==**
####网络方式
**这是一种不需要网络源文件，仅仅需要模型参数文件即可恢复出网络模型的方式。通过==Model.save(path)函数==可以将模型的结构以及模型的参数保存到一个path文件上，在不需要网络源文件的条件下，==通过Keras.model.load_model(path)函数==即可恢复网络结构和网络参数**
####SavedModel方式
* **TensorFlow之所以能够被业界青睐，除了优秀的神经网络层API支持之外，还得益于它强大的生态系统，包括移动端和网页端的支持。当需要将模型部署到其他平台时，采用TensorFlow提出的SavedModel方式更具有平台无关性。通过==tf.keras.experimental.export_saved_model(network,path)函数==即可将模型以SavedModel方式保存到path目录中，保存好网络文件后，用户无需关心文件的保存格式，只需通过==tf.keras.experimental.load_from_saved_model()函数==即可恢复出网络结构和参数，方便各个平台能够无缝对接训练好的网络模型**
###自定义类
* **尽管Keras提供了很多的常用的网络层，但深度学习可以使用的网络层，对于需要创建自定义逻辑的网络层，可以通过自定义类来实现。==在创建自定义网络层类时，需要继承来自layers.Layer基类；创建自定义的网络类，需要继承来自keras.Model基类，这样产生的自定义类才能够方便的利用Layer/Model基类提供的参数管理功能，同时也能够与其他的标准网络层类交互使用==**

####自定义网络层
**自定义的网络层，需要实现初始化__init__方法和前向传播逻辑call方法。以某个具体的自定义的网络层为例，假设需要的是一个没有偏置的全连接层，即bias=0，同时固定激活函数为Relu函数。尽管这可以通过标准的Dense层创建。**
~~~
class MyDense(layers.Layer):
    def __init__(self,in_dim,ou_dim):
        "super函数MyDense是子类，继承了layers.Layer这个父类"
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w',[in_dim,ou_dim],trainable=True)
        self.bias = self.add_weight('b',[ou_dim],trainable=True)

    def call(self,inputs,training=None):

        out = inputs @ self.kernel + self.bias
        out = tf.nn.relu(out)
        return out
~~~
**trainable设置张量是否需要加入待优化变量，一般设置为True，因为反向传播时要更新参数，所以张量要处于监管状态**
####自定义网络
自定义的网络结构也可以像keras其他标准类一样，通过Sequential容器方便地包裹成一个网络模型，如：在上面创建的自定义网络层的基础上搭建网络结构
~~~
Network = Sequential([
    MyDense(256,128),
    MyDense(128,64),
    MyDense(64,32),
    MyDense(32,10)])
Network.build(input_shape=(None,28*28))
Network.summary()
~~~
在一些情况下也需要通过继承基类来搭建任意逻辑的自定义网络结构，下面是自定义的网络结构
~~~
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)

    def call(self,inputs):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x
~~~
==**自定义网络的优点**：Sequential容器也能实现一些网络的搭建，但是Sequential容器在前向传播是依次调用每个网络层的前向传播函数，灵活性一般，而自定义网络的前向逻辑可以任意定制==

**自定义网络——MNIST手写数字识别实战**
~~~
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets
def Process(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    x = tf.reshape(x,[28*28])
    y = tf.cast(y,tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
"Label独热编码"
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
"map（）函数根据提供的函数对指定序列进行映射，不改变原有序列而是返回一个新的序列"
train_ds = train_ds.map(Process).shuffle(60000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_ds = test_ds.map(Process).batch(128)
class MyDense(layers.Layer):
    def __init__(self,in_dim,ou_dim):
        "super函数MyDense是子类，继承了layers.Layer这个父类"
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w',[in_dim,ou_dim],trainable=True)
        self.bias = self.add_weight('b',[ou_dim],trainable=True)

    def call(self,inputs,training=None):

        out = inputs @ self.kernel + self.bias
        out = tf.nn.relu(out)
        return out
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)

    def call(self,inputs):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

Network = MyModel()
"模型训练"
Network.compile(optimizer=optimizers.Adam(lr=0.001),loss = losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
Network.fit(train_ds,epochs=5,validation_data = test_ds,validation_freq=2)
Network.evaluate(test_ds)
sample = next(iter(test_ds))
print(sample)
x = sample[0]
y = sample[1]
pred = Network.predict(x)
y = tf.argmax(y,axis=1)
pred = tf.argmax(pred,axis=1)
print(y)
print(pred)
~~~
###模型乐园
* 对于常用的网络模型，如ResNet，VGG等，不需要手动搭建网络，可以直接从==keras.applications==子模块中调用这些经典模型，同时还可以通过设置weights参数加载预训练的网络参数。

####加载模型
以ResNet50迁移学习为例，==一般将ResNet50去掉最后一层后的网络作为新任务的特征提取子网络，即利用ImageNet上面训练的特征提取方法迁移到我们自定义的数据集上，并根据自定义任务的类别追加一个对应数据类别数的全连接分类层==，从而可以在预训练网络的基础上可以快速、高效地学习新任务。
~~~
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers,Model,Sequential,losses,applications
resnet = keras.applications.ResNet50(weights = 'imagenet',include_top = False)
resnet.summary()
x = tf.random.normal([4,224,224,3])
print(resnet(x))
~~~
在keras中调用了这些常用网络模型后，可以再进行自定义神经网络，可以利用Sequential容器封装自己新的网络，如果不想再训练ResNet网络参数，==可以通过设置resnet.trainable=False可以选择冻结ResNet部分的网络参数，只训练新建的网络层，从而快速、高效完成网络模型的训练。==
~~~
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers,Model,Sequential,losses,optimizers,applications
resnet = keras.applications.ResNet50(weights = 'imagenet',include_top = False)
x = tf.random.normal([4,224,224,3])
global_average_layer = layers.GlobalAveragePooling2D()
fc = layers.Dense(100)
mynet = Sequential([resnet,global_average_layer,fc])
mynet(x)
~~~
###测量工具
* 在网络的训练过程中，经常需要统计准确率，召回率等信息，除了可以通过手动计算并平均方式获取统计数据外，Keras提供了一些常用的测量工具Keras.metrics，专门用于统计训练过程中需要的指标数据。
####新建测量器
在==Keras.metrics模块==下，提供了较多的常用的测量类，如统计平均值的Mean类等等。以统计误差为例，在前向运算时，我们会得到每一个batch的平均误差，但是我们希望统计一个epoch的平均误差，因此我们选择使用Mean测量器：
~~~
loss_meter = metrics.Mean()
~~~
####写入数据
通过测量器的==update_state函数==可以写入新的数据：
~~~
loss_meter.update_state(float(loss))
~~~
上述的代码记录采样的数据，上述采样代码放置在每个batch运算完成后，测量器会自动根据采样的数据来统计平均值。
####读取统计信息
在采样多次后，可以通过测量器的==result()函数==获取统计值：
~~~
print(step,'loss:',loss_meter.result())
~~~
####清除
由于测量器会统计所有历史记录的数据，因此在合适的时候有必要清除历史状态，通过==reset_states()函数==即可实现。在每次读取完统计信息后，清零统计信息，以便下一轮统计的开始：
~~~
if step%100 ==0:
    print(step,'loss:',loss_meter.result())
    loss_meter.reset_states()
~~~
####准确率统计实战
~~~
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets
def Process(x,y):
    x = tf.cast(x,dtype=tf.float32)/255
    x = tf.reshape(x,[28*28])
    y = tf.cast(y,tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
"Label独热编码"
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
"map（）函数根据提供的函数对指定序列进行映射，不改变原有序列而是返回一个新的序列"
train_ds = train_ds.map(Process).shuffle(60000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_ds = test_ds.map(Process).batch(128)
class MyDense(layers.Layer):
    def __init__(self,in_dim,ou_dim):
        "super函数MyDense是子类，继承了layers.Layer这个父类"
        super(MyDense, self).__init__()
        self.kernel = self.add_weight('w',[in_dim,ou_dim],trainable=True)
        self.bias = self.add_weight('b',[ou_dim],trainable=True)

    def call(self,inputs,training=None):

        out = inputs @ self.kernel + self.bias
        out = tf.nn.relu(out)
        return out
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)

    def call(self,inputs):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x

Network = MyModel()
"模型训练"
Network.compile(optimizer=optimizers.Adam(lr=0.001),loss = losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
Network.fit(train_ds,epochs=5,validation_data = test_ds,validation_freq=2)
Network.evaluate(test_ds)
pred = Network.predict(x)
pred = tf.argmax(pred,axis=1)
sample = next(iter(test_ds))
print(sample)
pred = Network.predict(x)
y = tf.argmax(y,axis=1)


acc_meter = metrics.Accuracy()
acc_meter.update_state(y,pred)
print('Evaluate Acc:',acc_meter.result().numpy())
acc_meter.reset_states()
~~~
###可视化
* 在网络训练的过程中，通过Web端监控网络的训练进度，可视化网络的训练结果对于提高开发效率是非常重要的。TensorFlow提供了一个专门的可视化工具，叫做TensorBorad，它可以通过TensorFlow将监控数据写入到文件系统，并利用Web后端监控对应的文件目录，从而可以允许用户从远程查看网络的监控数据
TensorBoard的使用需要训练部分和浏览器交互工作
####模型端
**在模型端，需要创建写入监控数据的Summary类，并在需要的时候写入监控数据，首先通过==tf.summary,create_file_writer==创建监控对象，并制定监控数据的写入目录：**
~~~
“创建监控类，监控数据将写入log_dir目录”
summary_writer = tf.summary.create_file_write(log_dir)
~~~
以监控误差数据和可视化图片数据为例，介绍如何写入监控数据。在前向计算完成后，对于误差这种标量数据，可以通过==tf.summary.scalar函数==记录监控数据，并指定时间戳step：
~~~
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
    tf.summary.scalar('train-loss',float(loss),step=step)
~~~
TensorBroad通过字符串ID来区分不同类别的监控数据，因此对于误差数据，将其命名为“train-loss”，其他类的数据不可写入此对象，防止数据污染。

对于图片类型的数据，可以通过tf.summary.image函数写入监控图片数据：
~~~
summary_writer = tf.summary.create_file_writer(log_dir)
with summary_writer.as_default():
    tf.summary.scalar('test-acc',float(total_correct/total))
~~~
####浏览器端
在运行程序时，通过运行tensorboard--logdir path指定Web后端监控的文目录path，此时打开浏览器，输入网址http://localhost:6006即可监控网络训练进度。TensorBoard可以同时显示多条监控记录，在监控页面的左侧可以选择监控记录

在监控页面的上端还可以选择不同类型数据的监控页面，比如标量监控页面SCALARS，图片可视化页面IMAGES等。
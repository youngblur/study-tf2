# Tensorflow2.0 笔记

学习[eat_tensorflow2_in_30_days](https://github.com/lyhue1991/eat_tensorflow2_in_30_days) 摘要笔记

其他学习网址:

-  [tensorflow官网API](https://tensorflow.google.cn/versions?hl=en)
- [简单粗暴Tensorflow2.0](https://tf.wiki/)



## 0. 目录

[toc] 

## 1.  核心概念

### 1.1 张量数据结构

#### 1.1.1 常量张量

- `tf.constant([1,2,3])` , 常量张量，和numpy.array 基本一致
- `tf.rank(张量).numpy()`， 返回数据的维度(tf.Tensor类型)，使用numpy返回数字

- `tf.cast(张量,tf.float32)`, 改变张量的类型为 tf.float32
- `张量.numpy()`， 张量转化为 np.array

#### 1.1.2 变量张量

模型中需要被训练的参数一般被设置为变量

常量张量是不可以改变的，重新赋值相当于创造新的内存空间

- `tf.Variable([1,2], name='v')`  , 初始化变量
- `变量.assign_add([1,2])`,  重新赋值。变为 (2, 4)



### 1.2 三种计算图

#### 1.2.1 tensorflow 1.0 静态图 

分为两步: 1. 定义计算图， 2. 在会话中执行计算图

在tensorflow2.0 中 `tf.compat.v1` 中保留了对静态图构建的风格

#### 1.2.2 动态计算图

为每一个算子进行构建，构建后可以立即执行。

同时可以将计算图的输入输出封装成函数

#### 1.2.3 Tensorflow2.0 的 Autograph

动态图运行效率相对较低

可以使用 `@tf.function` 装饰器将普通python函数转化为静态图构建代码

那么采用Autograph分为两步: 1. 定义函数， 2. 调用函数

```
 # 启动tensorboard 在 jupyter中的魔法命令
%load_ext tensorflow

# 启动tensorboard
%tensorboard --logdir ....
```



### 1.3 自动微分机制

#### 1.3.1 使用 `tf.GradientTape` 来求微分

- 对变量求微分

``` python
# f(x) = a*x**2 + b*x + c的导数

x = tf.Variable(0.0,name = "x",dtype = tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a*tf.pow(x,2) + b*x + c
    
dy_dx = tape.gradient(y,x)
print(dy_dx)
# tf.Tensor(-2.0, shape=(), dtype=float32)
```

- 对常量求微分，需要增加`watch`, 如 : `tape.watch([a,b,c])`

#### 1.3.2 使用优化器求最小值

- 先`tape` 求梯度， 再`apply_gradient`

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a*tf.pow(x,2) + b*x + c
    dy_dx = tape.gradient(y,x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
tf.print(y, x)
```

- 直接使用 `optimizer.minimize`

```python
x = tf.Variable(0.0,name = "x",dtype = tf.float32)

#注意f()无参数, 相当于 loss function
def f():   
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a*tf.pow(x,2)+b*x+c
    return(y)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
for _ in range(1000):
    optimizer.minimize(f,[x])  
```



## 2. Tensorflow的层次结构

- 第一层 (硬件层): 支持 CPU, GPU, TPU 加入到计算资源池
- 第二层 (C++实现的内核层):  跨平台分布式实现
- 第三层 (python实现的操作符): 低阶api， 张量，计算图，自动微分
- 第四层 (python实现的模型组建): 中阶api，模型层，损失函数，优化器等
- 第五层 (python实现的模型成品): 高阶api， 为tf.keras.models提供模型的类接口

### 2.1 低阶 api 示例

主要包括 张量操作，计算图和自动微分

- 打印时间线的操作

  ```python
  import tensorflow as tf
  
  #打印时间分割线
  @tf.function
  def printbar():
      today_ts = tf.timestamp()%(24*60*60)
  
      hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
      minite = tf.cast((today_ts%3600)//60,tf.int32)
      second = tf.cast(tf.floor(today_ts%60),tf.int32)
      
      def timeformat(m):
          if tf.strings.length(tf.strings.format("{}",m))==1:
              return(tf.strings.format("0{}",m))
          else:
              return(tf.strings.format("{}",m))
      
      timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                  timeformat(second)],separator = ":")
      tf.print("=========="*8+timestring)
  ```

#### 2.1.1 线性回归模型

- 准备数据

  ```python
  import numpy as np 
  import pandas as pd
  from matplotlib import pyplot as plt 
  import tensorflow as tf
  
  
  #样本数量
  n = 400
  
  # 生成测试用数据集
  X = tf.random.uniform([n,2],minval=-10,maxval=10) 
  w0 = tf.constant([[2.0],[-3.0]])
  b0 = tf.constant([[3.0]])
  Y = X@w0 + b0 + tf.random.normal([n,1],mean = 0.0,stddev= 2.0)  # @表示矩阵乘法,增加正态扰动
  
  ```

- 数据管道迭代器

  ```python
  # 构建数据管道迭代器
  def data_iter(features, labels, batch_size=8):
      num_examples = len(features)
      indices = list(range(num_examples))
      np.random.shuffle(indices)  #样本的读取顺序是随机的
      for i in range(0, num_examples, batch_size):
          indexs = indices[i: min(i + batch_size, num_examples)]
          yield tf.gather(X,indexs), tf.gather(Y,indexs)
  ```

- 定义模型

  ```python
  w = tf.Variable(tf.random.normal(w0.shape))
  b = tf.Variable(tf.zeros_like(b0,dtype = tf.float32))
  
  # 定义模型
  class LinearRegression:     
      #正向传播
      def __call__(self,x): 
          return x@w + b
  
      # 损失函数
      def loss_func(self,y_true,y_pred):  
          return tf.reduce_mean((y_true - y_pred)**2/2)
  
  model = LinearRegression()
  ```

- 训练模型

  ```python
  # 使用动态图调试
  def train_step(model, features, labels):
      with tf.GradientTape() as tape:
          predictions = model(features)
          loss = model.loss_func(labels, predictions)
      # 反向传播求梯度
      dloss_dw,dloss_db = tape.gradient(loss,[w,b])
      # 梯度下降法更新参数
      w.assign(w - 0.001*dloss_dw)
      b.assign(b - 0.001*dloss_db)
      
      return loss
  
  def train_model(model,epochs):
      for epoch in tf.range(1,epochs+1):
          for features, labels in data_iter(X,Y,10):
              loss = train_step(model,features,labels)
  
          if epoch%50==0:
              printbar()
              tf.print("epoch =",epoch,"loss = ",loss)
              tf.print("w =",w)
              tf.print("b =",b)
  
  train_model(model,epochs = 200)
  ```

- 使用 autograph 机制转化为静态图加速

  ```python
  ##使用autograph机制转换成静态图加速
  
  @tf.function
  def train_step(model, features, labels):
      with tf.GradientTape() as tape:
          predictions = model(features)
          loss = model.loss_func(labels, predictions)
      # 反向传播求梯度
      dloss_dw,dloss_db = tape.gradient(loss,[w,b])
      # 梯度下降法更新参数
      w.assign(w - 0.001*dloss_dw)
      b.assign(b - 0.001*dloss_db)
      
      return loss
  
  def train_model(model,epochs):
      for epoch in tf.range(1,epochs+1):
          for features, labels in data_iter(X,Y,10):
              loss = train_step(model,features,labels)
          if epoch%50==0:
              printbar()
              tf.print("epoch =",epoch,"loss = ",loss)
              tf.print("w =",w)
              tf.print("b =",b)
  
  train_model(model,epochs = 200)
  ```

#### 2.1.2 二分类模型

- 定义模型

  ```python
  class DNNModel(tf.Module):
      def __init__(self,name = None):
          super(DNNModel, self).__init__(name=name)
          self.w1 = tf.Variable(tf.random.truncated_normal([2,4]),dtype = tf.float32)
          self.b1 = tf.Variable(tf.zeros([1,4]),dtype = tf.float32)
          self.w2 = tf.Variable(tf.random.truncated_normal([4,8]),dtype = tf.float32)
          self.b2 = tf.Variable(tf.zeros([1,8]),dtype = tf.float32)
          self.w3 = tf.Variable(tf.random.truncated_normal([8,1]),dtype = tf.float32)
          self.b3 = tf.Variable(tf.zeros([1,1]),dtype = tf.float32)
  
       
      # 正向传播
      @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
      def __call__(self,x):
          x = tf.nn.relu(x@self.w1 + self.b1)
          x = tf.nn.relu(x@self.w2 + self.b2)
          y = tf.nn.sigmoid(x@self.w3 + self.b3)
          return y
      
      # 损失函数(二元交叉熵)
      @tf.function(input_signature=[tf.TensorSpec(shape = [None,1], dtype = tf.float32),
                                tf.TensorSpec(shape = [None,1], dtype = tf.float32)])  
      def loss_func(self,y_true,y_pred):  
          #将预测值限制在 1e-7 以上, 1 - 1e-7 以下，避免log(0)错误
          eps = 1e-7
          y_pred = tf.clip_by_value(y_pred,eps,1.0-eps)
          bce = - y_true*tf.math.log(y_pred) - (1-y_true)*tf.math.log(1-y_pred)
          return  tf.reduce_mean(bce)
      
      # 评估指标(准确率)
      @tf.function(input_signature=[tf.TensorSpec(shape = [None,1], dtype = tf.float32),
                                tf.TensorSpec(shape = [None,1], dtype = tf.float32)]) 
      def metric_func(self,y_true,y_pred):
          y_pred = tf.where(y_pred>0.5,tf.ones_like(y_pred,dtype = tf.float32),
                            tf.zeros_like(y_pred,dtype = tf.float32))
          acc = tf.reduce_mean(1-tf.abs(y_true-y_pred))
          return acc
      
  model = DNNModel()
  ```



### 2.2 中阶 api 示范

#### 2.2.1 线性回归模型

- 构建管道

  ```python
  #构建输入数据管道
  ds = tf.data.Dataset.from_tensor_slices((X,Y)) \
       .shuffle(buffer_size = 100).batch(10) \
       .prefetch(tf.data.experimental.AUTOTUNE)  
  ```

- 定义模型

  ```python
  model = layers.Dense(units = 1) 
  model.build(input_shape = (2,)) #用build方法创建variables
  model.loss_func = losses.mean_squared_error
  model.optimizer = optimizers.SGD(learning_rate=0.001)
  ```

- 训练模型 (直接对 variables 求梯度)

  ```python
  #使用autograph机制转换成静态图加速
  
  @tf.function
  def train_step(model, features, labels):
      with tf.GradientTape() as tape:
          predictions = model(features)
          loss = model.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
      grads = tape.gradient(loss,model.variables)
      model.optimizer.apply_gradients(zip(grads,model.variables))
      return loss
  ```

#### 2.2.2 二分类模型

- 定义模型

  ```python
  class DNNModel(tf.Module):
      def __init__(self,name = None):
          super(DNNModel, self).__init__(name=name)
          self.dense1 = layers.Dense(4,activation = "relu") 
          self.dense2 = layers.Dense(8,activation = "relu")
          self.dense3 = layers.Dense(1,activation = "sigmoid")
  
       
      # 正向传播
      @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
      def __call__(self,x):
          x = self.dense1(x)
          x = self.dense2(x)
          y = self.dense3(x)
          return y
      
  model = DNNModel()
  model.loss_func = losses.binary_crossentropy
  model.metric_func = metrics.binary_accuracy
  model.optimizer = optimizers.Adam(learning_rate=0.001)
  ```



### 2.3 高阶 api 示例

#### 2.3.1 线性回归模型

- 定义模型

  ```python
  tf.keras.backend.clear_session()
  
  model = models.Sequential()
  model.add(layers.Dense(1,input_shape =(2,)))
  model.summary()
  ```

- 训练模型

  ```python
  ### 使用fit方法进行训练
  
  model.compile(optimizer="adam",loss="mse",metrics=["mae"])
  model.fit(X,Y,batch_size = 10,epochs = 200)  
  ```

#### 2.3.2 二分类模型



## 3. Tensorflow 的低阶 api

### 3.1 张量的结构操作

#### 3.1.1 创建张量

- `tf.constant([1,2,3], dtype=tf.float32)`

  - ```
    [1,2,3]
    ```

- `tf.range(1,10, delta=2)`

  - ```
    [1 3 5 7 9]
    ```

- `tf.linespace(0.0, 2*3.14, 100)`

  - ```
    [0 0.0634343475 0.126868695 ... 6.15313148 6.21656609 6.28]
    ```

- `tf.zeros([3,3])`

  - ```
    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    ```

- ```python
  a = tf.ones([3,3])
  b = tf.zeros_like(a,dtype= tf.float32)
  tf.print(a)
  tf.print(b)
  ```

  - ```
    [[1 1 1]
     [1 1 1]
     [1 1 1]]
    [[0 0 0]
     [0 0 0]
     [0 0 0]]
    ```

- `tf.fill([3,2], 5)`

  - ```
    [[5 5]
     [5 5]
     [5 5]]
    ```

- ```python
  #均匀分布随机
  tf.random.set_seed(1.0)
  a = tf.random.uniform([5],minval=0,maxval=10)
  ```

  - ```
    [1.65130854 9.01481247 6.30974197 4.34546089 2.9193902]
    ```

- ```python
  #正态分布随机
  b = tf.random.normal([3,3],mean=0.0,stddev=1.0)
  ```

  - ```
    [[0.403087884 -1.0880208 -0.0630953535]
     [1.33655667 0.711760104 -0.489286453]
     [-0.764221311 -1.03724861 -1.25193381]]
    ```

- ```
  #正态分布随机，剔除2倍方差以外数据重新生成
  c = tf.random.truncated_normal((5,5), mean=0.0, stddev=1.0, dtype=tf.float32)
  ```

- `I = tf.eye(3,3)` : 单位矩阵

- `t = tf.linalg.diag([1,2,3])`： 对角矩阵

#### 3.1.2 索引切片

```
t = tf.random.uniform([5,5],minval=0,maxval=10,dtype=tf.int32)

[[4 7 4 2 9]
 [9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]
 [3 7 0 0 3]]
```

- `t[1:4, :4:2]`, 第一行到第四行(不取)，第零列到第四列(不取)每隔两列取一列

  - ```
    [[9 2]
     [7 7]
     [9 9]]
    ```

- ```
  #对变量来说，还可以使用索引和切片修改部分元素
  x = tf.Variable([[1,2],[3,4]],dtype = tf.float32)
  x[1,:].assign(tf.constant([0.0,0.0]))
  tf.print(x)
  ```

  - ```
    [[1 2]
     [0 0]]
    ```

- `tf.gather(matrix, [0, 5, 9], axis=1)` :
  - 在第一轴取 0， 5， 9
- `tf.gather_nd(matrix, indices=[(0,0), (2,4)])`
  - 取 第零轴是0，第一轴是0 和 第零轴是 2 第一轴是4的全部数据
- `tf.boolean_mast(matrix, [True, False, False, True], axis=1)` 
  - 取 第一轴 是 0， 3 的所有数据
- `tf.where(a > 0.5, tf.ones_like(c), tf.zeros_like(c))`:
  - 对 a 大于零的位置补充 1， 否则补充 0
- `tf.where(c < 0)` 
  - 若只有一个参数，则返回满足条件的坐标

#### 3.1.3 三维变换

- `tf.reshape`, 改变张量形状， 不改变存储顺序
- `tf.squeeze`, 减少维度(只有一个元素的维度)
- `tf.expand_dims`, 增加维度
- `tf.transpose`, 交换维度， 改变存储顺序

#### 3.1.4 合并分割

- `tf.concat([a,b,c], axis=1)`
  - 在第一维 进行 concat
- `tf.stack([a,b,c])`
  - 增加一维进行叠加
- `tf.split(c, 3, axis=1)`
  - 指定分割数目3， 平均分割
- `tf.split(c, [2,2,2], axis=0)`
  - 指定每份的记录数量



### 3.2 张量的数学运算

#### 3.2.1 标量计算

- `+`, `-`, `*`, `/`, `**`, `%`, `//`, `&`, `|`
- `tf.sqrt()`
- `tf.maximum(a,b)`
  - 对每一个位置取 其中较大的
- `tf.minium(a,b)`

#### 3.2.2 向量运算

只在特定的一个轴上运算

- `tf.reduce_sum(a, axis=0, keepdims=True)`
  - 指定 0 轴， 进行求和，并保留维度
- `tf.reduce_mean()`
- `tf.reduce_max()`
- `tf.reduce_min()`
- `tf.reduce_prod()` 指定轴相乘
- `tf.reduce_any()`
- `tf.reduce_all()`
- `tf.math.cumsum()`
  - 扫描累计相加， [1,2,3] -> [1,3,6]
- `tf.math.cumprod()`
- `tf.argmax()`,  `tf.argmin()`
  - 最大值最小值索引
- `values, indices = tf.math.top_k(a, 3, sorted=True)`
  - 返回最大的三个值 (排序后)，及其索引

#### 3.2.3 矩阵运算

必须是二维的，大部分矩阵有关的运算在 `tf.linalg`

- `a@b` : 矩阵乘法， 相当于 `tf.matmul(a,b)`
- `tf.transpose()`  转置
- `tf.linalg.inv()`,  矩阵逆
- `tf.linalg.trance()`, 矩阵的迹 
- `tf.linalg.norm()`， 矩阵的范数
- `tf.linalg.det()`, 矩阵的特征值
- `q,r = tf.linalg.qr(a)`, 矩阵的qr分解
- `v, s, d = tf.linalg.svd(a)`, 矩阵的 svd分解

#### 3.2.4 广播机制

和numpy是一样的

- `tf.broadcast_to(a, b.shape)`
  - 显示的方式，扩展 a 的维度到 b



### 3.3 AutoGraph 的使用规范

#### 3.3.1 规范总结

- 被`@tf.function`修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用`tf.print`而不是`print`，使用`tf.range`而不是`range`，使用`tf.constant(True)`而不是`True`.
- 避免在`@tf.function`修饰的函数内部定义`tf.Variable`. 
- 被`@tf.function`修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

#### 3.3.2 AutoGraph 编码规范解析

- 使用 Tensorflow 自带的函数而不是python中的其他函数

  ```python
  import numpy as np
  import tensorflow as tf
  
  @tf.function
  def np_random():
      a = np.random.randn(3,3)
      tf.print(a)
      
  @tf.function
  def tf_random():
      a = tf.random.normal((3,3))
      tf.print(a)
  ```

  - `np_random()`, 调用多次结果一样
  - `tf_random()`， 每次都会重新生成随机数

- 避免在内部定义 `tf.Variable`

  ```python
  # 避免在@tf.function修饰的函数内部定义tf.Variable.
  
  x = tf.Variable(1.0,dtype=tf.float32)
  @tf.function
  def outer_var():
      x.assign_add(1.0)
      tf.print(x)
      return(x)
  ```

  ```python
  @tf.function
  def inner_var():
      x = tf.Variable(1.0,dtype = tf.float32)
      x.assign_add(1.0)
      tf.print(x)
      return(x)
  
  inner_var()  # 执行将报错
  ```

- 不可修改外部的列表或字典等结构性变量

  ```python
  tensor_list = []
  
  #@tf.function #加上这一行切换成Autograph结果将不符合预期！！！
  def append_tensor(x):
      tensor_list.append(x)
      return tensor_list
  
  append_tensor(tf.constant(5.0))
  ```

### 3.4 AutoGraph 的机制原理

- 1. 创建静态计算图， 如果开启 autograph=True(默认开启), 控制流语句转化为 tensorflow图内控制流(if, while, for)

- 2. 执行计算图，相当于 tensorflow1.0 开启session
- 再次输入相同参数只会进行2， 再次输入不同的参数会进行1和2



### 3.5 AutoGraph 和 tf.Module

使用 autograph 时避免内部定义 `tf.Variable`, 所以一个简单的思路就是定义一个类。 

`tf.Module` 就是这个基类， 可以方便的管理变量。 还可以利用 `tf.saved_model` 保存模型并实现跨平台部署使用。 

`tf.keras.models.Model`, `tf.keras.layers.Layer` 都是继承了 `tf.Module`

#### 3.5.1 应用 tf.Module 封装 AutoGraph

```python
class DemoModule(tf.Module):
    def __init__(self,init_value = tf.constant(0.0),name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:  #相当于with tf.name_scope("demo_module")
            self.x = tf.Variable(init_value,dtype = tf.float32,trainable=True)

     
    @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])  
    def addprint(self,a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return(self.x)

#执行
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))
```

- `demo.variables` , 查看全部变量
- `demo.trainable_variable`, 查看全部可训变量
- `demo.submodules`, 查看模块中的全部子模块

- `tf.saved_model.save(demo, "./data/demo/1", signatures={"serving_default":demo.addprint})`

  - 保存模型，并制定需要跨模型部署的方法

- `tf.saved_model.load("../data/demo/1")`  , 加载模型

- `!saved_model_cli show --dir ./data/demo/1 --all`

  - 查看模型的相关文件，输出信息在模型部署时可能会用到

- 查看计算图

  ```python
  import datetime
  
  # 创建日志
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  logdir = './data/demomodule/%s' % stamp
  writer = tf.summary.create_file_writer(logdir)
  
  #开启autograph跟踪
  tf.summary.trace_on(graph=True, profiler=True) 
  
  #执行autograph
  demo = DemoModule(init_value = tf.constant(0.0))
  result = demo.addprint(tf.constant(5.0))
  
  #将计算图信息写入日志
  with writer.as_default():
      tf.summary.trace_export(
          name="demomodule",
          step=0,
          profiler_outdir=logdir)
   
  #启动 tensorboard在jupyter中的魔法命令
  %reload_ext tensorboard
  
  from tensorboard import notebook
  notebook.list() 
  notebook.start("--logdir ./data/demomodule/")
  ```

### 3.5.2 tf.Module, tf.keras.Model 和 tf.keras.layers.Layer

`tf.keras`中的模型和层都是继承`tf.Module`实现的，也具有变量管理和子模块管理功能。



## 4. Tensorflow 的中阶 api

* 数据管道(tf.data)
* 特征列(tf.feature_column)
* 激活函数(tf.nn)
* 模型层(tf.keras.layers)
* 损失函数(tf.keras.losses)
* 评估函数(tf.keras.metrics)
* 优化器(tf.keras.optimizers)
* 回调函数(tf.keras.callbacks)

### 4.1 数据管道 Dataset

数据量不大时，例如不到1G，可以直接全部读入内存中进行训练

数据量很大，例如超过10G，无法一次载入内存，需要在训练过程中分批读入，可以使用 `tf.data` APO 构建数据输入管道

#### 4.1.1. 构建数据管道

- `tf.data.Dataset.from_tensor_slices((features, targets))`
  - 从 numpy 或者 pandas 读入数据

- `tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))`
  - 从 generator 构建数据管道
- `tf.data.experimental.make_csv_dataset()`
  - 从 csv 构建数据管道
- `tf.data.TextLineDataset`
  - 从文本文件 构建数据管道

#### 4.1.2 应用数据转换

* map: 将转换函数映射到数据集每一个元素。
* flat_map: 将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。
* interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。
* filter: 过滤掉某些元素。
* zip: 将两个长度相同的Dataset横向铰合。
* concatenate: 将两个Dataset纵向连接。
* reduce: 执行归并操作。
* batch : 构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。
* padded_batch: 构建批次，类似batch, 但可以填充到相同的形状。
* window :构建滑动窗口，返回Dataset of Dataset.
* shuffle: 数据顺序洗牌。
* repeat: 重复数据若干次，不带参数时，重复无数次。
* shard: 采样，从某个位置开始隔固定距离采样一个元素。
* take: 采样，从开始位置取前几个元素。

#### 4.1.3 提升管道性能

* 1，使用 prefetch 方法让数据准备和参数迭代两个过程相互并行。

* 2，使用 interleave 方法可以让数据读取过程多进程执行,并将不同来源数据夹在一起。

* 3，使用 map 时设置num_parallel_calls 让数据转换过程多进程执行。

* 4，使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。

* 5，使用 map转换时，先batch, 然后采用向量化的转换方法对每个batch进行转换。

### 4.2 特征列 feature_column

#### 4.2.1 特征列用法概述

将类别特征转化为 one-hot编码特征，将连续特征转化为分桶特征，以及交叉特征等等

* numeric_column 数值列，最常用。


* bucketized_column 分桶列，由数值列生成，可以由一个数值列出多个特征，one-hot编码。


* categorical_column_with_identity 分类标识列，one-hot编码，相当于分桶列每个桶为1个整数的情况。


* categorical_column_with_vocabulary_list 分类词汇列，one-hot编码，由list指定词典。


* categorical_column_with_vocabulary_file 分类词汇列，由文件file指定词典。


* categorical_column_with_hash_bucket 哈希列，整数或词典较大时采用。


* indicator_column 指标列，由Categorical Column生成，one-hot编码


* embedding_column 嵌入列，由Categorical Column生成，嵌入矢量分布参数需要学习。嵌入矢量维数建议取类别数量的 4 次方根。


* crossed_column 交叉列，可以由除categorical_column_with_hash_bucket的任意分类列构成。

### 4.3 激活函数

参考: [《一文概览深度学习中的激活函数》](https://zhuanlan.zhihu.com/p/98472075)

​	[《从ReLU到GELU,一文概览神经网络中的激活函数》](https://zhuanlan.zhihu.com/p/98863801)

#### 4.3.1 常用激活函数

- `tf.nn.sigmoid`
  - 压缩在0 -1之间，一般用于二分类的最后输出层使用
  - 主要缺点是存在梯度消失问题，计算复杂度高，输出不是0位中心
- `tf.nn.softmax`
- `tf.nn.tanh`
  - 压缩在 -1 ～1 之间，缺点是存在梯度消失问题，计算复杂度高
- `tf.nn.relu`
  - 输出不是 0 位中心，输入小于零存在梯度消失问题
- `tf.nn.leaky_relu`. 解决小于零梯度消失问题
- `tf.nn.elu`, 缓解 死亡relu问题
- `tf.nn.selu`
  - 需要和 AlphaDropout 一起使用
- `tf.nn.swish`, 门自控激活函数
- `gelu`, 在 transformer 中表现最好

#### 4.3.2 在模型中使用激活函数

1. 在某些层直接指定 activation 参数

   - ```
     layers.Dense(32, input_shape=(None, 16), activation=tf.nn.relu)
     ```

2. 显示的添加 layers.Activation 激活层

   - ```
     layers.Activation(tf.nn.softmax)
     ```

### 4.4 模型层

#### 4.4.1 内置模型层

##### 基础层

* `Dense`：密集连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)
* `Activation`：激活函数层。一般放在Dense层后面，等价于在Dense层中指定activation。
* `Dropout`：随机置零层。训练期间以一定几率将输入置0，一种正则化手段。
* `BatchNormalization`：批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差。可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。
* `SpatialDropout2D`：空间随机置零层。训练期间以一定几率将整个特征图置0，一种正则化手段，有利于避免特征图之间过高的相关性。
* `Input`：输入层。通常使用Functional API方式构建模型时作为第一层。
* `DenseFeature`：特征列接入层，用于接收一个特征列列表并产生一个密集连接层。
* `Flatten`：压平层，用于将多维张量压成一维。
* `Reshape`：形状重塑层，改变输入张量的形状。
* `Concatenate`：拼接层，将多个张量在某个维度上拼接。
* `Add`：加法层。
* `Subtract`： 减法层。
* `Maximum`：取最大值层。
* `Minimum`：取最小值层。

##### 卷积网络相关层

* `Conv1D`：普通一维卷积，常用于文本。参数个数 = 输入通道数×卷积核尺寸(如3)×卷积核个数
* `Conv2D`：普通二维卷积，常用于图像。参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数
* `Conv3D`：普通三维卷积，常用于视频。参数个数 = 输入通道数×卷积核尺寸(如3乘3乘3)×卷积核个数
* `SeparableConv2D`：二维深度可分离卷积层。不同于普通卷积同时对区域和通道操作，深度可分离卷积先操作区域，再操作通道。即先对每个通道做独立卷积操作区域，再用1乘1卷积跨通道组合操作通道。参数个数 = 输入通道数×卷积核尺寸 + 输入通道数×1×1×输出通道数。深度可分离卷积的参数数量一般远小于普通卷积，效果一般也更好。
* `DepthwiseConv2D`：二维深度卷积层。仅有SeparableConv2D前半部分操作，即只操作区域，不操作通道，一般输出通道数和输入通道数相同，但也可以通过设置depth_multiplier让输出通道为输入通道的若干倍数。输出通道数 = 输入通道数 × depth_multiplier。参数个数 = 输入通道数×卷积核尺寸× depth_multiplier。
* `Conv2DTranspose`：二维卷积转置层，俗称反卷积层。并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，卷积转置的输出尺寸恰好是卷积操作的输入尺寸。
* `LocallyConnected2D`: 二维局部连接层。类似Conv2D，唯一的差别是没有空间上的权值共享，所以其参数个数远高于二维卷积。
* `MaxPool2D`: 二维最大池化层。也称作下采样层。池化层无可训练参数，主要作用是降维。
* `AveragePooling2D`: 二维平均池化层。
* `GlobalMaxPool2D`: 全局最大池化层。每个通道仅保留一个值。一般从卷积层过渡到全连接层时使用，是Flatten的替代方案。
* `GlobalAvgPool2D`: 全局平均池化层。每个通道仅保留一个值。

##### 循环网络相关层

* `Embedding`：嵌入层。一种比Onehot更加有效的对离散特征进行编码的方法。一般用于将输入中的单词映射为稠密向量。嵌入层的参数需要学习。
* `LSTM`：长短记忆循环网络层。最普遍使用的循环网络层。具有携带轨道，遗忘门，更新门，输出门。可以较为有效地缓解梯度消失问题，从而能够适用长期依赖问题。设置return_sequences = True时可以返回各个中间步骤输出，否则只返回最终输出。
* `GRU`：门控循环网络层。LSTM的低配版，不具有携带轨道，参数数量少于LSTM，训练速度更快。
* `SimpleRNN`：简单循环网络层。容易存在梯度消失，不能够适用长期依赖问题。一般较少使用。
* `ConvLSTM2D`：卷积长短记忆循环网络层。结构上类似LSTM，但对输入的转换操作和对状态的转换操作都是卷积运算。
* `Bidirectional`：双向循环网络包装器。可以将LSTM，GRU等层包装成双向循环网络。从而增强特征提取能力。
* `RNN`：RNN基本层。接受一个循环网络单元或一个循环单元列表，通过调用tf.keras.backend.rnn函数在序列上进行迭代从而转换成循环网络层。
* `LSTMCell`：LSTM单元。和LSTM在整个序列上迭代相比，它仅在序列上迭代一步。可以简单理解LSTM即RNN基本层包裹LSTMCell。
* `GRUCell`：GRU单元。和GRU在整个序列上迭代相比，它仅在序列上迭代一步。
* `SimpleRNNCell`：SimpleRNN单元。和SimpleRNN在整个序列上迭代相比，它仅在序列上迭代一步。
* `AbstractRNNCell`：抽象RNN单元。通过对它的子类化用户可以自定义RNN单元，再通过RNN基本层的包裹实现用户自定义循环网络层。
* `Attention`：Dot-product类型注意力机制层。可以用于构建注意力模型。
* `AdditiveAttention`：Additive类型注意力机制层。可以用于构建注意力模型。
* `TimeDistributed`：时间分布包装器。包装后可以将Dense、Conv2D等作用到每一个时间片段上。

#### 4.4.2 自定义模型层

如果自定义模型层没有需要被训练的参数，一般推荐使用Lambda层实现

```python
mypower = layers.Lambda(lambda x: tf.math.pow(x, 2))
```

有需要被训练的参数，则可以通过对Layer基类子类化实现， 实现 build和 call方法

```python
class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    
    #build方法一般定义Layer需要被训练的参数。    
    def build(self, input_shape): 
        self.w = self.add_weight("w",shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True) #注意必须要有参数名称"w",否则会报错
        self.b = self.add_weight("b",shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
        super(Linear,self).build(input_shape) # 相当于设置self.built = True

    #call方法一般定义正向传播运算逻辑，__call__方法调用了它。  
    @tf.function
    def call(self, inputs): 
        return tf.matmul(inputs, self.w) + self.b
    
    #如果要让自定义的Layer通过Functional API 组合成模型时可以被保存成h5模型，需要自定义get_config方法。
    def get_config(self):  
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config
```

```
linear = Linear(uints = 8)
linear.build(input_shape=(None, 16))

# 或 直接. 无需使用None代表样本数量维
Linear(units = 8, input_shape=(16, ))
```

### 4.5 损失函数 losses

Objective = loss + regularization

目标函数的正则化项一般在各层中指定

- kernel_regularizer 和 bias_regularizer 等参数指定权重使用 l1或 l2 正则
- kernel_constraint 和 bias_constraint 等参数约束权重的取值范围

#### 4.5.1 损失函数和正则化项

```python
model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01), 
                activity_regularizer=regularizers.l1(0.01),
                kernel_constraint = constraints.MaxNorm(max_value=2, axis=0))) 
model.add(layers.Dense(10,
        kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))
model.compile(optimizer = "rmsprop",
        loss = "sparse_categorical_crossentropy",metrics = ["AUC"])
model.summary()
```

#### 4.5.2 内置损失函数

有两种实现，大写的是类的实现，小写的是函数的实现。 如：CategoricalCrossentropy 和 categorical_crossentropy

* mean_squared_error（均方误差损失，用于回归，简写为 mse, 类与函数实现形式分别为 MeanSquaredError 和 MSE）
* mean_absolute_error (平均绝对值误差损失，用于回归，简写为 mae, 类与函数实现形式分别为 MeanAbsoluteError 和 MAE)
* mean_absolute_percentage_error (平均百分比误差损失，用于回归，简写为 mape, 类与函数实现形式分别为 MeanAbsolutePercentageError 和 MAPE)
* Huber(Huber损失，只有类实现形式，用于回归，介于mse和mae之间，对异常值比较鲁棒，相对mse有一定的优势)
* binary_crossentropy(二元交叉熵，用于二分类，类实现形式为 BinaryCrossentropy)
* categorical_crossentropy(类别交叉熵，用于多分类，要求label为onehot编码，类实现形式为 CategoricalCrossentropy)
* sparse_categorical_crossentropy(稀疏类别交叉熵，用于多分类，要求label为序号编码形式，类实现形式为 SparseCategoricalCrossentropy)
* hinge(合页损失函数，用于二分类，最著名的应用是作为支持向量机SVM的损失函数，类实现形式为 Hinge)
* kld(相对熵损失，也叫KL散度，常用于最大期望算法EM的损失函数，两个概率分布差异的一种信息度量。类与函数实现形式分别为 KLDivergence 或 KLD)
* cosine_similarity(余弦相似度，可用于多分类，类实现形式为 CosineSimilarity)

#### 4.5.3 自定义损失函数

[如何评价Kaiming的Focal Loss for Dense Object Detection？》](https://www.zhihu.com/question/63581984)

```python
def focal_loss(gamma=2., alpha=0.25):
    
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(1e-07+pt_1)) \
           -tf.reduce_sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0 + 1e-07))
        return loss
    return focal_loss_fixed
```

```python
class FocalLoss(losses.Loss):
    
    def __init__(self,gamma=2.0,alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def call(self,y_true,y_pred):
        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss = -tf.reduce_sum(self.alpha * tf.pow(1. - pt_1, self.gamma) * tf.log(1e-07+pt_1)) \
           -tf.reduce_sum((1-self.alpha) * tf.pow( pt_0, self.gamma) * tf.log(1. - pt_0 + 1e-07))
        return loss
```

### 4.6 评估指标 metrics

通常损失函数都可以作为评估指标，如：MAE，MSE等

但是评估指标不一定可以作为损失函数，例如 AUC， Accuracy 等，因为其不要求连续可导

#### 4.6.1 常用的内置评估指标

* MeanSquaredError（均方误差，用于回归，可以简写为MSE，函数形式为mse）

* MeanAbsoluteError (平均绝对值误差，用于回归，可以简写为MAE，函数形式为mae)

* MeanAbsolutePercentageError (平均百分比误差，用于回归，可以简写为MAPE，函数形式为mape)

* RootMeanSquaredError (均方根误差，用于回归)

* Accuracy (准确率，用于分类，可以用字符串"Accuracy"表示，Accuracy=(TP+TN)/(TP+TN+FP+FN)，要求y_true和y_pred都为类别序号编码)

* Precision (精确率，用于二分类，Precision = TP/(TP+FP))

* Recall (召回率，用于二分类，Recall = TP/(TP+FN))

* TruePositives (真正例，用于二分类)

* TrueNegatives (真负例，用于二分类)

* FalsePositives (假正例，用于二分类)

* FalseNegatives (假负例，用于二分类)

* AUC(ROC曲线(TPR vs FPR)下的面积，用于二分类，直观解释为随机抽取一个正样本和一个负样本，正样本的预测值大于负样本的概率)

* CategoricalAccuracy（分类准确率，与Accuracy含义相同，要求y_true(label)为onehot编码形式）

* SparseCategoricalAccuracy (稀疏分类准确率，与Accuracy含义相同，要求y_true(label)为序号编码形式)

* MeanIoU (Intersection-Over-Union，常用于图像分割)

* TopKCategoricalAccuracy (多分类TopK准确率，要求y_true(label)为onehot编码形式)

* SparseTopKCategoricalAccuracy (稀疏多分类TopK准确率，要求y_true(label)为序号编码形式)

* Mean (平均值)

* Sum (求和)

#### 4.6.2 自定义评估指标

e.g.： KS=max(TPR-FPR)

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers,models,losses,metrics

#函数形式的自定义评估指标
@tf.function
def ks(y_true,y_pred):
    y_true = tf.reshape(y_true,(-1,))
    y_pred = tf.reshape(y_pred,(-1,))
    length = tf.shape(y_true)[0]
    t = tf.math.top_k(y_pred,k = length,sorted = False)
    y_pred_sorted = tf.gather(y_pred,t.indices)
    y_true_sorted = tf.gather(y_true,t.indices)
    cum_positive_ratio = tf.truediv(
        tf.cumsum(y_true_sorted),tf.reduce_sum(y_true_sorted))
    cum_negative_ratio = tf.truediv(
        tf.cumsum(1 - y_true_sorted),tf.reduce_sum(1 - y_true_sorted))
    ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio)) 
    return ks_value
```

```python
#类形式的自定义评估指标
class KS(metrics.Metric):
    
    def __init__(self, name = "ks", **kwargs):
        super(KS,self).__init__(name=name,**kwargs)
        self.true_positives = self.add_weight(
            name = "tp",shape = (101,), initializer = "zeros")
        self.false_positives = self.add_weight(
            name = "fp",shape = (101,), initializer = "zeros")
   
    @tf.function
    def update_state(self,y_true,y_pred):
        y_true = tf.cast(tf.reshape(y_true,(-1,)),tf.bool)
        y_pred = tf.cast(100*tf.reshape(y_pred,(-1,)),tf.int32)
        
        for i in tf.range(0,tf.shape(y_true)[0]):
            if y_true[i]:
                self.true_positives[y_pred[i]].assign(
                    self.true_positives[y_pred[i]]+1.0)
            else:
                self.false_positives[y_pred[i]].assign(
                    self.false_positives[y_pred[i]]+1.0)
        return (self.true_positives,self.false_positives)
    
    @tf.function
    def result(self):
        cum_positive_ratio = tf.truediv(
            tf.cumsum(self.true_positives),tf.reduce_sum(self.true_positives))
        cum_negative_ratio = tf.truediv(
            tf.cumsum(self.false_positives),tf.reduce_sum(self.false_positives))
        ks_value = tf.reduce_max(tf.abs(cum_positive_ratio - cum_negative_ratio)) 
        return ks_value

```

### 4.7 优化器 optimizers

#### 4.7.1 优化器的使用

主要使用 `apply_gradients` 方法传入变量和梯度， 或者直接使用 `minimize`

最常见是编译时将优化器传入 keras的Model， 调用 `model.fit` 实现对loss 的迭代

初始化优化器时会创建一个变量`optimier.iterations`用于记录迭代的次数。因此优化器和`tf.Variable`一样，一般需要在`@tf.function`外创建。

- ```python
  # 求f(x) = a*x**2 + b*x + c的最小值
  
  # 使用optimizer.apply_gradients
  
  x = tf.Variable(0.0,name = "x",dtype = tf.float32)
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  
  @tf.function
  def minimizef():
      a = tf.constant(1.0)
      b = tf.constant(-2.0)
      c = tf.constant(1.0)
      
      while tf.constant(True): 
          with tf.GradientTape() as tape:
              y = a*tf.pow(x,2) + b*x + c
          dy_dx = tape.gradient(y,x)
          optimizer.apply_gradients(grads_and_vars=[(dy_dx,x)])
          
          #迭代终止条件
          if tf.abs(dy_dx)<tf.constant(0.00001):
              break
              
          if tf.math.mod(optimizer.iterations,100)==0:
              printbar()
              tf.print("step = ",optimizer.iterations)
              tf.print("x = ", x)
              tf.print("")
                  
      y = a*tf.pow(x,2) + b*x + c
      return y
  
  tf.print("y =",minimizef())
  tf.print("x =",x)
  ```

- ```python
  # 求f(x) = a*x**2 + b*x + c的最小值
  
  # 使用optimizer.minimize
  
  x = tf.Variable(0.0,name = "x",dtype = tf.float32)
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
  
  def f():   
      a = tf.constant(1.0)
      b = tf.constant(-2.0)
      c = tf.constant(1.0)
      y = a*tf.pow(x,2)+b*x+c
      return(y)
  
  @tf.function
  def train(epoch = 1000):  
      for _ in tf.range(epoch):  
          optimizer.minimize(f,[x])
      tf.print("epoch = ",optimizer.iterations)
      return(f())
  
  train(1000)
  tf.print("y = ",f())
  tf.print("x = ",x)
  
  ```

- ```python
  # 求f(x) = a*x**2 + b*x + c的最小值
  # 使用model.fit
  
  tf.keras.backend.clear_session()
  
  class FakeModel(tf.keras.models.Model):
      def __init__(self,a,b,c):
          super(FakeModel,self).__init__()
          self.a = a
          self.b = b
          self.c = c
      
      def build(self):
          self.x = tf.Variable(0.0,name = "x")
          self.built = True
      
      def call(self,features):
          loss  = self.a*(self.x)**2+self.b*(self.x)+self.c
          return(tf.ones_like(features)*loss)
      
  def myloss(y_true,y_pred):
      return tf.reduce_mean(y_pred)
  
  model = FakeModel(tf.constant(1.0),tf.constant(-2.0),tf.constant(1.0))
  
  model.build()
  model.summary()
  
  model.compile(optimizer = 
                tf.keras.optimizers.SGD(learning_rate=0.01),loss = myloss)
  history = model.fit(tf.zeros((100,2)),
                      tf.ones(100),batch_size = 1,epochs = 10)  #迭代1000次
  
  ```

#### 4.7.2 内置优化器

* SGD, 默认参数为纯SGD, 设置momentum参数不为0实际上变成SGDM, 考虑了一阶动量, 设置 nesterov为True后变成NAG，即 Nesterov Accelerated Gradient，在计算梯度时计算的是向前走一步所在位置的梯度。
* Adagrad, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率。缺点是学习率单调下降，可能后期学习速率过慢乃至提前停止学习。
* RMSprop, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对Adagrad进行了优化，通过指数平滑只考虑一定窗口内的二阶动量。
* Adadelta, 考虑了二阶动量，与RMSprop类似，但是更加复杂一些，自适应性更强。
* Adam, 同时考虑了一阶动量和二阶动量，可以看成RMSprop上进一步考虑了一阶动量。
* Nadam, 在Adam基础上进一步考虑了 Nesterov Acceleration。



### 4.8 回调函数 callbacks

tf.keras的回调函数实际上是一个类，一般是在model.fit时作为参数指定，用于控制在训练过程开始或者在训练过程结束，在每个epoch训练开始或者训练结束，在每个batch训练开始或者训练结束时执行一些操作，例如收集一些日志信息，改变学习率等超参数，提前终止训练过程等等。

同样地，针对model.evaluate或者model.predict也可以指定callbacks参数，用于控制在评估或预测开始或者结束时，在每个batch开始或者结束时执行一些操作，但这种用法相对少见。

大部分时候，keras.callbacks子模块中定义的回调函数类已经足够使用了，如果有特定的需要，我们也可以通过对keras.callbacks.Callbacks实施子类化构造自定义的回调函数。

所有回调函数都继承至 keras.callbacks.Callbacks基类，拥有params和model这两个属性。

其中params 是一个dict，记录了训练相关参数 (例如 verbosity, batch size, number of epochs 等等)。

model即当前关联的模型的引用。

此外，对于回调类中的一些方法如on_epoch_begin,on_batch_end，还会有一个输入参数logs, 提供有关当前epoch或者batch的一些信息，并能够记录计算结果，如果model.fit指定了多个回调函数类，这些logs变量将在这些回调函数类的同名函数间依顺序传递。

#### 4.8.1 内置回调函数

* BaseLogger： 收集每个epoch上metrics在各个batch上的平均值，对stateful_metrics参数中的带中间状态的指标直接拿最终值无需对各个batch平均，指标均值结果将添加到logs变量中。该回调函数被所有模型默认添加，且是第一个被添加的。
* History： 将BaseLogger计算的各个epoch的metrics结果记录到history这个dict变量中，并作为model.fit的返回值。**该回调函数被所有模型默认添加**，在BaseLogger之后被添加。
* EarlyStopping： 当被监控指标在设定的若干个epoch后没有提升，则提前终止训练。
* TensorBoard： 为Tensorboard可视化保存日志信息。支持评估指标，计算图，模型参数等的可视化。
* ModelCheckpoint： 在每个epoch后保存模型。
* ReduceLROnPlateau：如果监控指标在设定的若干个epoch后没有提升，则以一定的因子减少学习率。
* TerminateOnNaN：如果遇到loss为NaN，提前终止训练。
* LearningRateScheduler：学习率控制器。给定学习率lr和epoch的函数关系，根据该函数关系在每个epoch前调整学习率。
* CSVLogger：将每个epoch后的logs结果记录到CSV文件中。
* ProgbarLogger：将每个epoch后的logs结果打印到标准输出流中。

#### 4.8.2 自定义回调函数

可以使用callbacks.LambdaCallback编写较为简单的回调函数，也可以通过对callbacks.Callback子类化编写更加复杂的回调函数逻辑。

- ```python
  # 示范使用LambdaCallback编写较为简单的回调函数
  
  import json
  json_log = open('./data/keras_log.json', mode='wt', buffering=1)
  json_logging_callback = callbacks.LambdaCallback(
      on_epoch_end=lambda epoch, logs: json_log.write(
          json.dumps(dict(epoch = epoch,**logs)) + '\n'),
      on_train_end=lambda logs: json_log.close()
  )
  ```

- ```
  # 示范通过Callback子类化编写回调函数（LearningRateScheduler的源代码）
  
  class LearningRateScheduler(callbacks.Callback):
      
      def __init__(self, schedule, verbose=0):
          super(LearningRateScheduler, self).__init__()
          self.schedule = schedule
          self.verbose = verbose
  
      def on_epoch_begin(self, epoch, logs=None):
          if not hasattr(self.model.optimizer, 'lr'):
              raise ValueError('Optimizer must have a "lr" attribute.')
          try:  
              lr = float(K.get_value(self.model.optimizer.lr))
              lr = self.schedule(epoch, lr)
          except TypeError:  # Support for old API for backward compatibility
              lr = self.schedule(epoch)
          if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
              raise ValueError('The output of the "schedule" function '
                               'should be float.')
          if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
              raise ValueError('The dtype of Tensor should be float')
          K.set_value(self.model.optimizer.lr, K.get_value(lr))
          if self.verbose > 0:
              print('\nEpoch %05d: LearningRateScheduler reducing learning '
                   'rate to %s.' % (epoch + 1, lr))
  
      def on_epoch_end(self, epoch, logs=None):
          logs = logs or {}
          logs['lr'] = K.get_value(self.model.optimizer.lr)
  ```



## 5. Tensorflow的高阶 api

* 模型的构建（Sequential、functional API、Model子类化）
* 模型的训练（内置fit方法、内置train_on_batch方法、自定义训练循环、单GPU训练模型、多GPU训练模型、TPU训练模型）
* 模型的部署（tensorflow serving部署模型、使用spark(scala)调用tensorflow模型）



### 5.1 构建模型的3种方法

1. 使用Sequential按层顺序构建模型
   - 对于顺序结构的模型，优先使用Sequential方法构建。
2. 使用函数式API构建任意结构模型
   - 模型有多输入或者多输出，或者模型需要共享权重，或者模型具有残差连接等非顺序结构
3. 继承Model基类构建自定义模型。
   - 如果无特定必要，尽可能避免使用Model子类化的方式构建模型，这种方式提供了极大的灵活性，但也有更大的概率出错。

#### 5.1.1 Sequential按层顺序创建模型

```python
tf.keras.backend.clear_session()

model = models.Sequential()

model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
model.add(layers.MaxPool1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation = "sigmoid"))

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()
```

```python
import datetime
baselogger = callbacks.BaseLogger(stateful_metrics=["AUC"])
logdir = "./data/keras_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
history = model.fit(ds_train,validation_data = ds_test,
        epochs = 6,callbacks=[baselogger,tensorboard_callback])
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history,"AUC")
```

#### 5.1.2 函数式API创建任意结构模型

```python
tf.keras.backend.clear_session()

inputs = layers.Input(shape=[MAX_LEN])
x  = layers.Embedding(MAX_WORDS,7)(inputs)

branch1 = layers.SeparableConv1D(64,3,activation="relu")(x)
branch1 = layers.MaxPool1D(3)(branch1)
branch1 = layers.SeparableConv1D(32,3,activation="relu")(branch1)
branch1 = layers.GlobalMaxPool1D()(branch1)

branch2 = layers.SeparableConv1D(64,5,activation="relu")(x)
branch2 = layers.MaxPool1D(5)(branch2)
branch2 = layers.SeparableConv1D(32,5,activation="relu")(branch2)
branch2 = layers.GlobalMaxPool1D()(branch2)

branch3 = layers.SeparableConv1D(64,7,activation="relu")(x)
branch3 = layers.MaxPool1D(7)(branch3)
branch3 = layers.SeparableConv1D(32,7,activation="relu")(branch3)
branch3 = layers.GlobalMaxPool1D()(branch3)

concat = layers.Concatenate()([branch1,branch2,branch3])
outputs = layers.Dense(1,activation = "sigmoid")(concat)

model = models.Model(inputs = inputs,outputs = outputs)

model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])

model.summary()

```

#### 5.1.3 Model子类化创建自定义模型

```python
# 先自定义一个残差模块，为自定义Layer

class ResBlock(layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
    
    def build(self,input_shape):
        self.conv1 = layers.Conv1D(filters=64,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv2 = layers.Conv1D(filters=32,kernel_size=self.kernel_size,
                                   activation = "relu",padding="same")
        self.conv3 = layers.Conv1D(filters=input_shape[-1],
                                   kernel_size=self.kernel_size,activation = "relu",padding="same")
        self.maxpool = layers.MaxPool1D(2)
        super(ResBlock,self).build(input_shape) # 相当于设置self.built = True
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = layers.Add()([inputs,x])
        x = self.maxpool(x)
        return x
    
    #如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):  
        config = super(ResBlock, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config
```

```python
# 测试ResBlock
resblock = ResBlock(kernel_size = 3)
resblock.build(input_shape = (None,200,7))
resblock.compute_output_shape(input_shape=(None,200,7))
```

```python
# 自定义模型，实际上也可以使用Sequential或者FunctionalAPI

class ImdbModel(models.Model):
    def __init__(self):
        super(ImdbModel, self).__init__()
        
    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS,7)
        self.block1 = ResBlock(7)
        self.block2 = ResBlock(5)
        self.dense = layers.Dense(1,activation = "sigmoid")
        super(ImdbModel,self).build(input_shape)
    
    def call(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = layers.Flatten()(x)
        x = self.dense(x)
        return(x)
```

### 5.2 训练模型的3种方法

模型的训练主要有内置fit方法、内置tran_on_batch方法、自定义训练循环

#### 5.2.1 内置 fit 方法

该方法功能非常强大, 支持对`numpy array`, `tf.data.Datase`t以及 `Python generator`数据进行训练。

并且**可以通过设置回调函数**实现对训练过程的复杂控制逻辑。

```python
tf.keras.backend.clear_session()
def create_model():
    
    model = models.Sequential()
    model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
    model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM,activation = "softmax"))
    return(model)

def compile_model(model):
    model.compile(optimizer=optimizers.Nadam(),
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=[metrics.SparseCategoricalAccuracy(),metrics.SparseTopKCategoricalAccuracy(5)]) 
    return(model)
 
model = create_model()
model.summary()
model = compile_model(model)  # 自定义方法不需要
```

```
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)
```

#### 5.2.2 内置train_on_batch方法

```python
def train_model(model,ds_train,ds_valid,epoches):

    for epoch in tf.range(1,epoches+1):
        model.reset_metrics()
        
        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("Lowering optimizer Learning Rate...\n\n")
        
        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y,reset_metrics=False)
            
        if epoch%1 ==0:
            printbar()
            tf.print("epoch = ",epoch)
            print("train:",dict(zip(model.metrics_names,train_result)))
            print("valid:",dict(zip(model.metrics_names,valid_result)))
            print("")

train_model(model,ds_train,ds_test,10)
```

#### 5.2.3 自定义训练循环

自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性。

```python
optimizer = optimizers.Nadam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features,training = True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)
    

@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)
    

def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        
        for features, labels in ds_train:
            train_step(model,features,labels)

        for features, labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
            tf.print("")
            
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model,ds_train,ds_test,10)

```

### 5.3 使用单GPU训练模型

[《用GPU加速Keras模型——Colab免费GPU使用攻略》](https://zhuanlan.zhihu.com/p/68509398)

tensorflow默认获取全部GPU的全部内存资源权限，但实际上只使用一个GPU的部分资源

#### 5.3.1 GPU 设置

```python
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    # 或者也可以设置GPU显存为固定使用量(例如：4G)
    #tf.config.experimental.set_virtual_device_configuration(gpu0,
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
    tf.config.set_visible_devices([gpu0],"GPU") 
```

### 5.4 使用多GPU训练模型

MirroredStrategy过程简介：

1. 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型；
2. 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）；
3. N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度；
4. 使用分布式计算的 All-reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和；
5. 使用梯度求和的结果更新本地变量（镜像变量）；
6. 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）。

```python
#此处在colab上使用1个GPU模拟出两个逻辑GPU进行多GPU训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑GPU模拟多GPU训练
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
```

**更改训练模型时的步骤**

```python
#增加以下两行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)
    
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)  
```

### 5.5 使用TPU训练数据

**更改训练模型时的步骤**

```python
#增加以下6行代码
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
    model = create_model()
    model.summary()
    model = compile_model(model)
    
```

### 5.6 使用tensorflow-serving部署模型

TensorFlow训练好的模型以tensorflow原生方式保存成protobuf(.pb)文件后可以用许多方式部署运行。

例如：通过 tensorflow-js 可以用javascrip脚本加载模型并在浏览器中运行模型。

通过 tensorflow-lite 可以在移动和嵌入式设备上加载并运行TensorFlow模型。

通过 tensorflow-serving 可以加载模型后提供网络接口API服务，通过任意编程语言发送网络请求都可以获取模型预测结果。

通过 tensorFlow for Java接口，可以在Java或者spark(scala)中调用tensorflow模型进行预测。

参考: [简单粗暴Tensorflow2.0](https://tf.wiki/)

### 5.7 使用spark-scala调用tensorflow2.0训练好的模型

利用spark的分布式计算能力，从而可以让训练好的tensorflow模型在成百上千的机器上分布式并行执行模型推断。

在spark(scala)中调用tensorflow模型进行预测需要完成以下几个步骤。

（1）准备protobuf模型文件

（2）创建spark(scala)项目，在项目中添加java版本的tensorflow对应的jar包依赖

（3）在spark(scala)项目中driver端加载tensorflow模型调试成功

（4）在spark(scala)项目中通过RDD在executor上加载tensorflow模型调试成功

（5） 在spark(scala)项目中通过DataFrame在executor上加载tensorflow模型调试成功


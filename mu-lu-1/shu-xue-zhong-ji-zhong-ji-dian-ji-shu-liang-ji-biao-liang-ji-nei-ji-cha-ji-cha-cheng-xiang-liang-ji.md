---
layout: landing
---

# 数学中几种积：点积(数量积/标量积/内积)、叉积(叉乘/向量积)、外积(张量积/Kronecker积)、哈达玛积(元素积)\_rosefunR的博客-CSDN博客\_元素积

#### 1 点积 <a href="#t0" id="t0"></a>

点积（dot product)，又称**数量积、标量积**.

**输入：** 一种接受两个等长的数字序列（通常是坐标向量）；\
**输出**：返回单个数字。

在欧几里几何空间中，向量的点积运算又称为**内积**。

**表示**\
\
**代数定义**\
\
**推广**\
矩阵的点积/[内积](https://so.csdn.net/so/search?q=%E5%86%85%E7%A7%AF\&spm=1001.2101.3001.7020)，为对应矩阵元素的积之和。

A，B是定义为两个相同大小的矩阵。\
\
值得注意的是，一些对于A，B大小不同，可以分别把它们组成的向量进行内积。\
比如在[numpy](https://so.csdn.net/so/search?q=numpy\&spm=1001.2101.3001.7020)中：

```
import numpy
x = numpy.mat([[1, 2], [3, 4]])
y = numpy.mat([10, 20])
print("Matrix inner:")
print(numpy.inner(x, y))
''' Output：
Matrix inner:
[[ 50]
 [110]]
'''
12345678910
```

#### 2 叉积 <a href="#t1" id="t1"></a>

叉积（Cross product），又称**向量积**（Vector product）、**叉乘**。

**输入：** 对三维空间中的两个向量；

**输出：** 返回一个向量；

**表示**\
\
**代数定义**

叉积 a × b {\displaystyle \mathbf {a} \times \mathbf {b} } a×b 是与 a {\displaystyle \mathbf {a} } a 和 b {\displaystyle \mathbf {b} } b都垂直的向量 c {\displaystyle \mathbf {c} } c 。

其方向由右手定则决定，模长等于以两个向量为边的平行四边形的面积。

\
\
n {\displaystyle \mathbf {n} } n 是与 a， b都垂直的单位向量。

**推广**\
\
矩阵表示：\
\


#### 3 外积 <a href="#t2" id="t2"></a>

外积（Outer product） ，又名**张量积**。\
外积与向量的内积相对， 是矩阵的克罗内克积的一种特例。

**输入：** 两个向量。

**输出：** 矩阵。

**表示**\


**代数定义**\
\
\
**推广**

矩阵的外积：**克罗内克积（Kronecker product）**\
如果A是一个 m × n 的矩阵，而B是一个 p × q 的矩阵，克罗内克积 A ⊗ B {\displaystyle A\otimes B} A⊗B则是一个 m p × n q mp × nq mp×nq 的分块矩阵.

示例：\


#### 4 哈达玛乘积 (矩阵) <a href="#t3" id="t3"></a>

哈达玛积（Hadamard product） ，又名**舒尔积**或**逐项积**。

在机器学习中，哈达玛积还称为，**元素积**(element-wise product/point-wise product）。

**输入：** 两个相同形状的矩阵。

**输出：** 具有同样形状的、各个位置的元素等于两个输入矩阵相同位置元素的乘积的矩阵。

**表示**\
A ∘ B A ∘ B A∘B

**代数定义**\


**推广**

如果矩阵维度不一样，矩阵/向量的哈达玛积计算如下：\


***

参考：

1. [矩阵运算](https://www.cnblogs.com/steven-yang/p/6348112.html)；
2. [wiki 点积](https://zh.wikipedia.org/wiki/%E7%82%B9%E7%A7%AF);
3. [wiki 叉积](https://zh.wikipedia.org/wiki/%E5%8F%89%E7%A7%AF)；
4. [wiki 哈达玛乘积](https://zh.wikipedia.org/wiki/%E9%98%BF%E9%81%94%E7%91%AA%E4%B9%98%E7%A9%8D\_\(%E7%9F%A9%E9%99%A3\))；
5. [wiki 外积](https://zh.wikipedia.org/wiki/%E5%A4%96%E7%A7%AF)；
6. [克劳内克积](https://zh.wikipedia.org/wiki/%E5%85%8B%E7%BD%97%E5%86%85%E5%85%8B%E7%A7%AF)；

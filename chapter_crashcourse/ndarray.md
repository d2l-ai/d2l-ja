# データの取り扱い

データを操作できなければ、何も行うことはできません。一般的に、データを処理する上で重要なことは次の2点です。（i）データを獲得すること、および（ii）コンピュータに取り込んだらそれを処理することです。データの保存方法さえわからなければ、データを獲得することに意味がありません。合成データを利用して実際に触ってみましょう。まず、NDArrayというデータを保存・変換するためのMXNetの主要なツールを紹介します。 NumPyを以前に使用したことがあれば、NDArrayは設計上、NumPyの多次元配列に似ていることがわかります。ただし、NDArrayにはいくつかの重要な利点があります。まず、NDArrayは、CPU、GPU、および分散クラウドアーキテクチャでの非同期計算をサポートしています。第二に、それらは自動微分をサポートしています。これらの特性によって、NDArrayは深層学習に必要不可欠なものとなっています。

## まずはじめに

この章を通して、読者の最初のステップを支援することを目的とし、基本的な機能について話を進めていきます。Element-wiseな演算や正規分布など、基本的な数学のすべてを理解していなくても心配しないでください。以降の2つの章では、同じコンテンツについて別の見方をし、実践的な例にもとづいてその内容を解説します。一方、数学的な内容を詳しく知りたい場合は、付録の["Math"](../chapter_appendix/math.md) のセクションを参照してください。

MXNetとMXNetから `ndarray`モジュールをインポートすることから始めます。ここで、`nd` は `ndarray` の短縮形です。


```{.python .input  n=1}
import mxnet as mx
from mxnet import nd
```

NDArraysは数値の (多次元の) 配列を表します。 1軸のNDArrayは(数学的には)*vector*に対応します。2軸のNDArrayは*行列*に対応します。3つ以上の軸を持つ配列に関しては、数学者は特別な名前を与えていません - 単にそれらを*テンソル*と呼びます。

作成できる最も単純なオブジェクトはベクトルです。まず始めに、 `arange`を使って12個の連続した整数をもつ行ベクトルを作りましょう。


```{.python .input  n=2}
x = nd.arange(12)
x
```
`x`を標準出力すると`<NDArray 12 @cpu（0）>`というプロパティを見ることができます。これは`x`が長さ12の1次元配列であり、それがCPUのメインメモリにあることを示します。 `@cpu(0)`の0は特別な意味を持たず、特定のコアを表すものでもありません。

NDArrayインスタンスの形状は `shape`のプロパティを利用して確認することができます。

```{.python .input  n=8}
x.shape
```

`size`のプロパティから、NDArrayインスタンスの要素の総数を得ることもできます。これは`shape`の要素の積となります。ここではベクトルを扱っているので、`size`も`shape`も同じ数になります。


```{.python .input  n=9}
x.size
```

ある一つの(多次元の)配列のshapeを、同じ数の要素を含む別のものに変えるためには`reshape`関数を使います。
たとえば、行ベクトル`x`のshapeを(3, 4)に変換できます。これは同じ値を含みますが、3行4列の行列として解釈されます。shapeは変わっていますが、`x`の要素は変わっていないことに注意してください。`size`は同じままです。


```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

各次元をそれぞれ手動で指定してreshapeすることは面倒なことがあります。一方の次元がわかっていれば、もう一方の次元を決定するために、わざわざ割り算を実行する必要があるでしょうか? たとえば、上の例では、3行の行列を取得するために、4列をもつように別途指定する必要がありました（12要素を考慮して）。幸いなことに、NDArrayは自動的に一方の次元から他方の次元を決定することができます。 NDArrayに自動的に推測させたい次元に `-1`を配置します。さきほどの例では `x.reshape((3, 4))`の代わりに、 `x.reshape((-1, 4))`または `x.reshape((3,-1))`を使用することが可能です。


```{.python .input}
nd.empty((3, 4))
```
`empty`のメソッドは、いくらかのメモリを確保して、その要素に対していずれの値も設定せずに行列を返します。これは非常に効率的ですが、各要素は非常に大きな値も含め、任意の値を取る可能性があります。通常は、行列を、1、ゼロ、既知の定数、または既知の分布から無作為に抽出された数値のいずれかで初期化しようとするでしょう。

そして、ほとんどの場合、すべてゼロの配列を必要とするでしょう。すべての要素が0、shapeが(2,3,4)であるようなテンソルを表すNDArrayを作成するには、以下を実行します。

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

すべての要素が1であるようなテンソルを作成するためには以下を実行します。

```{.python .input  n=5}
nd.ones((2, 3, 4))
```
数値の値を含む Python のリストを与えることで、特定の値を要素にもつNDArrayを作成することもできます。

```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

場合によっては、既知の確率分布に従って、NDArrayの各要素の値をランダムにサンプリングすることもあるでしょう。これは、ニューラルネットワークにおけるパラメータとして、配列を使用しようとする際に特に一般的に行われています。次のスニペットは、(3, 4)の形状をもつNDArrayを作成します。その要素は平均がゼロで分散が1の正規分布から無作為にサンプリングされた値をもちます。

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

## Operations

Oftentimes, we want to apply functions to arrays. Some of the simplest and most useful functions are the element-wise functions. These operate by performing a single scalar operation on the corresponding elements of two arrays. We can create an element-wise function from any function that maps from the scalars to the scalars. In math notations we would denote such a function as $f: \mathbb{R} \rightarrow \mathbb{R}$. Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and the function f,
we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$. Here, we produced the vector-valued $F: \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an element-wise vector operation. In MXNet, the common standard arithmetic operators (+,-,/,\*,\*\*) have all been *lifted* to element-wise operations for identically-shaped tensors of arbitrary shape. We can call element-wise operations on any two tensors of the same shape, including matrices.

```{.python .input}
x = nd.array([1, 2, 4, 8])
y = nd.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x / y', x / y)
```

Many more operations can be applied element-wise, such as exponentiation:

```{.python .input  n=12}
x.exp()
```

In addition to computations by element, we can also perform matrix operations, like matrix multiplication using the `dot` function. Next, we will perform matrix multiplication of `x` and the transpose of `y`. We define `x` as a matrix of 3 rows and 4 columns, and `y` is transposed into a matrix of 4 rows and 3 columns. The two matrices are multiplied to obtain a matrix of 3 rows and 3 columns (if you are confused about what this means, do not worry - we will explain matrix operations in much more detail in the chapter on [linear algebra](linear-algebra.md)).

```{.python .input  n=13}
x = nd.arange(12).reshape((3,4))
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
nd.dot(x, y.T)
```

We can also merge multiple NDArrays. For that, we need to tell the system along which dimension to merge. The example below merges two matrices along dimension 0 (along rows) and dimension 1 (along columns) respectively.

```{.python .input}
nd.concat(x, y, dim=0)
nd.concat(x, y, dim=1)
```

Sometimes, we may want to construct binary NDArrays via logical statements. Take `x == y` as an example. If `x` and `y` are equal for some entry, the new NDArray has a value of 1 at the same position; otherwise it is 0.

```{.python .input}
x == y
```

Summing all the elements in the NDArray yields an NDArray with only one element.

```{.python .input}
x.sum()
```

We can transform the result into a scalar in Python using the `asscalar` function. In the following example, the $\ell_2$ norm of `x` yields a single element NDArray. The final result is transformed into a scalar.

```{.python .input}
x.norm().asscalar()
```

For stylistic convenience, we can write `y.exp()`, `x.sum()`, `x.norm()`, etc. also as `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)`.

## Broadcast Mechanism

In the above section, we saw how to perform operations on two NDArrays of the same shape. When their shapes differ, a broadcasting mechanism may be triggered analogous to NumPy: first, copy the elements appropriately so that the two NDArrays have the same shape, and then carry out operations by element.

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

Since `a` and `b` are (3x1) and (1x2) matrices respectively, their shapes do not match up if we want to add them. NDArray addresses this by 'broadcasting' the entries of both matrices into a larger (3x2) matrix as follows: for matrix `a` it replicates the columns, for matrix `b` it replicates the rows before adding up both element-wise.

```{.python .input}
a + b
```

## Indexing and Slicing

Just like in any other Python array, elements in an NDArray can be accessed by its index. In good Python tradition the first element has index 0 and ranges are specified to include the first but not the last element. By this logic `1:3` selects the second and third element. Let's try this out by selecting the respective rows in a matrix.

```{.python .input  n=19}
x[1:3]
```

Beyond reading, we can also write elements of a matrix.

```{.python .input  n=20}
x[1, 2] = 9
x
```

If we want to assign multiple elements the same value, we simply index all of them and then assign them the value. For instance, `[0:2, :]` accesses the first and second rows. While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than 2 dimensions.

```{.python .input  n=21}
x[0:2, :] = 12
x
```

## Saving Memory

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and instead point it at the newly allocated memory. In the following example we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory. After running `y = y + x`, we will find that `id(y)` points to a different location. That is because Python first evaluates `y + x`, allocating new memory for the result and then subsequently redirects `y` to point at this new location in memory.

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons. First, we do not want to run around allocating memory unnecessarily all the time. In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second. Typically, we will want to perform these updates *in place*. Second, we might point at the same parameters from multiple variables. If we do not update in place, this could cause a memory leak, making it possible for us to inadvertently reference stale parameters.

Fortunately, performing in-place operations in MXNet is easy. We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`. To illustrate the behavior, we first clone the shape of a matrix using `zeros_like` to allocate a block of 0 entries.

```{.python .input  n=16}
z = y.zeros_like()
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

While this looks pretty, `x+y` here will still allocate a temporary buffer to store the result of `x+y` before copying it to `y[:]`. To make even better use of memory, we can directly invoke the underlying `ndarray` operation, in this case `elemwise_add`, avoiding temporary buffers. We do this by specifying the `out` keyword argument, which every `ndarray` operator supports:

```{.python .input  n=17}
before = id(z)
nd.elemwise_add(x, y, out=z)
id(z) == before
```

If the value of `x ` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## Mutual Transformation of NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do *not* share memory. This minor inconvenience is actually quite important: when you perform operations on the CPU or one of the GPUs, you do not want MXNet having to wait whether NumPy might want to be doing something else with the same chunk of memory. The  `array` and `asnumpy` functions do the trick.

```{.python .input  n=22}
import numpy as np

a = x.asnumpy()
print(type(a))
b = nd.array(a)
print(type(b))
```

## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of NDArray you can get.
1. Replace the two NDArrays that operate by element in the broadcast mechanism with other shapes, e.g. three dimensional tensors. Is the result the same as expected?
1. Assume that we have three matrices `a`, `b` and `c`. Rewrite `c = nd.dot(a, b.T) + c` in the most memory efficient manner.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)

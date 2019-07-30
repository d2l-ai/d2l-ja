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

## 演算

配列に対して関数を適用したいときは多いと思います。最も単純かつ便利な機能として要素ごとの (element-wise)の機能が挙げられます。これらは、2つの配列の対応する要素に対して、単一のスカラー演算を実行します。スカラーからスカラーへ写像するあらゆる関数に対して、element-wiseな関数を作成することができます。数学的な記法を使うと、以下の用に記述することができます: $f: \mathbb{R} \rightarrow \mathbb{R}$. 同じshapeの2つのベクトル$\mathbf{u}$と$\mathbf{v}$、関数$f$が与えられているとき、すべての$i$に対して、$c_i \gets f(u_i, v_i)$ となるようなベクトル$\mathbf{c} = F(\mathbf{u},\mathbf{v})$を作成することができます。

ここで、スカラー関数をelement-wiseなベクトル演算に*置き換える*ことで、ベクトル値関数$F: \mathbb{R}^d \rightarrow \mathbb{R}^d$を作成することもできます。MXNetでは、基本的な数式演算である (+,-,/,\*,\*\*) はすべて、任意のshapeに対して、shapeが同じテンソルであれば、element-wiseな演算に*置き換える*ことが可能です。同じ shapeをもつ2つのテンソルおよび行列に対して、element-wiseな演算を行うことができます。

```{.python .input}
x = nd.array([1, 2, 4, 8])
y = nd.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x / y', x / y)
```

より多くの演算をelement-wiseに適用することも可能です。例えば指数関数の場合は:

```{.python .input  n=12}
x.exp()
```

要素ごとの計算に加えて、`dot`関数を使った行列の乗算のような行列演算も実行できます。以下では、`x`と`y`の転置に対して、行列の演算を実行します。 `x`を3行4列の行列として定義し、`y`を4行3列の行列に転置します。 2つの行列の掛け算を計算することで3行3列の行列が得られます (これが何を意味するのか混乱していても心配しないでください。[線形代数](linear-algebra.md)の章で行列演算についてさらに詳しく説明します。)


```{.python .input  n=13}
x = nd.arange(12).reshape((3,4))
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
nd.dot(x, y.T)
```

複数のNDArrayを結合することもできます。そのためには、どの次元で結合するかをシステムに伝える必要があります。以下の例は、次元0 (行に沿って) と次元1 (列に沿って) に沿って2つの行列をそれぞれ結合します。

```{.python .input}
nd.concat(x, y, dim=0)
nd.concat(x, y, dim=1)
```

ときには、論理式を使って2値のNDArrayを作成したいと思うかもしれません。例えば `x == y`を取り上げましょう。ある、要素に関して`x`と`y`が等しい場合、新しく作成されるNDArrayにおいて、その要素と同じ位置には1の値が入ります。それ以外の場合は0です。

```{.python .input}
x == y
```
NDArrayにおける全要素の和を計算すると、その和だけを唯一の要素としてもつNDArrayを生成します。

```{.python .input}
x.sum()
```

`asscalar`関数を使って、結果をPythonのスカラーに変換することができます。次の例では、`x`の$\ell_2$ノルムが、単一の要素をもつNDArrayを生成し、その結果は`asscalar`によってスカラーに変換されます。

```{.python .input}
x.norm().asscalar()
```

またプログラミングの利便性から、`y.exp()`, `x.sum()`, `x.norm()`と書くこともできますし、 `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)`と書くこともできます。


## Broadcast の仕組み

上記の節では、同じshapeをもつ、2つのNDArrayに対する演算について説明しました。shapeが異なる場合は、NumPyと同様にBroadcastingが実行されます。まず、2つのNDArrayが同じ形状になるように要素を適切にコピーしてから、要素ごとに演算を実行します。


```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

`a`と`b`はそれぞれ（3x1）と（1x2）の行列なので、これらの加算を行おうと思っても、shapeが互いに一致しません。 NDArrayは、両方の行列の要素を次のようにBroadcastすることで、より大きな（3×2）行列を生成し、これに対処します。行列`a`に対しては列を複製し、行列`b`に対しては行を複製し、最後に要素ごとに加算します。

```{.python .input}
a + b
```

## Indexing と Slicing

他のPython配列と同じように、NDArrayの要素はそのインデックスによってアクセスできます。 Pythonでは伝統的に、最初の要素のインデックスは0で、範囲を最初の要素を含んで最後の要素は含まないように指定します。つまりは`1：3`で指定される範囲は、2番目と3番目の要素を選択します (インデックス1と2が選ばれ、それぞれ2番めと3番めの要素)。行列のそれぞれの行を選択して試してみましょう。


```{.python .input  n=19}
x[1:3]
```
上記で説明したように、行列の要素に値を書き込むこともできます。

```{.python .input  n=20}
x[1, 2] = 9
x
```

複数の要素に同じ値を割り当てたい場合は、それらのすべてにインデックスに対して値を割り当てれば良いです。例えば、 `[0:2,:]`は1行目と2行目にアクセスします。以下では、それらの行に対して12を割り当てます。行列のindexingについて説明しましたが、いうまでもなくベクトルや2次元以上のテンソルに対しても同様のことが機能します。


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

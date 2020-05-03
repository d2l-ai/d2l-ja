# データの取り扱い

データを格納したり操作したりできなければ、何も行うことはできません。一般的に、データを処理する上で重要なことは次の2点です。（i）データを獲得すること、および（ii）コンピュータに取り込んだらそれを処理することです。データの格納方法さえわからなければ、データを獲得することに意味がありません。合成データを利用して実際に触ってみましょう。はじめに、データ格納と変換のための MXNet の主なツールである $n$次元配列 (`ndarray`) を紹介します。MXNet では、`ndarray` はクラスであり、インスタンスを "an `ndarray`" と呼びます。

Python の科学計算パッケージに広く利用されている NumPy を以前に使用したことがあれば、この説は馴染みがあると感じるでしょう。意図的にそうなっているのです。 We designed MXNet's `ndarray` to be an extension to NumPy's `ndarray` with a few killer features.
まず、MXNet の NDArrayは、CPU、GPU、および分散クラウドアーキテクチャでの非同期計算をサポートしています。第二に、MXNet の NDArray は自動微分をサポートしています。これらの特性によって、NDArrayは深層学習に適うものとなっています。この書籍では、`ndarray` といえば、特に記述がない場合は、MXNet の NDArray を指すものとします。

## まずはじめに

In this section, we aim to get you up and running,
equipping you with the basic math and numerical computing tools
that you will build on as you progress through the book.
Do not worry if you struggle to grok some of
the mathematical concepts or library functions.
The following sections will revisit this material
in the context of practical examples and it will sink.
On the other hand, if you already have some background
and want to go deeper into the mathematical content, just skip this section.

To start, we import the `np` (`numpy`) and
`npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes functions supported by NumPy,
while the `npx` module contains a set of extensions
developed to empower deep learning within a NumPy-like environment.
When using `ndarray`, we almost always invoke the `set_np` function:
this is for compatibility of `ndarray` processing by other components of MXNet.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()
```

`ndarray` は数値の (多次元の) 配列を表します。 1軸のNDArrayは(数学的には)*vector*に対応します。2軸のNDArrayは*行列*に対応します。3つ以上の軸を持つ配列に関しては、数学者は特別な名前を与えていません - 単にそれらを*テンソル*と呼びます。

To start, we can use `arange` to create a row vector `x`
containing the first $12$ integers starting with $0$,
though they are created as floats by default.
Each of the values in an `ndarray` is called an *element* of the `ndarray`.
For instance, there are $12$ elements in the `ndarray` `x`.
Unless otherwise specified, a new `ndarray`
will be stored in main memory and designated for CPU-based computation.

```{.python .input  n=2}
x = nd.arange(12)
x
```
`

`ndarray`の shape (各軸に対する長さ) は `shape`のプロパティを利用して確認することができます。

```{.python .input  n=8}
x.shape
```

`size`のプロパティから、NDArrayインスタンスの要素の総数を得ることもできます。これは`shape`の要素の積となります。ここではベクトルを扱っているので、`size`も`shape`の一要素も同じ数になります。


```{.python .input  n=9}
x.size
```

ある`ndarray`に関して、要素数や要素の値を変えることなくshapeを変えるためには`reshape`関数を使います。例えば、shape が ($12$,)の行ベクトル`x`を、shape が ($3$, $4$) の行列に変換することができます。この新しい `ndarray` は同じ値で構成されますが、3行4列の行列となります。shapeは変わっていますが、`x`の要素は変わっていないことに注意してください。`size`は同じままです。


```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

すべての次元をそれぞれ手動で指定してreshapeする必要はありません。 (高さ, 幅) の shape をもつ行列を対象としていて、幅の値がわかったとすれば、高さの値も暗にわかるでしょう。つまり、割り算を実行すればよいのです。
たとえば、上の例では、3行の行列を取得するために、3行と4列の両方を指定しました。幸いなことに、`ndArray` はある次元数を残りの次元数から自動的に決定することができます。 `ndarray`では、自動的に推測させたい次元に `-1`を配置します。さきほどの例では `x.reshape(3, 4)`の代わりに、 `x.reshape(-1, 4)`または `x.reshape(3,-1)`を使用することが可能です。

`empty`のメソッドは、いくらかのメモリを確保して、その要素に対する値を気にしない行列を返します。これは非常に効率的ですが、各要素は非常に大きな値も含め、任意の値を取る可能性がありますので注意しましょう


```{.python .input}
nd.empty((3, 4))
```

通常は、行列を、1、ゼロ、既知の定数、または既知の分布から無作為に抽出された数値のいずれかで初期化しようとするでしょう。
すべての要素が0、shapeが(2,3,4)であるようなテンソルを表すNDArrayを作成することもできます。

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

同様に、すべての要素が1であるようなテンソルを作成するためには以下を実行します。

```{.python .input  n=5}
nd.ones((2, 3, 4))
```
場合によっては、既知の確率分布に従って、`ndarray` の各要素の値をランダムにサンプリングすることもあるでしょう。例えば、ニューラルネットワークにおいて、パラメータのための配列を作成する場合、一般にその配列の値はランダムに初期化されるでしょう。次のスニペットは、(3, 4)の形状をもつ`ndarray`を作成します。その要素は平均がゼロで分散が1のガウス分布(正規分布)から無作為にサンプリングされた値をもちます。

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

数値を含む Python のリストを与えることで、特定の値を要素にもつ `ndarray` を作成することもできます。Here, the outermost list corresponds to axis $0$, and the inner list to axis $1$.


```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```



## 演算

This book is not about software engineering.
Our interests are not limited to simply
reading and writing data from/to arrays.
配列に対して数学演算を適用したい場合があると思います。最も単純かつ便利な機能として要素ごとの (elementwise)の機能が挙げられます。These apply a standard scalar operation
to each element of an array.
For functions that take two arrays as inputs,
elementwise operations apply some standard binary operator
on each pair of corresponding elements from the two arrays. スカラーからスカラーへ写像するあらゆる関数に対して、element-wiseな関数を作成することができます。

単項のスカラー演算 (入力を1つだけとる) は、数学的な記法を用いると、 $f: \mathbb{R} \rightarrow \mathbb{R}$ で表すことができます。This just mean that the function is mapping
from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator
(taking two real inputs, and yielding one output)
by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. 同じshapeの2つのベクトル$\mathbf{u}$と$\mathbf{v}$、バイナリの演算子$f$が与えられているとき、すべての$i$に対して、$c_i \gets f(u_i, v_i)$ となるようなベクトル$\mathbf{c} = F(\mathbf{u},\mathbf{v})$を作成することができます。ここで、where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
ここで、スカラー関数をelement-wiseなベクトル演算に*置き換える*ことで、ベクトル値関数$F: \mathbb{R}^d \rightarrow \mathbb{R}^d$を作成することもできます。

MXNetでは、基本的な数式演算である (+,-,/,\*,\*\*) はすべて、任意のshapeに対して、shapeが同じテンソルであれば、element-wiseな演算に*置き換える*ことが可能です。同じ shapeをもつ2つのテンソルおよび行列に対して、element-wiseな演算を行うことができます。
In the following example, we use commas to formulate a $5$-element tuple,
where each element is the result of an elementwise operation.

```{.python .input  n=11}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

より多くの演算をelement-wiseに適用することも可能です。例えば指数の単項演算子は:

```{.python .input  n=12}
x.exp()
```


In addition to elementwise computations,
we can also perform linear algebra operations,
including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra
(with no assumed prior knowledge) in :numref:`sec_linear-algebra`.

We can also *concatenate* multiple `ndarray`s together,
stacking them end-to-end to form a larger `ndarray`.
We just need to provide a list of `ndarray`s
and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate
two matrices along rows (axis $0$, the first element of the shape)
vs. columns (axis $1$, the second element of the shape).
We can see that, the first output `ndarray`'s axis-$0$ length ($6$)
is the sum of the two input `ndarray`s' axis-$0$ lengths ($3 + 3$);
while the second output `ndarray`'s axis-$1$ length ($8$)
is the sum of the two input `ndarray`s' axis-$1$ lengths ($4 + 4$).

```{.python .input  n=14}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

ときには、*論理式*を使って2値のNDArrayを作成したいと思うかもしれません。例えば `x == y`を取り上げましょう。ある、要素に関して`x`と`y`が等しい場合、新しく作成されるNDArrayにおいて、その要素と同じ位置には1の値が入ります。それ以外の場合は0です。

```{.python .input}
x == y
```
NDArrayにおける全要素の和を計算すると、その和だけを唯一の要素としてもつNDArrayを生成します。

```{.python .input}
x.sum()
```

利便性の観点から、`x.sum()` は `np.sum(x)` と書くこともできます。

## Broadcast の仕組み

上記の節では、同じshapeをもつ、2つのNDArrayに対する演算について説明しました。shapeが異なっていたとしても、特定の条件下においては、*broadcasting* によって要素ごとの演算が実行可能です。These mechanisms work in the following way:
First, expand one or both arrays
by copying elements appropriately
so that after this transformation,
the two `ndarray`s have the same shape.
Second, carry out the elementwise operations
on the resulting arrays.

In most cases, we broadcast along an axis where an array
initially only has length $1$, such as in the following example:


```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

`a`と`b`はそれぞれ（$3 \times 1$）と（$1 \times 2$）の行列なので、これらの加算を行おうと思っても、shapeが互いに一致しません。 NDArrayは、両方の行列の要素を次のようにBroadcastすることで、より大きな（3×2）行列を生成し、これに対処します。行列`a`に対しては列を複製し、行列`b`に対しては行を複製し、最後に要素ごとに加算します。

```{.python .input}
a + b
```

## Indexing と Slicing

他のPython配列と同じように、NDArrayの要素はそのインデックスによってアクセスできます。 Python の配列と同様に、最初の要素のインデックスは0で、範囲を最初の要素から最後の要素の*手前*までを含むように指定します。As in standard Python lists, we can access elements
according to their relative position to the end of the list
by using negative indices.

Thus, `[-1]` selects the last element and `[1:3]`
selects the second and the third elements as follows:

```{.python .input  n=19}
x[-1], x[1:3]
```

上記で説明したように、行列の要素に値を書き込むこともできます。

```{.python .input  n=20}
x[1, 2] = 9
x
```

複数の要素に同じ値を割り当てたい場合は、それらのすべてにインデックスに対して値を割り当てれば良いです。例えば、 `[0:2,:]`は1行目と2行目にアクセスします。ここで、`:` は軸 $1$ (列) に関するすべての要素をとります。行列のindexingについて説明しましたが、いうまでもなくベクトルや $2$ 次元以上のテンソルに対しても同様のことが機能します。


```{.python .input  n=21}
x[0:2, :] = 12
x
```

## メモリの節約

前に紹介した例では、演算を実行するたびに、その結​​果を格納するために新しいメモリを割り当てていました。たとえば、 `y = x + y`と書くと、もともと`y`が指していた行列への参照をはずし、代わりに新しく割り当てられたメモリ上の`y`を指します。以下の例では、Pythonの`id()`関数という、メモリの参照オブジェクトの正確なアドレスを返す関数を使って実際に説明します。`y = y + x`を実行した後、`id(y)`が別の場所を指していることがわかります。これは、Pythonが最初に`y + x`を評価し、その結果に新しいメモリを割り当て、それからメモリ内のこの新しい位置を`y`が指すようにしているからです。


```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```
これが望まれない場合として、以下の2つが挙げられます。第一に、私たちは常に不必要なメモリ割り当てを行いたくありません。 機械学習では、数百メガバイトのパラメータがあり、1秒のうちにそれらすべてを複数回更新します。 通常は、これらの更新を*その場で実行(in-place)*します。 第二に、同じパラメータは複数の変数が参照しているかもしれません。 適切に更新しないと、メモリリークが発生し、誤って古いパラメータを参照する可能性があります。

幸運にも、in-placeな演算は、MXNetでは簡単に行うことができます。sliceを利用して以前に確保された配列に対して、演算の結果を割り当てることができます。つまり、y[:] = <expression>とします。この挙動を示すために、0の要素ブロックを割り当てるzero_likeを利用して、行列のshapeをコピーします。


```{.python .input  n=16}
z = y.zeros_like()
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

もし、`x`の値が以降の計算において再利用されないのであれば、その演算のオーバーヘッドを削減するために`x[:] = x + y` or `x += y` とすることも可能です。

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## 他の Python オブジェクトへの変換

MXNet `ndarray`と NumPy `ndarray`との間の変換は容易です。変換された配列はメモリを共有*しません*。 これは少し不便に感じるかもしれませんが、実は非常に重要です。CPUまたは複数GPUの1つで演算を実行する際、NumPyで何か実行する場合、同じメモリ領域でMXNetがその処理を待つということは望ましくありません。`array` や ` asnumpy` の関数はこれに対処しています。


```{.python .input  n=25}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```


To convert a size-$1$ `ndarray` to a Python scalar,
we can invoke the `item` function or Python's built-in functions.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

## Summary

* MXNet's `ndarray` is an extension to NumPy's `ndarray`
  with a few killer advantages that make it suitable for deep learning.
* MXNet's `ndarray` provides a variety of functionalities including
  basic mathematics operations, broadcasting, indexing, slicing,
  memory saving, and conversion to other Python objects.


## 練習

1. この節のコードを実行しましょう。この節の条件文 `x == y`を`x < y`または`x > y`に変更して、どのようなNDArrayを得られるか確認してください。
1. Broadcastの仕組みで要素ごとの演算を行った2つのNDArraysを別のshapeに変えてみましょう。例えば、三次元テンソルです。結果は予想と同じでしょうか?
1. 3つの行列a, b, cがあるとします。 コード`c = nd.dot（a, b.T）+ c`を、最もメモリ効率の良い方法に書き換えてください。

## [議論](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)

# 線形代数

データを格納・操作できるようになったので、ほとんどのモデルを理解するために必要となる、基本的な線形代数の一部を簡単に見てみましょう。基本的な数学的対象、算術、線形代数の演算を紹介し、それらの数学的な表記とコードで実装する方法を示します。


## スカラー

これまで線形代数や機械学習を勉強したことがなかったとしたら、おそらく一度に1つの数だけを扱ってきたかもしれません。そして、小切手帳の帳尻を合わしたり、レストランでの支払いをしたことがあれば、数のペアを足したり、掛けたりするといった基本的な方法はご存知でしょう。パロアルトの気温が華氏52度というのを例にあげましょう。正式には、これらの値を*スカラー*と呼びます。この値を摂氏に変換したい場合 (温度測定する単位としてメートル法が採用するより賢明な方法)、$f$を$52$として、式$c=(f-32)*5/9$を評価します。この式において、$32$、$5$、および$9$の各項はスカラー値です。何らかの値を代入するためのプレースホルダー$c$と$f$は変数と呼ばれ、それらは未知のスカラー値を表します。

この書籍では、スカラーを通常の小文字 ($x$、$y$、$z$) とする数学的表記を利用します。また、すべての (連続の) 実数値スカラーがとりうる空間を$\mathcal{R}$と表します。便宜上、*空間*の厳密な説明は後で行いますが、今のところ、$x \in \mathcal{R}$ という表現は $x$ が実数値スカラーであることを示す公式な方法であることを覚えておいてください。同様に、$x, y \in {0, 1}$ は $x$ と $y$ が $0$ または $1$ をとることを表しています。


MXNetでは、1つの要素だけをもつ `ndarray` を作成することでスカラーを表します。以下のスニペットでは、2つのスカラーをインスタンス化し、加算、乗算、除算、べき乗など、見慣れた算術演算を実行します。


```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```


## ベクトル

ベクトルは単にスカラー値のリストとして考えることができます。ベクトル内の各数値は、単一のスカラー値で構成されています。これらの値をベクトルの*要素*や*成分* (英語では *entries* や *components*) と呼びます。ベクトルがデータセットに含まれるデータ例を表す場合、ベクトルの値は実世界の意味をもっているといえます。たとえば、ローンの債務不履行のリスクを調査している場合、収入、雇用期間、過去の債務不履行の数などに対応する要素を持つベクトルに、各申請者を関連付けることができるでしょう。もし、病院の患者の心臓発作のリスクを調べる場合は、最新のバイタルサイン、コレステロール値、1日当たりの運動時間などからなるベクトルで、患者の状態を表すかもしれません。数学表記では、通常、太字の小文字でベクトル (例えば、$\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$)を表します。


In MXNet, we work with vectors via $1$-dimensional `ndarray`s. In general `ndarray`s can have arbitrary lengths, subject to the memory limits of your machine.


```{.python .input}
x = np.arange(4)
x
```


We can refer to any element of a vector by using a subscript. For example, we can refer to the $i^\mathrm{th}$ element of $\mathbf{x}$ by $x_i$. Note that the element $x_i$ is a scalar, so we do not bold-face the font when referring to it. Extensive literature considers column vectors to be the default orientation of vectors, so does this book. In math, a vector $\mathbf{x}$ can be written as

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

ここで $x_1, \ldots, x_n$ はベクトルの要素です。
コードでは、`ndarray` にインデックスを付けることで任意の要素$i$にアクセスします。

```{.python .input}
x[3]
```

## 長さ、次元、shape

:numref:`sec_ndarray` からいくつかの概念を見直してみましょう。ベクトルは単に数値の配列です。そして、すべての配列が長さをもつのと同じように、すべてのベクトルも長さをもっています。ベクトル$\mathbf{x}$が$n$個の実数値スカラーで構成されているとき、数学的な表記を用いて、これを$\mathbf{x} \in \mathcal{R}^n$のように表現することができます。ベクトルの長さは通常、*次元*と呼ばれます。

通常のPython配列と同様に、Pythonの組み込み関数``len()``を呼び出すことで `ndarray` の長さにアクセスできます。

```{.python .input  n=4}
len(x)
```

`ndarray` が (1軸で構成される) ベクトルを表すとき、
`.shape`属性を利用することで、ベクトルの長さにアクセスすることもできます。shapeは、`ndarray` の各軸に沿った長さ (次元) をリスト形式で表現するタプルです。1つの軸だけをもつ `ndarray` において、shapeはたった1つの要素をもちます。

```{.python .input}
x.shape
```


英語では次元をdimensionといいますが、これが様々な意味をもつがゆえに、人々を混乱させる傾向にあります。そこで、*dimensionality*という単語を使って、ベクトルまたは軸の*dimensionality*で長さ(つまりは要素数)を指すことがあります。しかし、`ndarray の`*dimensionality*は、`ndarray`がもつ軸の数を指すこともあります。この意味においては、`ndarray` の軸の *dimensionality* が軸の長さに相当するでしょう。


## 行列

ベクトルがスカラーを0次から1次に一般化したもののように、行列はベクトルを$1$次元から$2$次元に一般化したものになります。通常、大文字 の太字 (例えば、$X$、$Y$、$Z$) で表す行列は、コードのなかでは2つの軸をもつ`ndarray`として表されます。


In math notation, we use $\mathbf{A} \in \mathbb{R}^{m \times n}$
to express that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars.
Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as a table,
where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column:


$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix}$$
:eqlabel:`eq_matrix_def`


For any $\mathbf{A} \in \mathbb{R}^{m \times n}$, the shape of $\mathbf{A}$
is ($m$, $n$) or $m \times n$.
Specifically, when a matrix has the same number of rows and columns,
its shape becomes a square; thus, it is called a *square matrix*.

We can create an $m \times n$ matrix in MXNet
by specifying a shape with two components $m$ and $n$
when calling any of our favorite functions for instantiating an `ndarray`

```{.python .input}
A = np.arange(20).reshape(5,4)
A
```


行 ($i$) と列 ($j$) のインデックスを指定することで、行列$A$のスカラー要素$a_{ij}$にアクセスすることができます。 `:`を利用してインデックスを指定しなければ、それぞれの次元に沿ってすべての要素をとることができます (前の節で説明しました)。

When the scalar elements of a matrix $\mathbf{A}$, such as in :eqref:`eq_matrix_def`, are not given,
we may simply use the lower-case letter of the matrix $\mathbf{A}$ with the index subscript, $a_{ij}$,
to refer to $[\mathbf{A}]_{ij}$.
To keep notation simple, commas are inserted to separate indices only when necessary,
such as $a_{2, 3j}$ and $[\mathbf{A}]_{2i-1, 3}$.


Sometimes, we want to flip the axes.
When we exchange a matrix's rows and columns,
the result is called the *transpose* of the matrix.
Formally, we signify a matrix $\mathbf{A}$'s transpose by $\mathbf{A}^\top$
and if $\mathbf{B} = \mathbf{A}^\top$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.
Thus, the transpose of $\mathbf{A}$ in :eqref:`eq_matrix_def` is
a $n \times m$ matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

In code, we access a matrix's transpose via the `T` attribute.

```{.python .input  n=7}
A.T
```

As a special type of the square matrix,
a *symmetric matrix* $\mathbf{A}$ is equal to its transpose:
$\mathbf{A} = \mathbf{A}^\top$.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
B == B.T
```



行列は便利なデータ構造です。行列を使用することで、異なる様々なデータを1つのデータとして構成することができます。たとえば、行列の行が各異なる家 (データ点) に対応し、列が異なる属性に対応します。This should sound familiar if you have ever used spreadsheet software or
have read :numref:`sec_pandas`.
Thus, although the default orientation of a single vector is a column vector,
in a matrix that represents a tabular dataset,
it is more conventional to treat each data point as a row vector in the matrix.
And, as we will see in later chapters,
this convention will enable common deep learning practices.
For example, along the outermost axis of an `ndarray`,
we can access or enumerate minibatches of data points,
or just data points if no minibatch exists.

## テンソル

ベクトルがスカラーの一般化であるように、また、行列がベクトルの一般化であるように、さらに多くの軸をもつデータ構造を作成することができます。テンソルは、任意の数の軸を持つ`ndarray`を記述する汎用的な方法を提供しています。たとえば、ベクトルは1次テンソル、行列は2次テンソルです。Tensors are denoted with capital letters of a special font face
(e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$)
and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) is similar to that of matrices.

画像を扱い始める際には、テンソルはより重要なものとなります。なぜなら画像は、高さ、幅、カラーチャンネル (RGB) の3軸をもつ `ndarray`だからです。しかしこの章では、さらに高次のテンソルについてはスキップして、基本的な事項に注目します。

```{.python .input}
X = np.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)
```

## テンソル計算の基本的性質

スカラー、ベクトル、行列、そして任意の次数のテンソルは、頼りになる良い性質をもっています。たとえば、elementwiseな演算の定義で気付いた方もいるかもしれませんが、同じshapeの計算対象が与えられた場合、elementwiseな演算の結果は同じshapeのテンソルになります。
Similarly, given any two tensors with the same shape,
the result of any binary elementwise operation
will be a tensor of that same shape.
For example, adding two matrices of the same shape
performs elementwise addition over these two matrices.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

Specifically, elementwise multiplication of two matrices is called their *Hadamard product* (math notation $\odot$).
Consider matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ whose element of row $i$ and column $j$ is $b_{ij}$. The Hadamard product of matrices $\mathbf{A}$ (defined in :eqref:`eq_matrix_def`) and $\mathbf{B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```
Multiplying or adding a tensor by a scalar also does not change the shape of the tensor,
where each element of the operand tensor will be added or multiplied by the scalar.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## 縮約

任意のテンソルに対する演算で有用なものといえば、要素の合計を計算することでしょう。数学的表記では、合計を$\sum$記号を使って表現します。長さ$d$のベクトル$\mathbf{x}$の要素の合計を表すために$\sum_{i=1}^d x_i$と書くことができます。コード上は、`sum`の関数を呼び出すだけです。


```{.python .input  n=11}
x = np.arange(4)
x, x.sum()
```

任意のshapeをもつテンソルの要素についても総和を計算することができます。たとえば、$m \times n$の行列　$\mathbf{A}$　の要素の合計は、$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$と書くことができます。

```{.python .input  n=12}
A.shape, A.sum()
```


By default, invoking the `sum` function *reduces* a tensor along all its axes to a scalar.
We can also specify the axes along which the tensor is reduced via summation.
Take matrices as an example.
To reduce the row dimension (axis $0$) by summing up elements of all the rows,
we specify `axis=0` when invoking `sum`.
Since the input matrix reduces along axis $0$ to generate the output vector,
the dimension of axis $0$ of the input is lost in the output shape.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Specifying `axis=1` will reduce the column dimension (axis $1$) by summing up elements of all the columns.
Thus, the dimension of axis $1$ of the input is lost in the output shape.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Reducing a matrix along both rows and columns via summation
is equivalent to summing up all the elements of the matrix.

```{.python .input}
A.sum(axis=[0, 1])  # Same as A.sum()
```

関連する計算として*平均(mean)*があります。英語では*mean*以外に*average*とも呼ばれます。合計を要素の数で割ることで平均を計算します。コードでは、任意の形のテンソルに `mean` を呼び出すだけです。

```{.python .input  n=13}
A.mean(), A.sum() / A.size
```

Like `sum`, `mean` can also reduce a tensor along the specified axes.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```


### Non-Reduction Sum

However, sometimes it can be useful to keep the number of axes unchanged when invoking `sum` or `mean` by setting `keepdims=True`.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

For instance, since `sum_A` still keeps its $2$ axes after summing each row, we can divide `A` by `sum_A` with broadcasting.

```{.python .input}
A / sum_A
```

If we want to calculate the cumulative sum of elements of `A` along some axis, say `axis=0` (row by row),
we can call the `cumsum` function. This function will not reduce the input tensor along any axis.

```{.python .input}
A.cumsum(axis=0)
```

## ドット積

これまでのところ、elementwiseな演算、合計、平均のみを扱ってきました。もし、これだけしかできないのであれば、線形代数として1つの章を設けるほどではないでしょう。ここで紹介したいのが、最も基本的な演算の1つであるドット積です。 2つのベクトル$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ が与えられたとき、それらの*ドット積* $\mathbf{x}^T \mathbf{y}$ (または $\langle \mathbf{x}, \mathbf{y}  \rangle$) は、同じ位置の要素の積の和となります。すなわち、$\mathbf{x}^T \mathbf{y} = \sum_{i=1}^{d} x_i \cdot y_i$ となります。



```{.python .input  n=14}
y = np.ones(4)
x, y, np.dot(x, y)
```

2つのベクトルのドット積は、要素ごとにelementwiseな乗算を実行して、その総和をとることと等価です。

```{.python .input}
np.sum(x * y)
```

ドット積の有用性は幅広いです。たとえば、ベクトル $\mathbf{x}  \in \mathbb{R}^d$ で表現される値の集合と、重みの集合 $\mathbf{w} \in \mathbb{R}^d$ が与えられたとき、重み $\mathbf{w}$ による $\mathbf{x}$の値の重み付け和は、ドット積$\mathbf{u}^T \mathbf{w}$として表すことができます。重みが負ではなく、その総和が1になる場合$\left(\sum_{i=1}^{d} {w_i}=1 \right)$、ドット積は*加重平均*を表します。2つのベクトルがそれぞれ長さ1になるように正規化されているとき、ドット積はそれらの間のコサイン角度を表します。この節では後ほど、*長さ*が何を意味するのかを説明します。

## 行列ベクトル積

ドット積の計算方法がわかったところで、*行列ベクトル積*についても理解しましょう。行列$\mathbf{A} \in \mathbb{R}^{m \times n}$とベクトル$\mathbf{x} \in \mathbb{R}^n$ を定義して、eqref:`eq_matrix_def` と :eqref:`eq_vec_def` で見てみましょう。まず、行ベクトルの視点から行列 $\mathbf{A}$ を見てみましょう。

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

それぞれ、$\mathbf{a}^\top_{i} \in \mathbb{R}^{n}$は、行列$A$の$i$番目の行を表す行ベクトルです。
行列ベクトル積$\mathbf{A}\mathbf{x}$は、$i^\mathrm{th}$番目の要素がドット積 $\mathbf{a}^\top_i \mathbf{x}$となる長さ $m$ の列ベクトルとなります。


$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$


行列$\mathbf{A}\in \mathbb{R}^{m \times n}$による乗算は、ベクトルを$\mathbb{R}^{n}$から$\mathbb{R}^{m}$への変換として考えることができます。

これらの変換は非常に便利ですあることがわかります。たとえば回転という変換は、ある正方行列による乗算として表すことができます。以降の章で見られるように、前の層の値からニューラルネットワークの各層を計算する際、必要となるそれらの大量の計算を記述する際に、行列ベクトル積を使います。

行列ベクトル積を`ndarray`を利用してコード内で表現するには、ドット積と同じ`dot`関数を使います。行列`Aとベクトル`xを指定して `np.dot(A, x)` を呼び出すと、行列ベクトル積が実行されます。 A`の列次元 (その長さは軸$1$方向のもの) は `x`の次元 (長さ) と同じでなければならないことに注意してください。


```{.python .input  n=16}
A.shape, x.shape, np.dot(A, x)
```

## 行列同士の積

もし、ドット積と行列ベクトル積を理解できたとしたら、行列同士の積も同様に理解できるでしょう。

2つの行列$\mathbf{A} \in \mathbb{R}^{n \times k}$ と $\mathbf{B} \in \mathbb{R}^{k \times m}$を考えます。


$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

$\mathbf{a}^\top_{i} \in \mathbb{R}^k$ で、行列 $\mathbf{A}$ の $i$ 番目の行の行ベクトルを表し、 $\mathbf{b}_{j} \in \mathbb{R}^k$ で、行列 $\mathbf{B}$ の $j$ 番目の列の列ベクトルを表します。
行列積 $\mathbf{C} = \mathbf{A}\mathbf{B}$ を計算するには、行ベクトルに関して$\mathbf{A}$を考えて、列ベクトルに関して$\mathbf{B}$ を考えるのが最も簡単でしょう。

$$A=
\begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix},
\quad B=\begin{pmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{pmatrix}.
$$



ここで行列積  $\mathbf{C} \in \mathbb{R}^{n \times m}$ は、各要素$\mathbf{C}_{ij}$ はドット積 $\mathbf{a}^\top_i \mathbf{b}_j$として計算されます。

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

$\mathbf{AB}$の行列積については、単純に$m$個の行列ベクトル積を実行し、$n \times m$ の行列になるように、その結果をつなげていく操作とみなすことができます。ドット積や行列ベクトル積と同様に、`dot`関数を利用して行列積を計算することができます。
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with $5$ rows and $4$ columns,
and `B` is a matrix with $4$ rows and $3$ columns.
After multiplication, we obtain a matrix with $5$ rows and $3$ columns.


```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.


## ノルム

線形代数で最も有用な演算子であるノルムです。簡単に言えば、ベクトルのノルムは、ベクトルがどれくらい*大きい*かを示します。ここで考慮される*サイズ*は dimensionality ではなく、むしろ要素の大きさを表します。

線形代数では、ベクトルノルムはベクトルをスカラーにマップする関数 $f$ であり、扱いやすい性質を持っています。あらゆるベクトル $\mathbf{x}$ に対して成り立つ最初の性質は、ベクトルの要素をすべて定数 $\alpha$ 倍したとき、そのノルムも同じ定数の*絶対値*で倍にしたものになります。


$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

次の性質は三角不等式です。


$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

3つ目の性質は、単にノルムは非負でなければならないというものです。

$$f(\mathbf{x}) \geq 0.$$

これは、最小の*サイズ*が0であるという意味において納得できる性質です。最後の性質は、最小のノルムは全てのベクトルの要素がゼロであることと必要十分であることです。

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$


ひょっとするとノルムは距離の尺度の一種のように思えるかもしれません。
あなたが小学校で学んだユークリッド距離 (ピタゴラスの定理を思い浮かべてください) を覚えていれば、そこから非負性と三角不等式について気づくかもしれません。
実際のところ、ユークリッド距離はノルムで、具体的には$\ell_2$ノルムです。
Suppose that the elements in the $n$-dimensional vector
$\mathbf{x}$ are $x_1, \ldots, x_n$.
The $\ell_2$ *norm* of $\mathbf{x}$ is the square root of the sum of the squares of the vector elements:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

where the subscript $2$ is often omitted in $\ell_2$ norms, i.e., $\|\mathbf{x}\|$ is equivalent to $\|\mathbf{x}\|_2$. コードでは、$\ell_2$ノルムを計算するためには`linalg.norm`を呼ぶだけです。

```{.python .input  n=18}
u = np.array([3, -4])
np.linalg.norm(u)
```


In deep learning, we work more often
with the squared $\ell_2$ norm.
You will also frequently encounter the $\ell_1$ *norm*,
which is expressed as the sum of the absolute values of the vector elements:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

As compared with the $\ell_2$ norm,
it is less influenced by outliers.
To calculate the $\ell_1$ norm, we compose
the absolute value function with a sum over the elements.

```{.python .input  n=19}
np.abs(u).sum()
```

Both the $\ell_2$ norm and the $\ell_1$ norm
are special cases of the more general $\ell_p$ *norm*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Analogous to $\ell_2$ norms of vectors,
the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$
is the square root of the sum of the squares of the matrix elements:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $\ell_2$ norm of a matrix-shaped vector. Invoking `linalg.norm` will calculate the Frobenius norm of a matrix.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

## ノルムと目的関数
:label:`subsec_norms_and_objectives`

先に進みすぎることは本意ではないのですが、なぜこれらの概念が有用なのか知ってほしいと思います。深層学習では、最適化問題を解くことがよくあります。最適化問題においては、観測されたデータに割り当てられる確率を最大化したり、予測と真実の観測との間の距離を*最小化*します。つまり、類似アイテム (単語、商品、ニュース記事など) 間の距離を最小化したり、非類似アイテム間の距離を最大化するように、アイテムにベクトル表現を割り当てます。この目的関数は、おそらく深層学習のアルゴリズムの構成要素で (データと並んで) 最も重要で、多くの場合、ノルムで表されます。

## 線形代数のさらに先

In just this section,
we have taught you all the linear algebra
that you will need to understand
a remarkable chunk of modern deep learning.
There is a lot more to linear algebra
and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors,
and these decompositions can reveal
low-dimensional structure in real-world datasets.
There are entire subfields of machine learning
that focus on using matrix decompositions
and their generalizations to high-order tensors
to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics
once you have gotten your hands dirty
deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on,
we will wrap up this section here.

If you are eager to learn more about linear algebra,
you may refer to either :numref:`sec_geometry-linear-algebric-ops`
or other excellent resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.


## まとめ

* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* In the `ndarray` representation, scalars, vectors, matrices, and tensors have 0, 1, 2, and an arbitrary number of axes, respectively.
* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $\ell_1$ norm, the $\ell_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors with `ndarray` functions.


## 練習

1. Prove that the transpose of a matrix $\mathbf{A}$'s transpose is $\mathbf{A}$: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that the sum of transposes is equal to the transpose of a sum: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Why?
1. We defined the tensor `X` of shape ($2$, $3$, $4$) in this section. What is the output of `len(X)`?
1. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
1. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
1. When traveling between two points in Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor with shape ($2$, $3$, $4$). What are the shapes of the summation outputs along axis $0$, $1$, and $2$?
1. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for `ndarray`s of arbitrary shape?

## [議論のための](https://discuss.mxnet.io/t/2317)QRコード

![](../img/qr_linear-algebra.svg)

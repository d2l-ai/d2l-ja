# 線形代数

データを保存・操作できるようになったので、ほとんどのモデルを理解するために必要となる、基本的な線形代数の一部を簡単に見てみましょう。基本的な概念、それに対応する数学的表記、そしてそれらを実現する方法を1か所で紹介します。基本的な線形代数にすでに自信がある場合は、この章を読み飛ばすか、スキップしてください。

```{.python .input}
from mxnet import nd
```

## スカラー

これまで線形代数や機械学習を勉強したことがなかったとしたら、おそらく一度に1つの数だけを扱ってきたかもしれません。そして、それらを一緒に足したり、掛けたりするといった基本的な方法はご存知でしょう。パロアルトの気温が華氏52度というのを例にあげましょう。正式には、これらの値を*スカラー*と呼びます。この値を摂氏に変換したい場合 (温度測定する単位としてメートル法が採用するより賢明な方法)、$f$を$52$として、式$c=(f-32)*5/9$を評価します。この式において、$32$、$5$、および$9$の各項はスカラー値です。何らかの値を代入するためのプレースホルダー$c$と$f$は変数と呼ばれ、それらは未知のスカラー値を表します。

数学的な表記では、スカラーを通常の小文字 ($x$、$y$、$z$) で表します。また、すべてのスカラーがとりうる空間を$\mathcal{R}$と表します。便宜上、空間とは何かについて少し説明しますが、今のところ、$x$がスカラーであると言いたい場合には、単に$x \in \mathcal{R}$と記述することにします。$\in$という記号は"in (イン)"と発音でき、集合の要素であることを表します。

MXNetでは、1つの要素だけをもつNDArrayを作成することでスカラーを扱います。以下のスニペットでは、2つのスカラーをインスタンス化し、加算、乗算、除算、べき乗など、使い慣れた算術演算を実行します。


```{.python .input}
x = nd.array([3.0])
y = nd.array([2.0])

print('x + y = ', x + y)
print('x * y = ', x * y)
print('x / y = ', x / y)
print('x ** y = ', nd.power(x,y))
```

`asscalar`の関数を呼び出すことで、NDArrayをPythonのfloat型に変換することができます。通常、これは良い方法とはいえません。この変換を行っている間、結果とプロセス制御をPythonに戻すために、NDArrayは他のことを中断する必要があります。そして残念なことに、Pythonは並列処理が得意ではありません。そのため、この変換をコード全体に好き勝手に入れてしまえば、ネットワークの学習に長時間かかってしまうので、避けましょう。

```{.python .input}
x.asscalar()
```


## ベクトル

ベクトルは単に数字のリスト、例えば``[1.0,3.0,4.0,2.0]``、として考えることができます。ベクトル内の各数値は、単一のスカラー値で構成されています。これらの値をベクトルの*要素*や*成分* (英語では *entries* や *components*) と呼びます。多くの場合、実世界において意味のある値をもったベクトルに興味があると思います。たとえば、ローンの債務不履行のリスクを調査している場合、収入、雇用期間、過去の債務不履行の数などに対応する要素を持つベクトルに、各申請者を関連付けることができるでしょう。もし、病院の患者の心臓発作のリスクを調べる場合は、最新のバイタルサイン、コレステロール値、1日当たりの運動時間などからなるベクトルで、患者の状態を表すかもしれません。数学表記では、通常、太字の小文字でベクトル ($\mathbf{u}$、$\mathbf{v}$、$\mathbf{w}$)を表します。MXNetでは、任意の数の要素をもつ1D NDArrayをベクトルとして使用します。


```{.python .input}
x = nd.arange(4)
print('x = ', x)
```
添字を使用して、ベクトルの任意の要素を参照することができます。たとえば、$\mathbf{u}$の$4$番目の要素を$u_4$で参照できます。要素$u_4$はスカラーなので、参照するときにフォントを太字にしないでください。コードでは、``NDArray``にインデックスを付けることで任意の要素$i$にアクセスします。

```{.python .input}
x[3]
```

## 長さ、次元、shape

上の節からいくつかの概念を見直してみましょう。ベクトルは単に数値の配列です。そして、すべての配列が長さをもつのと同じように、すべてのベクトルも長さをもっています。ベクトル$\mathbf{x}$が$n$個の実数値スカラーで構成されているとき、数学的な表記を用いて、これを$\mathbf{x} \in \mathcal{R}^n$のように表現することができます。ベクトルの長さは通常、*次元*と呼ばれます。通常のPython配列と同様に、Pythonの組み込み関数``len()``を呼び出すことでNDArrayの長さにアクセスできます。

`.shape`属性を利用することで、ベクトルの長さにアクセスすることもできます。shapeは、NDArrayの各軸に沿った次元をリスト形式で表現するタプルです。ベクトルは1つの軸に沿ってインデックスされるため、shapeは1つの要素のみをもちます。

```{.python .input}
x.shape
```


英語では次元をdimensionといいますが、これが様々な意味をもつがゆえに、人々を混乱させる傾向にあります。そこで、*dimensionality*という単語を使って、ベクトルの*dimensionality*で長さ(つまりは要素数)を指すことがあります。しかし、 *dimensionality*という単語は、ときには配列がもつ軸の数を指すこともあります。この場合、スカラーは0次元をもっている、ベクトルは1次元をもっている、といった言い方をすることがあります。


**混乱を避けるために、ここでは*2D*arrayまたは*3D*arrayと言う場合、それぞれ2軸または3軸の配列を意味します。しかし、もしここで$n$次元のvectorと言えば、長さ$n$のベクトルを意味します。**


```{.python .input}
a = 2
x = nd.array([1,2,3])
y = nd.array([10,20,30])
print(a * x)
print(a * x + y)
```



## 行列

ベクトルがスカラーを0次から1次に一般化したもののように、行列はベクトルを$1D$から$2D$に一般化したものになります。通常、大文字 ($A$、$B$、$C$) で表す行列は、コードのなかでは2つの軸をもつ配列として表されます。視覚的には、各エントリ$a_{ij}$が$i$番目の行と$j$番目の列に属するような表として、行列を表現することができます。



$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix}$$

`ones`や`zeros`といった`ndarray`をインスタンス化するためによく用いられる関数を呼ぶ際に、2つの要素 `(n, m)` でshapeを指定すると、MXNetで$n$行$m$列の行列を作成できます。

```{.python .input}
A = nd.arange(20).reshape((5,4))
print(A)
```
行列は便利なデータ構造です。行列を使用することで、さまざまなバリエーションをもつデータを構成できます。たとえば、行列の行が各患者に対応し、列が異なる属性に対応するといったことを表現できるでしょう。

行 ($i$) と列 ($j$) のインデックスを指定することで、行列$A$のスカラー要素$a_{ij}$にアクセスすることができます。 `:`を利用してインデックスを指定しなければ、それぞれの次元に沿ってすべての要素をとることができます (前の節で説明しました)。

行列は`T`で転置することができます。つまり、$B=A^T$の場合、すべての$i$および$j$に対して、$b_{ij}=a_{ji}$が成立します。

```{.python .input}
print(A.T)
```

## テンソル

ベクトルがスカラーの一般化であるように、また、行列がベクトルの一般化であるように、実際には、さらに多くの軸をもつデータ構造を構成できます。テンソルは、任意の数の軸を持つ配列について議論するための、汎用的な方法を提供しています。たとえば、ベクトルは1次テンソル、行列は2次テンソルです。

テンソルを使用することは、画像に関する仕事を始める際に、より重要なものとなります。なぜなら、画像は縦、横、3つの(RGB)カラーチャンネルに対応する軸をもつ3Dデータ構造だからです。しかしこの章では、この部分をスキップして、知っておくべき基本的な事項をおさえていきます。

```{.python .input}
X = nd.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)
```

## テンソル計算の基本的性質

スカラー、ベクトル、行列、そして任意の次数のテンソルは、頼りになる良い性質をもっています。たとえば、element-wiseな演算の定義で気付いた方もいるかもしれませんが、同じshapeの計算対象が与えられた場合、element-wiseな演算の結果は同じshapeのテンソルになります。もう1つの便利な性質は、すべてのテンソルに対して、スカラーを掛けると同じshapeのテンソルが生成されることです。数学的には、同じshapeの2つのテンソル$X$と$Y$を考えると、$\alpha X+Y$は同じshapeになります (数値計算の専門家はこれをAXPY操作と呼びます)。

```{.python .input}
a = 2
x = nd.ones(3)
y = nd.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)
```

shapeはスカラーの加算と乗算によってのみ保存されるわけではありません。これらの演算は、あるベクトル空間へ属していることも保持します。ただし、最初のモデルを起動させる上で、それほど重要ではないため、この章の後半までこの説明を先延ばしにします。


## 総和と平均

任意のテンソルに対して可能なことのうち、次に洗練されていることといえば、要素の合計を計算することでしょう。数学的表記では、合計を$\sum$記号を使って表現します。長さ$d$のベクトル$\mathbf{u}$の要素の合計を表すために$\sum_{i=1}^d u_i$と書くことができます。コード上は、``nd.sum()``を呼び出すだけです。

```{.python .input}
print(x)
print(nd.sum(x))
```

任意のshapeをもつテンソルの要素についても、同様に総和を計算ことができます。たとえば、$m \times n$の行列$A$の要素の合計は、$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$と書くことができます。


```{.python .input}
print(A)
print(nd.sum(A))
```
関連する計算として*平均(mean)*があります。英語では*mean*以外に*average*とも呼ばれます。合計を要素の数で割ることで平均を計算します。数学的表記では、ベクトル$\mathbf{u}$の平均は$\frac{1}{d} \sum_{i=1}^{d} u_i$、行列Aの平均は$\frac{1}{n \cdot m} \sum_{i = 1}^{m} \sum_{j=1}^{n} a_{ij}$として記述できます。コードでは、任意の形のテンソルに ``nd.mean()``を呼び出すだけです。

```{.python .input}
print(nd.mean(A))
print(nd.sum(A) / A.size)
```

## ドット積

これまでのところ、element-wiseな演算、合計、平均のみを扱ってきました。もし、これだけしかできないのであれば、線形代数として1つの章を設けるほどではないでしょう。ここで紹介したいのが、最も基本的な演算の1つであるドット積です。 2つのベクトル$\mathbf{u}$と$\mathbf{v}$が与えられたとき、内積$\mathbf{u}^T \mathbf{v}$は対応する要素の積の和となります。すなわち、$\mathbf{u}^T \mathbf{v} = \sum_{i=1}^{d} u_i \cdot v_i$ となります。



```{.python .input}
x = nd.arange(4)
y = nd.ones(4)
print(x, y, nd.dot(x, y))
```

2つのベクトルのドット積 ``nd.dot(x,y)``は、要素ごとにelement-wiseな乗算を実行して、その総和をとることと等価です。

```{.python .input}
nd.sum(x * y)
```

ドット積の有用性は幅広いです。たとえば、一連の重み$\mathbf{w}$を考えると、いくつかの値${u}$の重み付け和は、内積$\mathbf{u}^T \mathbf{w}$として表すことができます。重みが負ではなく、その総和が1になる場合$\left(\sum_{i=1}^{d} {w_i}=1 \right)$、ドット積は*加重平均*を表します。 2つのベクトルがそれぞれ長さ1をもつ場合 (ノルムに関する節で*長さ*が何を意味するのかを説明します)、ドット積はそれらの間のコサイン角度を表します。

## 行列ベクトル積

ドット積の計算方法がわかったところで、行列ベクトル積についても理解しましょう。
行列$A$と列ベクトル$\mathbf{x}$を可視化するところから始めます。

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix},\quad\mathbf{x}=\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix} $$

行ベクトルに関して行列を可視化することもできます。

$$A=
\begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix},$$

それぞれ、$\mathbf{a}^T_{i} \in \mathbb{R}^{m}$は、行列$A$の$i$番目の行を表す行ベクトルです。

行列ベクトル積$\mathbf{y} = A\mathbf{x}$では、単に列ベクトル$\mathbf{y} \in \mathbb{R}^n$において、その要素$y_i$がドット積$\mathbf{a}^T_i \mathbf{x}$になります。

$$A\mathbf{x}=
\begin{pmatrix}
\mathbf{a}^T_{1}  \\
\mathbf{a}^T_{2}  \\
 \vdots  \\
\mathbf{a}^T_n \\
\end{pmatrix}
\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix}
= \begin{pmatrix}
 \mathbf{a}^T_{1} \mathbf{x}  \\
 \mathbf{a}^T_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^T_{n} \mathbf{x}\\
\end{pmatrix}
$$

したがって、行列$\A \in \mathbb{R}^{n \times m}$による乗算は、ベクトルを$\mathbb{R}^{m}$から$\mathbb{R}^{m}$へ射影する変換として考えることができます。

これらの変換は非常に便利です。たとえば回転という変換は、ある正方行列による乗算として表すことができます。以降の章で見られるように、ニューラルネットワークの各層の計算を記述する際に、行列ベクトル積を使うこともできます。

行列ベクトル積を`ndarray`を利用してコード内で表現するには、ドット積と同じ`nd.dot()`関数を使います。行列``A``とベクトル``x``を指定して``nd.dot(A, x)``を呼び出すと、MXNetは行列ベクトル積の実行を求められていることを認識します。 ``A``の列次元は ``x``の次元と同じでなければならないことに注意してください。


```{.python .input}
nd.dot(A, x)
```

## 行列同士の積

もし、ドット積と行列ベクトル積を理解できたとしたら、行列同士の積も同様に理解できるでしょう。

2つの行列$A \in \mathbb{R}^{n \times k}$ と $B \in \mathbb{R}^{k \times m}$を考えます。

$$A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{pmatrix},\quad
B=\begin{pmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{pmatrix}$$

行列積 $C = AB$ を計算するには、行ベクトルに関して$A$を考えて、列ベクトルに関して$B$を考えます。

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

ここで、各行ベクトル$\mathbf{a}^T_{i}$は$\mathbb{R}^k$に属していて、各列ベクトル$\mathbf{b}_j$は$\mathbb{R}^k$に属していることに注意しましょう。

行列積$C \in \mathbb{R}^{n \times m}$を計算するためには、各要素$c_{ij}$をドット積$\mathbf{a}^T_i \mathbf{b}_j$として計算するだけです。

$$C = AB = \begin{pmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{pmatrix}
\begin{pmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{pmatrix}
= \begin{pmatrix}
\mathbf{a}^T_{1} \mathbf{b}_1 & \mathbf{a}^T_{1}\mathbf{b}_2& \cdots & \mathbf{a}^T_{1} \mathbf{b}_m \\
 \mathbf{a}^T_{2}\mathbf{b}_1 & \mathbf{a}^T_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^T_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^T_{n} \mathbf{b}_1 & \mathbf{a}^T_{n}\mathbf{b}_2& \cdots& \mathbf{a}^T_{n} \mathbf{b}_m
\end{pmatrix}
$$

$AB$の行列積については、単純に$m$個の行列ベクトル積を実行し、$n \times m$ の行列になるように、その結果をつなげていく操作とみなすことができます。MXNetでは、ドット積や行列ベクトル積と同様に、行列積についても``nd.dot()``を利用して計算できます。


```{.python .input}
B = nd.ones(shape=(4, 3))
nd.dot(A, B)
```

## Norms

Before we can start implementing models,
there is one last concept we are going to introduce.
Some of the most useful operators in linear algebra are norms.
Informally, they tell us how big a vector or matrix is.
We represent norms with the notation $\|\cdot\|$.
The $\cdot$ in this expression is just a placeholder.
For example, we would represent the norm of a vector $\mathbf{x}$
or matrix $A$ as $\|\mathbf{x}\|$ or $\|A\|$, respectively.

All norms must satisfy a handful of properties:

1. $\|\alpha A\| = |\alpha| \|A\|$
1. $\|A + B\| \leq \|A\| + \|B\|$
1. $\|A\| \geq 0$
1. If $\forall {i,j}, a_{ij} = 0$, then $\|A\|=0$

To put it in words, the first rule says
that if we scale all the components of a matrix or vector
by a constant factor $\alpha$,
its norm also scales by the *absolute value*
of the same constant factor.
The second rule is the familiar triangle inequality.
The third rule simply says that the norm must be non-negative.
That makes sense, in most contexts the smallest *size* for anything is 0.
The final rule basically says that the smallest norm is achieved by a matrix or vector consisting of all zeros.
It is possible to define a norm that gives zero norm to nonzero matrices,
but you cannot give nonzero norm to zero matrices.
That may seem like a mouthful, but if you digest it then you probably have grepped the important concepts here.

If you remember Euclidean distances (think Pythagoras' theorem) from grade school,
then non-negativity and the triangle inequality might ring a bell.
You might notice that norms sound a lot like measures of distance.

In fact, the Euclidean distance $\sqrt{x_1^2 + \cdots + x_n^2}$ is a norm.
Specifically it is the $\ell_2$-norm.
An analogous computation,
performed over the entries of a matrix, e.g. $\sqrt{\sum_{i,j} a_{ij}^2}$,
is called the Frobenius norm.
More often, in machine learning we work with the squared $\ell_2$ norm (notated $\ell_2^2$).
We also commonly work with the $\ell_1$ norm.
The $\ell_1$ norm is simply the sum of the absolute values.
It has the convenient property of placing less emphasis on outliers.

To calculate the $\ell_2$ norm, we can just call ``nd.norm()``.

```{.python .input}
nd.norm(x)
```

To calculate the L1-norm we can simply perform the absolute value and then sum over the elements.

```{.python .input}
nd.sum(nd.abs(x))
```

## Norms and objectives

While we do not want to get too far ahead of ourselves, we do want you to anticipate why these concepts are useful.
In machine learning we are often trying to solve optimization problems: *Maximize* the probability assigned to observed data. *Minimize* the distance between predictions and the ground-truth observations. Assign vector representations to items (like words, products, or news articles) such that the distance between similar items is minimized, and the distance between dissimilar items is maximized. Oftentimes, these objectives, perhaps the most important component of a machine learning algorithm (besides the data itself), are expressed as norms.


## Intermediate linear algebra

If you have made it this far, and understand everything that we have covered,
then honestly, you *are* ready to begin modeling.
If you are feeling antsy, this is a perfectly reasonable place to move on.
You already know nearly all of the linear algebra required
to implement a number of many practically useful models
and you can always circle back when you want to learn more.

But there is a lot more to linear algebra, even as concerns machine learning.
At some point, if you plan to make a career in machine learning,
you will need to know more than what we have covered so far.
In the rest of this chapter, we introduce some useful, more advanced concepts.



### Basic vector properties

Vectors are useful beyond being data structures to carry numbers.
In addition to reading and writing values to the components of a vector,
and performing some useful mathematical operations,
we can analyze vectors in some interesting ways.

One important concept is the notion of a vector space.
Here are the conditions that make a vector space:

* **Additive axioms** (we assume that x,y,z are all vectors):
  $x+y = y+x$ and $(x+y)+z = x+(y+z)$ and $0+x = x+0 = x$ and $(-x) + x = x + (-x) = 0$.
* **Multiplicative axioms** (we assume that x is a vector and a, b are scalars):
  $0 \cdot x = 0$ and $1 \cdot x = x$ and $(a b) x = a (b x)$.
* **Distributive axioms** (we assume that x and y are vectors and a, b are scalars):
  $a(x+y) = ax + ay$ and $(a+b)x = ax +bx$.

### Special matrices

There are a number of special matrices that we will use throughout this tutorial. Let's look at them in a bit of detail:

* **Symmetric Matrix** These are matrices where the entries below and above the diagonal are the same. In other words, we have that $M^\top = M$. An example of such matrices are those that describe pairwise distances, i.e. $M_{ij} = \|x_i - x_j\|$. Likewise, the Facebook friendship graph can be written as a symmetric matrix where $M_{ij} = 1$ if $i$ and $j$ are friends and $M_{ij} = 0$ if they are not. Note that the *Twitter* graph is asymmetric - $M_{ij} = 1$, i.e. $i$ following $j$ does not imply that $M_{ji} = 1$, i.e. $j$ following $i$.
* **Antisymmetric Matrix** These matrices satisfy $M^\top = -M$. Note that any square matrix can always be decomposed into a symmetric and into an antisymmetric matrix by using $M = \frac{1}{2}(M + M^\top) + \frac{1}{2}(M - M^\top)$.
* **Diagonally Dominant Matrix** These are matrices where the off-diagonal elements are small relative to the main diagonal elements. In particular we have that $M_{ii} \geq \sum_{j \neq i} M_{ij}$ and $M_{ii} \geq \sum_{j \neq i} M_{ji}$. If a matrix has this property, we can often approximate $M$ by its diagonal. This is often expressed as $\mathrm{diag}(M)$.
* **Positive Definite Matrix** These are matrices that have the nice property where $x^\top M x > 0$ whenever $x \neq 0$. Intuitively, they are a generalization of the squared norm of a vector $\|x\|^2 = x^\top x$. It is easy to check that whenever $M = A^\top A$, this holds since there $x^\top M x = x^\top A^\top A x = \|A x\|^2$. There is a somewhat more profound theorem which states that all positive definite matrices can be written in this form.


## Summary

In just a few pages (or one Jupyter notebook) we have taught you all the linear algebra you will need to understand a good chunk of neural networks. Of course there is a *lot* more to linear algebra. And a lot of that math *is* useful for machine learning. For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets. There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems. But this book focuses on deep learning. And we believe you will be much more inclined to learn more mathematics once you have gotten your hands dirty deploying useful machine learning models on real datasets. So while we reserve the right to introduce more math much later on, we will wrap up this chapter here.

If you are eager to learn more about linear algebra, here are some of our favorite resources on the topic

* For a solid primer on basics, check out Gilbert Strang's book [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)
* Zico Kolter's [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2317)

![](../img/qr_linear-algebra.svg)

# 線形代数
:label:`sec_linear-algebra`

データの保存と操作ができるようになったところで、本書で説明するほとんどのモデルを理解して実装するために必要な、基本的な線形代数のサブセットについて簡単に説明します。以下では、線形代数における基本的な数学オブジェクト、算術、演算を紹介します。それぞれを数学的表記法と対応するコード実装で表現します。 

## スカラー

線形代数や機械学習を学んだことがないなら、数学に関する過去の経験は、おそらく一度に一つの数字について考えることだったでしょう。また、小切手帳のバランスを取ったり、レストランで夕食代を支払ったりしたことがある場合は、数字のペアの加算や乗算などの基本的なことを行う方法をすでに知っています。たとえば、パロアルトの気温は華氏$52$度です。正式には、1 つの数値量だけで構成される値を「スカラー」と呼びます。この値を摂氏 (メートル法のより適切な温度スケール) に変換する場合は、$c = \frac{5}{9}(f - 32)$ という式を評価し、$f$ を $52$ に設定します。この方程式では、$5$、$9$、$32$ の各項はスカラー値です。プレースホルダー $c$ と $f$ は*変数* と呼ばれ、不明なスカラー値を表します。 

本書では、スカラー変数を通常の小文字で表す数学表記法を採用しています (例:$x$、$y$、$z$)。すべての (連続) *実数値* スカラーの空間を $\mathbb{R}$ で表します。便宜上、*space* が正確に何であるかを厳密に定義しますが、$x \in \mathbb{R}$ という式は $x$ が実数値のスカラーであると言う正式な言い方であることを覚えておいてください。記号$\in$は「in」と発音でき、単に集合のメンバーであることを示します。同様に、$x$ と $y$ は値が $0$ または $1$ にしかならない数値であることを示すために $x, y \in \{0, 1\}$ と書くことができます。 

(**スカラーは、要素が 1 つだけのテンソルで表されます。**) 次のスニペットでは、2 つのスカラーをインスタンス化し、加算、乗算、除算、べき乗という使い慣れた算術演算を行います。

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## ベクター

[**ベクトルは単にスカラー値のリストと考えることができます**] これらの値をベクトルの*要素* (*entries* または*components*) と呼びます。ベクトルがデータセットの例を表す場合、その値には実世界での意味があります。たとえば、ローン債務不履行のリスクを予測するモデルをトレーニングする場合、各申請者を、収入、雇用期間、以前の債務不履行回数、その他の要因に対応する要素をもつベクトルに関連付けることができます。入院患者が直面する可能性のある心臓発作のリスクを研究している場合、各患者を最新のバイタルサイン、コレステロール値、1日あたりの運動時間などを捉えたベクターで表すことができます。数学表記では、通常、ベクトルを太字の小文字で表します。英字 ($\mathbf{x}$、$\mathbf{y}$、$\mathbf{z})$ など) 

一次元テンソルを介してベクトルを扱います。一般に、テンソルはマシンのメモリ制限に応じて任意の長さを持つことができます。

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

添字を使うと、ベクトルのどの要素でも参照できます。たとえば、$x_i$ によって $\mathbf{x}$ の $i^\mathrm{th}$ エレメントを参照できます。要素 $x_i$ はスカラーなので、参照するときにフォントを太字にしないことに注意してください。広範な文献では、列ベクトルがベクトルの既定の方向であると見なされています。この本も同様です。数学では、ベクトル $\mathbf{x}$ は次のように記述できます。 

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

$x_1, \ldots, x_n$ はベクトルの要素です。コードでは、(**テンソルにインデックスを付けて任意の要素にアクセスする**)

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### 長さ、次元、形状

:numref:`sec_ndarray` のいくつかの概念をもう一度見てみましょう。ベクトルは単なる数値の配列です。そして、すべての配列が長さを持つように、すべてのベクトルもそうです。数学表記法では、ベクトル $\mathbf{x}$ が $n$ の実数値のスカラーで構成されているとすると、$\mathbf{x} \in \mathbb{R}^n$ と表現できます。ベクトルの長さは、一般にベクトルの*次元* と呼ばれます。 

通常の Python 配列と同様に、Python に組み込まれている `len()` 関数を呼び出すことで [**テンソルの長さにアクセスできます**]。

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

テンソルが (正確に 1 つの軸をもつ) ベクトルを表す場合、`.shape` 属性を介してその長さにアクセスすることもできます。形状は、テンソルの各軸に沿った長さ (次元) を列挙したタプルです。(**軸が 1 つだけのテンソルの場合、形状には要素が 1 つしかありません。**)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

これらの文脈では「次元」という言葉が過負荷になりがちで、人々を混乱させる傾向があることに注意してください。明確にするために、*vector* または*axis* の次元を使用して、その長さ、つまりベクトルまたは軸の要素数を参照します。ただし、テンソルの次元性は、テンソルが持つ軸の数を参照するために使用します。この意味で、テンソルのある軸の次元は、その軸の長さになります。 

## 行列

ベクトルがスカラーを 0 次から 1 次まで一般化するように、行列はベクトルを次数 1 から次 2 に一般化します。通常、太字の大文字で表される行列 ($\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$ など) は、2 つの軸をもつテンソルとしてコードで表されます。 

数学表記法では $\mathbf{A} \in \mathbb{R}^{m \times n}$ を使用して、行列 $\mathbf{A}$ が $m$ 行と $n$ 列の実数スカラーで構成されることを表します。任意の行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ をテーブルとして説明できます。各要素 $a_{ij}$ は $i^{\mathrm{th}}$ 行と $j^{\mathrm{th}}$ 列に属します。 

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

$\mathbf{A} \in \mathbb{R}^{m \times n}$ の場合、$\mathbf{A}$ の形状は ($m$、$n$) または $m \times n$ になります。具体的には、行列の行数と列数が同じ場合、その形状は正方形になるため、「*正方行列*」と呼ばれます。 

テンソルをインスタンス化するためにお気に入りの関数を呼び出すときに $m$ と $n$ の 2 つの成分をもつ形状を指定することで [**$m \times n$ 行列を作成**] できます。

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

:eqref:`eq_matrix_def` の行列 $\mathbf{A}$ のスカラー要素 $a_{ij}$ にアクセスするには、$[\mathbf{A}]_{ij}$ のように行 ($i$) と列 ($j$) のインデックスを指定します。行列 $\mathbf{A}$ のスカラー要素 (:eqref:`eq_matrix_def` など) が指定されない場合、行列 $\mathbf{A}$ の小文字をインデックス添字 $a_{ij}$ とともに使用して $[\mathbf{A}]_{ij}$ を参照することができます。表記を単純にするために、$a_{2, 3j}$ や $[\mathbf{A}]_{2i-1, 3}$ のように、必要な場合にのみカンマを別々のインデックスに挿入します。 

時々、軸を反転させたいことがあります。行列の行と列を交換すると、その結果は行列の*転置*と呼ばれます。正式には、行列 $\mathbf{A}$ の $\mathbf{A}^\top$ による転置を意味し、$\mathbf{B} = \mathbf{A}^\top$ の場合は $i$ と $j$ に対して $b_{ij} = a_{ji}$ を転置することを表します。したがって、:eqref:`eq_matrix_def` における $\mathbf{A}$ の転置は $n \times m$ 行列になります。 

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

ここで、コード内で a (**行列の転置**) にアクセスします。

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

正方行列の特殊な型として、[**a*対称行列* $\mathbf{A}$ はその転置と等しい:$\mathbf{A} = \mathbf{A}^\top$.**] ここでは対称行列 `B` を定義します。

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

ここで `B` をその転置と比較します。

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

行列は有用なデータ構造です。行列を使用すると、さまざまな変動様式を持つデータを整理できます。たとえば、マトリックスの行は異なる住宅 (データ例) に対応し、列は異なる属性に対応することがあります。スプレッドシートソフトウェアを使用したことがある人や :numref:`sec_pandas` を読んだことがある人なら、これはおなじみのように思えます。したがって、単一のベクトルの既定の方向は列ベクトルですが、表形式のデータセットを表す行列では、各データ例を行列の行ベクトルとして扱うのがより一般的です。また、後の章で説明するように、この規則により、一般的なディープラーニングの実践が可能になります。たとえば、テンソルの最も外側の軸に沿って、データ例のミニバッチ、またはミニバッチが存在しない場合はデータ例のみにアクセスまたは列挙できます。 

## テンソル

ベクトルがスカラーを一般化し、行列がベクトルを一般化するように、さらに多くの軸をもつデータ構造を構築できます。[**Tensors**](本項の「テンソル」は代数的オブジェクトを指す) (**$n$ 次元の配列を任意の軸数で記述する一般的な方法を挙げてください。**) ベクトルは一次テンソル、行列は二次テンソルです。テンソルは特殊なフォントフェース ($\mathsf{X}$、$\mathsf{Y}$、$\mathsf{Z}$ など) の大文字で表され、インデックスの仕組み ($x_{ijk}$ や $[\mathsf{X}]_{1, 2i-1, 3}$ など) は行列のものと似ています。 

テンソルは、高さ、幅、およびカラーチャンネル (赤、緑、青) を積み重ねるための*channel* 軸に対応する 3 つの軸を持つ $n$ 次元の配列として到着するイメージで作業を開始するとより重要になります。ここでは、高次のテンソルをスキップして、基本に焦点を当てます。

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## テンソル演算の基本的性質

任意の数の軸のスカラー、ベクトル、行列、テンソル (この項の「テンソル」は代数的オブジェクトを指します) には、便利な便利なプロパティがいくつかあります。たとえば、要素単位の単項演算の定義から、要素単位の単項演算ではオペランドの形状が変化しないことに気付いたかもしれません。同様に、[**同じ形状のテンソルが2つあれば、要素ごとの2進演算の結果は同じ形状のテンソルになります。**] たとえば、同じ形状の2つの行列を加算すると、これら 2 つの行列に対して要素単位の加算が行われます。

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

具体的には、[**2つの行列の要素ごとの乗算を*アダマール積***](数学表記 $\odot$) と呼びます。行 $i$ と列 $j$ の要素が $b_{ij}$ である行列 $\mathbf{B} \in \mathbb{R}^{m \times n}$ について考えてみます。行列 $\mathbf{A}$ (:eqref:`eq_matrix_def` で定義されている) と $\mathbf{B}$ のアダマール積 

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

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**テンソルにスカラーを乗算または加算する**] もテンソルの形状は変化せず、オペランドテンソルの各要素にスカラーが加算または乗算されます。

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## 減少
:label:`subseq_lin-alg-reduction`

任意のテンソルで実行できる便利な操作の 1 つは、[**要素の和**] を計算することです。数学的表記法では、$\sum$ 記号を使用して和を表現します。要素の和を長さ $d$ のベクトル $\mathbf{x}$ で表すために、$\sum_{i=1}^d x_i$ と書きます。コードでは、合計を計算する関数を呼び出すだけです。

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

たとえば、$m \times n$ 行列 $\mathbf{A}$ の要素の和は $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ と書くことができます。

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

デフォルトでは、合計を計算する関数を呼び出します。
*テンソルをそのすべての軸に沿ってスカラーに縮小* します。
また、[**加算によってテンソルを減少させる軸を指定することもできます**] 行列を例にとります。すべての行の要素を合計して行の次元 (軸 0) を減らすには、関数を呼び出すときに `axis=0` を指定します。入力行列は軸 0 に沿って縮小されて出力ベクトルが生成されるため、入力の軸 0 の次元は出力シェイプでは失われます。

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

`axis=1` を指定すると、すべての列の要素が合計され、列の次元 (軸 1) が縮小されます。したがって、入力の軸 1 の次元は出力形状では失われます。

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

加算によって行と列の両方に沿って行列を削減することは、行列のすべての要素を合計することと等価です。

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

[**関連する量は*平均*で、*平均*とも呼ばれます。**] 合計を要素の総数で割ることで平均を計算します。コードでは、任意の形状のテンソルの平均を計算する関数を呼び出すだけで済みます。

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同様に、平均を計算する関数では、指定した軸に沿ってテンソルを減らすこともできます。

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### 非リダクション合計
:label:`subseq_lin-alg-non-reduction`

ただし、和または平均を計算する関数を呼び出す場合、[**軸の数を変更しない**] と便利な場合があります。

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

たとえば、`sum_A` は各行を合計した後も 2 つの軸を保持しているので、ブロードキャストで `A` を `sum_A` で割ることができます。

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

[**軸に沿った `A` の要素の累積和**]、たとえば `axis=0` (行ごと) を計算したい場合は、`cumsum` 関数を呼び出すことができます。この関数は入力テンソルを軸に沿って減少させません。

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## ドットプロダクト

これまでは、要素単位の演算、合計、平均のみを実行してきました。そして、これが私たちにできることのすべてであるならば、線形代数はおそらくそれ自身のセクションに値しないでしょう。ただし、最も基本的な演算の 1 つは内積です。2 つのベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ が与えられた場合、それらの*内積* $\mathbf{x}^\top \mathbf{y}$ (または $\langle \mathbf{x}, \mathbf{y}  \rangle$) は、同じ位置 $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$ にある要素の積の和になります。 

[~~2つのベクトルの*内積* は、同じ位置にある要素の積の和です~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

注意 (**要素ごとの乗算と和を実行することで、2つのベクトルの内積を等価的に表現できます:**)

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

内積は幅広い状況で役に立ちます。たとえば、ベクトル $\mathbf{x}  \in \mathbb{R}^d$ で表される値のセットと $\mathbf{w} \in \mathbb{R}^d$ で表される重みのセットがある場合、$\mathbf{x}$ の値の重み $\mathbf{w}$ に従った加重和は、ドット積 $\mathbf{x}^\top \mathbf{w}$ として表すことができます。重みが負でなく、合計が 1 の場合 ($\left(\sum_{i=1}^{d} {w_i} = 1\right)$)、内積は*加重平均*を表します。2 つのベクトルを正規化して単位長をもつと、内積はそれらの間の角度の余弦を表します。この*length* の概念については、このセクションの後半で正式に紹介します。 

## マトリックス-ベクトル積

ドット積の計算方法がわかったところで、*行列-ベクトル積*について理解できるようになります。:eqref:`eq_matrix_def` と :eqref:`eq_vec_def` でそれぞれ定義され可視化された行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ とベクトル $\mathbf{x} \in \mathbb{R}^n$ を思い出してください。まず、行列 $\mathbf{A}$ を行ベクトルで可視化することから始めましょう。 

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

ここで、各 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ は、行列 $\mathbf{A}$ の $i^\mathrm{th}$ 行を表す行ベクトルです。 

[**行列-ベクトル積 $\mathbf{A}\mathbf{x}$ は長さが $m$ の列ベクトルで、$i^\mathrm{th}$ の要素はドット積 $\mathbf{a}^\top_i \mathbf{x}$: **] 

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

行列 $\mathbf{A}\in \mathbb{R}^{m \times n}$ による乗算は、ベクトルを $\mathbb{R}^{n}$ から $\mathbb{R}^{m}$ に投影する変換と考えることができます。これらの変換は非常に有用であることが分かります。たとえば、回転を正方行列による乗算として表すことができます。以降の章で説明するように、行列とベクトルの積を使用して、前の層の値からニューラルネットワークの各層を計算するときに必要な最も集中的な計算を記述することもできます。

:begin_tab:`mxnet`
行列とベクトルの積をテンソルでコードで表現する場合、ドット積と同じ関数 `dot` を使用します。行列 `A` とベクトル `x` をもって `np.dot(A, x)` を呼び出すと、行列とベクトルの積が実行されます。`A` の列の次元 (軸 1 に沿った長さ) は `x` の次元 (長さ) と同じでなければならないことに注意してください。
:end_tab:

:begin_tab:`pytorch`
行列とベクトルの積をテンソルを使ったコードで表現するには、`mv` 関数を使用します。行列 `A` とベクトル `x` をもって `torch.mv(A, x)` を呼び出すと、行列とベクトルの積が実行されます。`A` の列の次元 (軸 1 に沿った長さ) は `x` の次元 (長さ) と同じでなければならないことに注意してください。
:end_tab:

:begin_tab:`tensorflow`
行列とベクトルの積をテンソルを使ったコードで表現するには、`matvec` 関数を使用します。行列 `A` とベクトル `x` をもって `tf.linalg.matvec(A, x)` を呼び出すと、行列とベクトルの積が実行されます。`A` の列の次元 (軸 1 に沿った長さ) は `x` の次元 (長さ) と同じでなければならないことに注意してください。
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## マトリックス-マトリックス乗算

ドット積と行列ベクトル積のコツをつかんだなら、*matrix-matrix乗算*は簡単なはずです。 

$\mathbf{A} \in \mathbb{R}^{n \times k}$ と $\mathbf{B} \in \mathbb{R}^{k \times m}$ の 2 つの行列があるとします。 

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

行列 $\mathbf{A}$ の $i^\mathrm{th}$ 行を表す行ベクトルを $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ で表し、$\mathbf{b}_{j} \in \mathbb{R}^k$ を行列 $\mathbf{B}$ の $j^\mathrm{th}$ 列の列ベクトルとします。行列積 $\mathbf{C} = \mathbf{A}\mathbf{B}$ を生成するには、$\mathbf{A}$ を行ベクトルから、$\mathbf{B}$ を列ベクトルから考えるのが最も簡単です。 

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

次に、各要素 $c_{ij}$ をドット積 $\mathbf{a}^\top_i \mathbf{b}_j$ として計算するだけで、行列積 $\mathbf{C} \in \mathbb{R}^{n \times m}$ が生成されます。 

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

[**行列と行列の乗算 $\mathbf{AB}$ は、単に $m$ の行列とベクトルの積を実行し、その結果をつなぎ合わせて $n \times m$ 行列を形成すると考えることができます。**] 次のスニペットでは、`A` と `B` で行列の乗算を実行しています。ここで、`A` は 5 行 4 列の行列で、`B` は 4 行 3 列の行列です。乗算後、5行3列の行列が得られます。

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

行列-行列の乗算は単に「*行列乗算*」と呼ぶことができ、アダマール積と混同しないでください。 

## 規範
:label:`subsec_lin-algebra-norms`

線形代数で最も有用な演算子には、*norms* があります。非公式には、ベクトルのノルムによってベクトルがどれだけ*大きい*かがわかります。ここで検討中の*size* の概念は、次元性ではなく、成分の大きさに関係します。 

線形代数では、ベクトルノルムは、ベクトルをスカラーにマッピングし、いくつかの特性を満たす関数 $f$ です。任意のベクトル $\mathbf{x}$ が与えられた場合、1 番目の特性は、ベクトルのすべての要素を定数係数 $\alpha$ でスケーリングすると、そのノルムも同じ定数因子の*絶対値* でスケーリングされることを示しています。 

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

2 つ目の特性は、おなじみの三角不等式です。 

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

3番目の特性は、ノルムが非負でなければならないことを単純に示しています。 

$$f(\mathbf{x}) \geq 0.$$

ほとんどのコンテキストでは、何かの最小の*size*は0なので、これは理にかなっています。最終的な特性では、最小ノルムが達成され、すべてゼロで構成されるベクトルによってのみ達成されることが要求されます。 

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

ノルムは距離の尺度によく似ていることに気付くかもしれません。小学校からのユークリッド距離（ピタゴラスの定理を考えて）を覚えていれば、非否定性と三角不等式の概念が鐘を鳴らすかもしれません。実際、ユークリッド距離はノルムです。具体的には $L_2$ ノルムです。$n$ 次元ベクトル $\mathbf{x}$ の要素が $x_1, \ldots, x_n$ であると仮定します。 

[**$\mathbf{x}$ の $L_2$ *ノルム* は、ベクトル要素の二乗和の平方根です:**] 

(** $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$ドル**) 

$L_2$ の基準では、添字の $2$ が省略されることがよくあります。つまり $\|\mathbf{x}\|$ は $\|\mathbf{x}\|_2$ に相当します。コードでは、ベクトルの $L_2$ ノルムを次のように計算できます。

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

ディープラーニングでは、$L_2$ の二乗ノルムを使用する頻度が高くなります。 

また、[**the $L_1$ *norm***] もよく見かけますが、これはベクトル要素の絶対値の和で表されます。 

(** $\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$ドル**) 

$L_2$ ノルムと比較すると、外れ値の影響は小さくなります。$L_1$ ノルムを計算するために、要素の合計をもつ絶対値関数を作成します。

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

$L_2$ ノルムと $L_1$ ノルムはどちらも、より一般的な $L_p$ *ノルム* の特殊なケースです。 

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

ベクトルのノルム $L_2$ と同様に、[***フロベニウスノルム* の行列 $\mathbf{X} \in \mathbb{R}^{m \times n}$**] は、行列要素の二乗和の平方根です。 

[** $\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$ドル**] 

Frobenius ノルムは、ベクトルノルムのすべての特性を満たします。行列型のベクトルの $L_2$ ノルムであるかのように動作します。次の関数を呼び出すと、行列のフロベニウスノルムが計算されます。

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### 規範と目標
:label:`subsec_norms_and_objectives`

私たちは自分より先に進みたくはありませんが、なぜこれらの概念が役に立つのかについて、すでにいくつかの直感を植え付けることができます。ディープラーニングでは、最適化問題を解こうとすることがよくあります。
*観測データに割り当てられる確率を最大化*
*予測間の距離を最小化*
そしてグラウンドトゥルースの観測。類似するアイテム間の距離が最小化され、異なるアイテム間の距離が最大になるように、アイテム (単語、製品、ニュース記事など) にベクトル表現を割り当てます。多くの場合、(データ以外に) ディープラーニングアルゴリズムの最も重要なコンポーネントである目的は標準として表現されます。 

## 線形代数の詳細

このセクションでは、現代のディープラーニングの驚くべき部分を理解するために必要なすべての線形代数について説明しました。線形代数には他にも多くの機能があり、その数学の多くは機械学習に役立ちます。たとえば、行列を因子に分解することができ、この分解によって実世界のデータセットでは低次元の構造が明らかになります。機械学習には、行列分解とその高次テンソルへの一般化を使用してデータセット内の構造を発見し、予測問題を解決することに重点を置いたサブフィールドがあります。しかし、この本はディープラーニングに焦点を当てています。また、実際のデータセットに有用な機械学習モデルを導入して手を汚すと、より多くの数学を学ぶ傾向が強まると私たちは信じています。したがって、後で数学をさらに紹介する権利を留保しますが、このセクションをここでまとめます。 

線形代数についてもっと知りたければ、[online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) またはその他の優れたリソース :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008` を参照してください。 

## [概要

* スカラー、ベクトル、行列、テンソルは、線形代数の基本的な数学オブジェクトです。
* ベクトルはスカラーを一般化し、行列はベクトルを一般化します。
* スカラー、ベクトル、行列、テンソルにはそれぞれ 0、1、2、任意の数の軸があります。
* 指定した軸に沿って `sum` と `mean` だけテンソルを減らすことができます。
* 2 つの行列の要素ごとの乗算は、アダマール積と呼ばれます。行列の乗算とは異なります。
* ディープラーニングでは、$L_1$ ノルム、$L_2$ ノルム、フロベニウスノルムなどのノルムを扱うことがよくあります。
* スカラー、ベクトル、行列、テンソルに対してさまざまな演算を実行できます。

## 演習

1. 行列 $\mathbf{A}$ の転置の転置が $\mathbf{A}$:$(\mathbf{A}^\top)^\top = \mathbf{A}$ であることを証明します。
1. 2 つの行列 $\mathbf{A}$ と $\mathbf{B}$ が与えられた場合、転置の和が和の転置に等しいことを示します。$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$
1. 正方行列 $\mathbf{A}$ がある場合、$\mathbf{A} + \mathbf{A}^\top$ は常に対称ですか?なぜ？
1. この節では、形状 (2, 3, 4) のテンソル `X` を定義しました。`len(X)`の出力は何ですか？
1. 任意の形状のテンソル `X` に対して、`len(X)` は常に `X` の特定の軸の長さに対応しますか？その軸は何ですか？
1. `A / A.sum(axis=1)` を実行して、何が起こるか見てみましょう。その理由を分析できますか？
1. マンハッタンの2地点間を移動する場合、座標、つまり道路と道路の観点からカバーする必要がある距離はどれくらいですか？斜めに旅行できますか。
1. 形状 (2, 3, 4) のテンソルを考えてみます。軸 0、1、2 に沿った加算出力の形状を教えてください。
1. 3 軸以上のテンソルを関数 `linalg.norm` に送り、その出力を観測します。この関数は任意の形状のテンソルに対して何を計算しますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

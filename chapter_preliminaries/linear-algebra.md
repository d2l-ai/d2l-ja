```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 線形代数
:label:`sec_linear-algebra`

今では、データセットをテンソルに読み込み、これらのテンソルを基本的な数学演算で操作できるようになりました。洗練されたモデルの構築を始めるには、線形代数のツールもいくつか必要になります。このセクションでは、スカラー算術から始まり、行列乗算に至るまで、最も重要な概念を穏やかに紹介します。 

## スカラー

ほとんどの日常的な数学は、一度に1つずつ数字を操作することで構成されています。正式には、これらの値を*スカラー*と呼びます。たとえば、パロアルトの気温は華氏$72$度です。温度を摂氏に変換する場合は、$f$を$72$に設定して、$c = \frac{5}{9}(f - 32)$という式を評価します。この方程式では、$5$、$9$、および$32$という値はスカラーです。変数 $c$ と $f$ は不明なスカラーを表します。 

スカラーは、通常の小文字の文字（例：$x$、$y$、$z$）とすべてのスペース（連続）で表します 
*$\mathbb{R}$ による実数値* スカラー。
便宜上、*スペース*の厳密な定義はスキップします。$x \in \mathbb{R}$という式は、$x$が実数値のスカラーであることを表す正式な言い方であることを覚えておいてください。記号$\in$（「in」と発音）は、セットのメンバーシップを示します。たとえば、$x, y \in \{0, 1\}$ は、$x$ と $y$ が値 $0$ または $1$ のみを取ることができる変数であることを示します。 

(**スカラーは、1つの要素のみを含むテンソルとして実装されます。**) 以下では、2つのスカラーを割り当て、おなじみの加算、乗算、除算、べき乗演算を実行します。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## ベクター

私たちの目的のために、[**ベクトルはスカラーの固定長配列と考えることができます。**] 対応するコードと同様に、これらの値をベクトルの*要素*と呼びます（同義語には*エントリ*と*コンポーネント*が含まれます）。ベクトルが実世界のデータセットの例を表す場合、その値は現実世界での意味を持ちます。たとえば、ローンの債務不履行のリスクを予測するモデルをトレーニングする場合、各申請者を、収入、雇用期間、以前のデフォルトの数などの数量に対応する要素を持つベクターに関連付けることができます。心臓発作のリスクを研究していた場合、各ベクターは患者を表し、その成分は最新のバイタルサイン、コレステロール値、1日の運動時間などに対応している可能性があります。ベクターを太字の小文字で表します（例：$\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$）。 

ベクトルは $1^{\mathrm{st}}$ 次テンソルとして実装されます。一般に、このようなテンソルは、メモリの制限に応じて、任意の長さを持つことができます。注意：Pythonでは、ほとんどのプログラミング言語と同様に、ベクトルインデックスは$0$から始まり、*ゼロベースのインデックス*とも呼ばれますが、線形代数では添字は$1$（1ベースのインデックス）から始まります。

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

添字を使用してベクトルの要素を参照できます。たとえば、$x_2$ は $\mathbf{x}$ の 2 番目の要素を示します。$x_2$ はスカラーなので、太字にはしません。既定では、要素を垂直に積み重ねることでベクトルを視覚化します。 

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

ここで $x_1, \ldots, x_n$ はベクトルの要素です。後で、そのような*列ベクトル*と、要素が水平に積み重なっている*行ベクトル*を区別します。[**インデックスを使ってテンソルの要素にアクセスします。**]

```{.python .input}
%%tab mxnet
x[2]
```

```{.python .input}
%%tab pytorch
x[2]
```

```{.python .input}
%%tab tensorflow
x[2]
```

ベクトルに $n$ 個の要素が含まれていることを示すために、$\mathbf{x} \in \mathbb{R}^n$ と記述します。正式には、$n$をベクトルの*次元*と呼びます。[**コードでは、これはテンソルの長さに対応します**]、Pythonの組み込み`len`関数を介してアクセスできます。

```{.python .input}
%%tab mxnet
len(x)
```

```{.python .input}
%%tab pytorch
len(x)
```

```{.python .input}
%%tab tensorflow
len(x)
```

`shape` 属性を使用して長さにアクセスすることもできます。形状は、各軸に沿ったテンソルの長さを示すタプルです。(**軸が1つだけのテンソルには、1つの要素しかない形状があります。**)

```{.python .input}
%%tab mxnet
x.shape
```

```{.python .input}
%%tab pytorch
x.shape
```

```{.python .input}
%%tab tensorflow
x.shape
```

多くの場合、「ディメンション」という言葉は、軸の数と特定の軸に沿った長さの両方を意味するようにオーバーロードされます。この混乱を避けるために、*順序*は軸の数を表し、*次元*はコンポーネントの数だけを参照するために使用します。 

## マトリックス

スカラーが $0^{\mathrm{th}}$ 次テンソルで、ベクトルが $1^{\mathrm{st}}$ 次テンソルであるように、行列は $2^{\mathrm{nd}}$ 次テンソルです。行列を太字の大文字 (例:$\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$) で表し、コードでは 2 つの軸をもつテンソルで表します。式 $\mathbf{A} \in \mathbb{R}^{m \times n}$ は、行列 $\mathbf{A}$ に $m \times n$ の実数値のスカラーが含まれ、$m$ 行と $n$ 列として配置されていることを示します。$m = n$のとき、行列は*二乗*だと言います。視覚的には、任意のマトリックスを表として説明できます。個々の要素を参照するには、行インデックスと列インデックスの両方に添字を付けます。たとえば、$a_{ij}$ は $\mathbf{A}$ の $i^{\mathrm{th}}$ 行と $j^{\mathrm{th}}$ 列に属する値です。 

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

コードでは、$2^{\mathrm{nd}}$ オーダーのテンソルで行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ を形状 ($m$、$n$) で表します。[**任意の適切なサイズの $m \times n$ テンソルを $m \times n$ 行列に変換できます**] 希望の形状を `reshape` に渡します。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

時々、軸を反転させたいことがあります。行列の行と列を交換すると、その結果は*転置*と呼ばれます。正式には、$\mathbf{A}$の転置を$\mathbf{A}^\top$で表し、$\mathbf{B} = \mathbf{A}^\top$の場合は、$i$と$j$のすべてに対して$b_{ij} = a_{ji}$を転置することを表します。したがって、$m \times n$ 行列の転置は $n \times m$ 行列になります。 

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

コードでは、以下のように任意の (**行列の転置**) にアクセスできます。

```{.python .input}
%%tab mxnet
A.T
```

```{.python .input}
%%tab pytorch
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**対称行列は、それ自体の転置と等しい正方行列のサブセットです:$\mathbf{A} = \mathbf{A}^\top$.**] 次の行列は対称です:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

マトリックスはデータセットを表すのに便利です。通常、行は個々のレコードに対応し、列は個別の属性に対応します。 

## テンソル

機械学習はスカラー、ベクトル、行列だけで遠くまで進むことができますが、最終的には高次 [**テンソル**] で作業する必要があるかもしれません。テンソル (**$n^{\mathrm{th}}$次配列の拡張を記述する一般的な方法を教えてください。**) *テンソルクラス*のソフトウェアオブジェクトは、これらも任意の数の軸を持つことができるため、正確に「テンソル」と呼びます。単語を使うのは混乱するかもしれませんが
*テンソル* 両方の数学的オブジェクト
そしてコードでのその実現、私たちの意味は通常文脈から明らかであるべきです。一般的なテンソルは、特殊なフォントフェース（例：$\mathsf{X}$、$\mathsf{Y}$、$\mathsf{Z}$）を持つ大文字で表し、それらのインデックスメカニズム（例：$x_{ijk}$と$[\mathsf{X}]_{1, 2i-1, 3}$）は行列のそれと自然に従います。 

テンソルは、画像を扱い始めるとより重要になります。各イメージは、高さ、幅、および*チャネル* に対応する軸を持つ $3^{\mathrm{rd}}$ 次テンソルとして届きます。各空間位置で、各色 (赤、緑、青) の強度がチャネルに沿って積み重ねられます。さらに、画像の集合は、$4^{\mathrm{th}}$次テンソルによってコードで表され、異なる画像が第1軸に沿って索引付けされる。高次テンソルは、形状成分の数を増やすことによって、ベクトルや行列と同様に構築されます。

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

## テンソル算術の基本的性質

スカラー、ベクトル、行列、高次テンソルにはすべて便利なプロパティがあります。たとえば、要素単位の演算では、オペランドと同じ形状の出力が生成されます。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

[**2つの行列の要素ごとの積は、それらの*アダマール積***]（$\odot$と表記）と呼ばれます。以下に、2つの行列$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$のアダマール積のエントリを綴ります。 

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
%%tab mxnet
A * B
```

```{.python .input}
%%tab pytorch
A * B
```

```{.python .input}
%%tab tensorflow
A * B
```

[**スカラーとテンソルの加算または乗算**] は、元のテンソルと同じ形状の結果を生成します。ここでは、テンソルの各要素がスカラーに加算 (または乗算) されます。

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## 削減
:label:`subsec_lin-alg-reduction`

しばしば、[**テンソルの要素の合計**] を計算したいとします。長さ$n$のベクトル$\mathbf{x}$の要素の合計を表現するには、$\sum_{i=1}^n x_i$と記述します。それには簡単な機能があります:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

[**任意の形状のテンソルの要素の合計**] を表現するには、単純にそのすべての軸を合計します。たとえば、$m \times n$ 行列 $\mathbf{A}$ の要素の合計は、$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ と記述できます。

```{.python .input}
%%tab mxnet
A.shape, A.sum()
```

```{.python .input}
%%tab pytorch
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

デフォルトでは、sum 関数を呼び出します
*すべての軸に沿ってテンソルを減らす*、
最終的にスカラーを生成します。私たちのライブラリでは、[**テンソルを減少させる軸を指定する。**] 行に沿ったすべての要素 (軸0) を合計するには、`sum`に`axis=0`を指定します。入力行列は、出力ベクトルを生成するために軸 0 に沿って減少するため、この軸は出力の形状から欠落しています。

```{.python .input}
%%tab mxnet
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab pytorch
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1` を指定すると、すべての列の要素が合計され、列の次元 (軸 1) が小さくなります。

```{.python .input}
%%tab mxnet
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab pytorch
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

合計によって行と列の両方に沿って行列を削減することは、行列のすべての要素を合計することと同じです。

```{.python .input}
%%tab mxnet
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
%%tab pytorch
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A) # Same as `tf.reduce_sum(A)`
```

[**関連する数量は*平均*で、*平均*とも呼ばれます。**] 合計を要素の総数で割ることによって平均を計算します。平均値の計算は非常に一般的であるため、`sum`と同様に機能する専用のライブラリ関数を取得します。

```{.python .input}
%%tab mxnet
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同様に、平均を計算する関数も特定の軸に沿ってテンソルを減らすことができます。

```{.python .input}
%%tab mxnet
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## 非還元合計
:label:`subsec_lin-alg-non-reduction`

合計または平均を計算する関数を呼び出すときに [**軸の数を変更しない**] と便利な場合があります。これは、ブロードキャストメカニズムを使用する場合に重要です。

```{.python .input}
%%tab mxnet
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

たとえば、`sum_A`は各行の合計後に2つの軸を保持するため、（**`A`をブロードキャストで`sum_A`で割る**）、各行の合計が$1$になる行列を作成できます。

```{.python .input}
%%tab mxnet
A / sum_A
```

```{.python .input}
%%tab pytorch
A / sum_A
```

```{.python .input}
%%tab tensorflow
A / sum_A
```

[**いくつかの軸に沿った`A`の要素の累積合計**]、たとえば`axis=0`（行ごと）を計算する場合、`cumsum`関数を呼び出すことができます。設計上、この関数はどの軸にも沿って入力テンソルを減少させません。

```{.python .input}
%%tab mxnet
A.cumsum(axis=0)
```

```{.python .input}
%%tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## ドットプロダクツ

これまでは、要素単位の演算、合計、平均のみを実行してきました。そして、これが私たちにできるすべてだったら、線形代数は独自のセクションに値しないでしょう。幸いなことに、これは物事がより面白くなるところです。最も基本的な操作の 1 つは内積です。2つのベクトル$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$が与えられた場合、それらの*ドット積* $\mathbf{x}^\top \mathbf{y}$（または$\langle \mathbf{x}, \mathbf{y}  \rangle$）は、同じ位置にある要素の積の合計です：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。 

[~~2つのベクトルの*内積* は、同じ位置にある要素の積の合計です~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

同等に、(**要素単位の乗算とそれに続く合計を実行することにより、2つのベクトルの内積を計算できます:**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

ドット積は幅広い状況で役立ちます。たとえば、ベクトル $\mathbf{x}  \in \mathbb{R}^n$ で示されるいくつかの値のセットと $\mathbf{w} \in \mathbb{R}^n$ で示される重みのセットがある場合、重み $\mathbf{w}$ に従った $\mathbf{x}$ の値の加重合計は、内積 $\mathbf{x}^\top \mathbf{w}$ として表すことができます。重みが負でなく、合計が1になる場合、つまり$\left(\sum_{i=1}^{n} {w_i} = 1\right)$の場合、内積は*加重平均*を表します。2 つのベクトルを単位長に正規化した後、内積はそれらの間の角度の余弦を表します。このセクションの後半で、この*length*の概念を正式に紹介します。 

## マトリックス-ベクトル製品

ドット積の計算方法がわかったので、$m \times n$ 行列 $\mathbf{A}$ と $n$ 次元ベクトル $\mathbf{x}$ の間の*積* を理解し始めることができます。まず、行列を行ベクトルで視覚化します。 

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

ここで、各 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ は、行列 $\mathbf{A}$ の $i^\mathrm{th}$ 行を表す行ベクトルです。 

[**行列ベクトル積 $\mathbf{A}\mathbf{x}$ は長さ$m$の単純な列ベクトルで、その$i^\mathrm{th}$要素はドット積 $\mathbf{a}^\top_i \mathbf{x}$: **] 

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

行列 $\mathbf{A}\in \mathbb{R}^{m \times n}$ を使用した乗算は、$\mathbb{R}^{n}$ から $\mathbb{R}^{m}$ へのベクトルを投影する変換と考えることができます。これらの変換は非常に便利です。たとえば、回転を特定の正方行列による乗算として表すことができます。マトリックスベクトル積は、前の層からの出力を前提として、ニューラルネットワークの各層の出力を計算する際に必要な主要な計算も記述します。

:begin_tab:`mxnet`
行列とベクトルの積をコードで表すには、同じ `dot` 関数を使用します。操作は、引数の型に基づいて推測されます。`A` (軸 1 に沿った長さ) の列の次元は、`x` (長さ) の次元と同じでなければならないことに注意してください。
:end_tab:

:begin_tab:`pytorch`
行列とベクトルの積をコードで表すには、`mv` 関数を使用します。`A` (軸 1 に沿った長さ) の列の次元は、`x` (長さ) の次元と同じでなければならないことに注意してください。PyTorch には、(引数に応じて) 行列ベクトルと行列行列積の両方を実行できる便利な演算子 `@` があります。こうして私達は `A @x `を書ける。
:end_tab:

:begin_tab:`tensorflow`
行列とベクトルの積をコードで表すには、`matvec` 関数を使用します。`A` (軸 1 に沿った長さ) の列の次元は、`x` (長さ) の次元と同じでなければならないことに注意してください。
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## マトリックス-マトリックス乗算

ドット積と行列ベクトル積のコツをつかんだら、*行列-行列の乗算* は簡単なはずです。 

$\mathbf{A} \in \mathbb{R}^{n \times k}$と$\mathbf{B} \in \mathbb{R}^{k \times m}$の2つの行列があるとします。 

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

$\mathbf{a}^\top_{i} \in \mathbb{R}^k$ が行列 $\mathbf{A}$ の $i^\mathrm{th}$ 行を表す行ベクトルを表し、$\mathbf{b}_{j} \in \mathbb{R}^k$ が行列 $\mathbf{B}$ の $j^\mathrm{th}$ 列からの列ベクトルを表すとします。 

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

行列積 $\mathbf{C} \in \mathbb{R}^{n \times m}$ を形成するには、各要素 $c_{ij}$ を、$\mathbf{A}$ の $i^{\mathrm{th}}$ 行と $\mathbf{B}$ の $j^{\mathrm{th}}$ 行、つまり $\mathbf{a}^\top_i \mathbf{b}_j$ の間の内積として計算します。 

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

[**行列と行列の乗算$\mathbf{AB}$は、$m$の行列ベクトル積または$m \times n$のドット積を実行し、結果をステッチして$n \times m$行列を形成すると考えることができます。**] 次のスニペットでは、`A`と`B`で行列の乗算を実行します。ここで、`A`は2行3列の行列で、`B`は3行4列の行列です。乗算後、2行4列の行列が得られます。

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

*行列-行列乗算* という用語は、多くの場合、*行列乗算* に簡略化されており、アダマール積と混同しないでください。 

## 規範
:label:`subsec_lin-algebra-norms`

線形代数で最も有用な演算子のいくつかは、*ノルム*です。非公式に、ベクトルのノルムは、それがどれほど*大きい*かを教えてくれます。たとえば、$\\ell_2$ ノルムは、ベクトルの (ユークリッド) 長さを測定します。ここでは、ベクトルの成分（次元性ではない）の大きさに関係する*サイズ*の概念を採用しています。  

ノルムは、ベクトルをスカラーにマッピングし、次の 3 つのプロパティを満たす関数 $\| \cdot \|$ です。 

1. 任意のベクトル $\mathbf{x}$ が与えられた場合、ベクトル (のすべての要素) をスカラー $\alpha \in \mathbb{R}$ でスケーリングすると、そのノルムはそれに応じてスケーリングされます:$$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. 任意のベクトル $\mathbf{x}$ と $\mathbf{y}$: ノルムは三角形の不等式を満たす:$$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. ベクトルのノルムは非負で、ベクトルがゼロの場合にのみ消滅します:$$\|\mathbf{x}\| > 0 \text{ for all } \mathbf{x} \neq 0.$$

多くの関数は有効な規範であり、異なる規範は異なるサイズの概念をエンコードします。直角三角形の斜辺を計算するときに小学校の幾何学で学んだユークリッドノルムは、ベクトルの要素の平方和の平方根です。正式には、これは [**$\ell_2$ *norm***] と呼ばれ、次のように表されます。 

(**$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**) 

`norm`という方法は、$\ell_2$ノルムを計算します。

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

[**$\ell_1$ norm**] も人気があり、関連するメトリックはマンハッタン距離と呼ばれます。定義上、$\ell_1$ノルムは、ベクトルの要素の絶対値を合計します。 

(**$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**) 

$\ell_2$ ノルムと比較して、外れ値に対する感度は低くなります。$\ell_1$ ノルムを計算するために、絶対値を合計演算で構成します。

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

$\ell_2$と$\ell_1$の規範はどちらも、より一般的な$\ell_p$*規範*の特殊なケースです。 

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

行列の場合、問題はもっと複雑です。結局のところ、行列は個々のエントリの集まりとしても見ることができます 
*と* は、ベクトルを操作して他のベクトルに変換するオブジェクトです。 
たとえば、行列とベクトルの積 $\mathbf{X} \mathbf{v}$ が $\mathbf{v}$ に比べてどれくらい長くなるかを尋ねることができます。この考え方は、*スペクトル*ノルムと呼ばれるノルムにつながります。ここでは、[**計算がはるかに簡単なフロベニウスノルム**] を紹介し、行列の要素の二乗和の平方根として定義されます。 

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**] 

フロベニウスノルムは、行列型ベクトルの $\ell_2$ ノルムであるかのように動作します。次の関数を呼び出すと、行列のフロベニウスノルムが計算されます。

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

私たちは自分たちより先を行き過ぎたくありませんが、これらの概念がなぜ役立つのかについて、すでに直感を植え付けることができます。ディープラーニングでは、最適化問題を解こうとすることがよくあります。
*観測データに割り当てられた確率を最大化*。
*レコメンダーモデルに関連する収益を最大化* 
*予測間の距離を最小化*
そしてグラウンドトゥルースの観察。 
*表現間の距離を最小化* 
異なる人物の写真の表現間の距離を*最大化*しながら、同じ人物の写真の。ディープラーニングアルゴリズムの目的を構成するこれらの距離は、しばしば規範として表現されます。  

## ディスカッション

このセクションでは、最新のディープラーニングの注目すべき部分を理解するために必要なすべての線形代数について説明しました。線形代数には他にもたくさんあり、その多くは機械学習に役立ちます。たとえば、行列は因子に分解でき、これらの分解によって実世界のデータセットの低次元構造を明らかにすることができます。データセットの構造を発見し、予測問題を解決するために、行列分解とその高次テンソルへの汎化を使用することに焦点を当てた機械学習のすべてのサブフィールドがあります。しかし、この本はディープラーニングに焦点を当てています。そして、実際のデータセットに機械学習を適用して手を汚したら、もっと数学を学ぶ傾向が強くなると私たちは信じています。したがって、後でさらに数学を導入する権利を留保しますが、このセクションをここでまとめます。 

もっと線形代数を学びたいと思っているなら、たくさんの優れた本やオンラインリソースがあります。より高度なクラッシュコースについては、:cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`をチェックすることを検討してください。 

要点をまとめると: 

* スカラー、ベクトル、行列、テンソルは、線形代数で使用される基本的な数学オブジェクトであり、それぞれ 0、1、2、および任意の数の軸を持ちます。
* テンソルは、インデックス付け、または `sum` や `mean` などの操作によって、指定された軸に沿ってスライスまたは削減できます。
* Elementwise 製品はアダマール製品と呼ばれます。対照的に、ドット積、行列-ベクトル積、および行列-行列積は要素単位の演算ではなく、一般にオペランドとは異なる形状を持つオブジェクトを返します。 
* アダマール積と比較して、行列-行列積は計算にかなり時間がかかります（二次時間よりも立方時間）。
* ノルムは、ベクトルの大きさのさまざまな概念を捉え、一般的に2つのベクトルの距離を測定するために2つのベクトルの差に適用されます。
 * 一般的なベクトルノルムには $\ell_1$ と $\ell_2$ ノルムが含まれ、一般的な行列ノルムには*スペクトル* ノルムと*フロベニウス* ノルムが含まれます。

## 演習

1. 行列の転置の転置が行列そのものであることを証明する:$(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. $\mathbf{A}$ と $\mathbf{B}$ の 2 つの行列が与えられると、和と転置が通勤することを示します:$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. $\mathbf{A}$の正方行列があれば、$\mathbf{A} + \mathbf{A}^\top$は常に対称ですか？前の2つの練習の結果だけを使って結果を証明できますか？
1. このセクションでは、形状 (2、3、4) のテンソル `X` を定義しました。`len(X)`の出力はどれくらいですか？コードを実装せずに回答を記述し、コードを使用して回答を確認します。 
1. 任意の形状のテンソル`X`の場合、`len(X)`は`X`の特定の軸の長さに常に対応しますか？その軸は何ですか？
1. `A / A.sum(axis=1)` を実行して、何が起こるかを確認します。その理由を分析できますか？
1. マンハッタンのダウンタウンの2地点間を移動する場合、座標、つまり大通りや通りの観点からカバーする必要がある距離はどれくらいですか？斜めに旅行できますか？
1. 形状 (2、3、4) のテンソルを考えてみましょう。軸0、1、および2に沿った合計出力の形状は何ですか？
1. 3 軸以上のテンソルを `linalg.norm` 関数に送り、その出力を観察します。この関数は任意の形状のテンソルについて何を計算しますか?
1. たとえば、ガウス確率変数で初期化された$\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$、$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$、$\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$など、3つの大きな行列を定義します。製品 $\mathbf{A} \mathbf{B} \mathbf{C}$ を計算するとします。$(\mathbf{A} \mathbf{B}) \mathbf{C}$と$\mathbf{A} (\mathbf{B} \mathbf{C})$のどちらを計算するかに応じて、メモリフットプリントと速度に違いはありますか。なぜ？
1. $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$、$\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$、$\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$ など、3 つの大きな行列を定義します。$\mathbf{A} \mathbf{B}$と$\mathbf{A} \mathbf{C}^\top$のどちらを計算するかによって、速度に違いはありますか？なぜ？メモリをクローンせずに$\mathbf{C} = \mathbf{B}^\top$を初期化すると何が変わりますか？なぜ？
1. たとえば $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$ という 3 つの行列を定義します。$[\mathbf{A}, \mathbf{B}, \mathbf{C}]$を積み重ねて3軸のテンソルを構成します。次元性って何ですか？3番目の軸の2番目の座標をスライスして、$\mathbf{B}$を回復します。あなたの答えが正しいか確認してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

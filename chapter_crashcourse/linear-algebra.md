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

したがって、行列$A \in \mathbb{R}^{n \times m}$による乗算は、ベクトルを$\mathbb{R}^{m}$から$\mathbb{R}^{m}$へ射影する変換として考えることができます。

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

## ノルム

モデルの実装を始める前に、最後に紹介しておきたい概念が1つあります。線形代数で最も有用な演算子であるノルムです。簡単に言えば、ノルムはベクトルや行列がどれくらい大きいかを表すものです。ノルムを$\|\cdot \|$という表記で表します。この式の$\cdot$は単なるプレースホルダーで、たとえばベクトル$\mathbf{x}$または行列$A$のノルムを、それぞれ$\|\mathbf{x}\|$または$\|A\|$として表します。

すべてのノルムは、以下の特性を満たします。

1. $\|\alpha A\| = |\alpha| \|A\|$
1. $\|A + B\| \leq \|A\| + \|B\|$
1. $\|A\| \geq 0$
1. If $\forall {i,j}, a_{ij} = 0$, then $\|A\|=0$

言葉で説明すると、最初の特性は、行列またはベクトルのすべての要素に対して定数$\alpha$をかけると、そのノルムも同じ定数の*絶対値*をかけたものと一致することを示しています。 2番目の特性はおなじみの三角不等式です。 3番目の特性は、ノルムは非負でなければならないことを表しています。すべてにおいて最小の*サイズ*というのは0であるという意味において、もっともといえることです。最後の特性は、基本的に最小のノルムは、すべてゼロからなる行列またはベクトルによってのみ得られることを表します。ゼロでない行列に対してゼロとなるノルムを定義することは可能ですが、ゼロの行列に対してゼロでないノルムを定義することはできません。長ったらしい説明に思えるかもしれませんが、もしその意味を把握できたとすれば、おそらく重要な概念を掴んだかもしれません。

あなたが小学校で学んだユークリッド距離 (ピタゴラスの定理を思い浮かべてください) を覚えていれば、そこから非負性と三角不等式について気づくかもしれません。また、ノルムは距離の尺度に非常に似ているように思えるでしょう。

実際のところ、ユークリッド距離$\sqrt{x_1^2 + \cdots + x_n^2}$はノルムです。 具体的には$\ell_2$ノルムです。行列の要素に対して実行される同様の計算、つまり $\sqrt{\sum_{i,j} a_{ij}^2}$はフロベニウスノルムと呼ばれます。多くの場合、機械学習では、二乗の$\ell_2$ノルム ($\ell_2^2$と表記) を使用します。また、通常$\ell_1$ノルムを利用することもあります。 $\ell_1$ノルムは単に絶対値の合計です。外れ値をあまり強調しないという便利な性質があります。

$\ell_2$ノルムを計算するためには``nd.norm()``を呼ぶだけです。

```{.python .input}
nd.norm(x)
```
L1ノルムを計算したい場合は、絶対値をとって、それらの要素の総和を計算します。

```{.python .input}
nd.sum(nd.abs(x))
```

## ノルムと目的関数

ここで少し先の話になってしまいますが、なぜこれらの概念が役に立つのかをあなたに予想してもらおうと思います。機械学習では最適化問題を解くことがよくあります。最適化問題においては、*観測されたデータに割り当てられる確率を最大化したり、予測と真実の観測との間の距離を*最小化*します。つまり、アイテム (単語、商品、ニュース記事など) をベクトルで表現し、類似アイテム間の距離を最小化したり、非類似アイテム間の距離を最大化します。これらの目的は、おそらく機械学習アルゴリズムの最も重要な構成要素（データに加えて）であり、ノルムとして表現されます。


## 線形代数: 中級

ここまでの内容をすべて理解していれば、正直なところモデリングを始める準備ができています。もし不安を感じていれば、ここから、さらに学習を進めるべきでしょう。多くの実用的なモデルを実装するために必要な線形代数について、それらのほぼすべてを知っているので、さらに学習したいときにはいつでもここに戻ることができます。

しかし、機械学習に関係があるといっても、線形代数が含む内容にはさらに多くのものがあります。あるとき、機械学習でキャリアを積もうと考えたなら、これまでに扱ってきた以上のことを知る必要があります。この章の残りの部分では、便利で高度な概念をいくつか紹介します。


### 基本的なベクトルの性質

ベクトルには、数値を受け渡すためのデータ構造以上に有用な点があります。ベクトルの要素に対する値の読み取りと書き込みや、有用な数学演算の実行に加えて、いくつかの興味深い特性をみることができます。

1つの重要な概念はベクトル空間の概念です。ベクトル空間を構成する条件は次のとおりです。

* **交換法則** (x, y, z はすべてベクトルとする):
  $x+y = y+x$, $(x+y)+z = x+(y+z)$, $0+x = x+0 = x$, $(-x) + x = x + (-x) = 0$.
* **スカラー倍** (x はベクトルで a と b はスカラーとする):
  $0 \cdot x = 0$, $1 \cdot x = x$, $(a b) x = a (b x)$.
* **分配法則** (x と y はベクトルで a と b はスカラーとする):
  $a(x+y) = ax + ay$,  $(a+b)x = ax +bx$.

### 特殊な行列

このチュートリアルでは、いくつかの特殊な行列を紹介します。詳細を少し覗いてみましょう。

* **対称行列** 行列の対角線の下と上の要素が同じ行列です。つまり、$M^\top=M$になります。そのような行列の例としては、ペアとなるデータ間の距離を記述するもの、つまり$M_{ij} = \| x_i-x_j \|$を満たすような行列です。同様に、Facebook上のつながりを表すグラフは、対称行列として記述できます。つまり、$i$と$j$がつながっている場合は$M_{ij} = 1$、そうでない場合は$M_{ij} = 0$となるような行列として表現されます。 *Twitter*のグラフは非対称です - $M_{ij} = 1$ つまり$i$が$j$をフォローしているからといって、$M_{ji} = 1$つまり$j$が$i$をフォローしているとは限りません。
* **交代行列** $M^\top = -M$を満たすような行列です。どのような正方行列も対称行列と交代行列に分解することが可能で、次の式を満たします。$M = \frac{1}{2}(M + M^\top) + \frac{1}{2}(M - M^\top)$
* **対角優位行列** 対角項以外の要素が対角項の要素よりも小さい行列で、数学表記を用いると $M_{ii} \geq \sum_{j \neq i} M_{ij}$ ならびに $M_{ii} \geq \sum_{j \neq i} M_{ji}$ を満たす行列と表現できます。ある行列がこの特性を満たすなら、その行列 $M$ は対角成分$\mathrm{diag}(M)$によって近似することができるでしょう。
* **正定値行列** 非負の$x$に対して$x^\top M x > 0$という性質を満たす行列です。直感的には、ベクトルの二乗ノルム$\|x\|^2 = x^\top x$の一般化であるといえます。$M = A^\top A$を満たすとき、$x^\top M x = x^\top A^\top A x = \|A x\|^2$が成立することを調べるのは容易です。実は正定値行列はここで書く内容より、もう少し深い内容があります。

## まとめ

In just a few pages (or one Jupyter notebook) we have taught you all the linear algebra you will need to understand a good chunk of neural networks. Of course there is a *lot* more to linear algebra. And a lot of that math *is* useful for machine learning. For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets. There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems. But this book focuses on deep learning. And we believe you will be much more inclined to learn more mathematics once you have gotten your hands dirty deploying useful machine learning models on real datasets. So while we reserve the right to introduce more math much later on, we will wrap up this chapter here.

If you are eager to learn more about linear algebra, here are some of our favorite resources on the topic

* For a solid primer on basics, check out Gilbert Strang's book [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)
* Zico Kolter's [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2317)

![](../img/qr_linear-algebra.svg)

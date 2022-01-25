# 凸性
:label:`sec_convexity`

凸性は、最適化アルゴリズムの設計において重要な役割を果たします。これは主に、このような状況でアルゴリズムの分析とテストがはるかに簡単であるという事実によるものです。言い換えれば、凸面の設定でもアルゴリズムのパフォーマンスが悪い場合、通常は良い結果が得られるとは思わないはずです。さらに、深層学習における最適化問題は一般に凸ではないが、局所的最小値に近い凸問題の性質を示すことが多い。これにより、:cite:`Izmailov.Podoprikhin.Garipov.ea.2018` のような新しい最適化バリアントが生まれます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## 定義

凸解析の前に、*凸集合* と*凸関数* を定義する必要があります。それらは機械学習に一般的に適用される数学的ツールにつながります。 

### 凸面集合

セットは凸の基礎です。簡単に言えば、$a, b \in \mathcal{X}$ で $a$ と $b$ を結ぶ線分が $\mathcal{X}$ にもあれば、ベクトル空間の集合 $\mathcal{X}$ は*凸* になります。数学的に言えば、これはすべての$\lambda \in [0, 1]$に対して 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

これは少し抽象的に聞こえる。:numref:`fig_pacman` を考えてみましょう。最初のセットには含まれていない線分が存在するため、最初のセットは凸状ではありません。他の2つのセットにはそのような問題はありません。 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

定義自体は、何かできない限り、特に有用ではありません。この場合、:numref:`fig_convex_intersect` に示すように交点を見ることができます。$\mathcal{X}$ と $\mathcal{Y}$ が凸集合であると仮定します。すると、$\mathcal{X} \cap \mathcal{Y}$ も凸状になります。これを確認するには、$a, b \in \mathcal{X} \cap \mathcal{Y}$ を考えてみてください。$\mathcal{X}$ と $\mathcal{Y}$ は凸状であるため、$a$ と $b$ を結ぶ線分は $\mathcal{X}$ と $\mathcal{Y}$ の両方に含まれます。それを考えると、それらは$\mathcal{X} \cap \mathcal{Y}$にも含まれている必要があり、私たちの定理を証明しています。 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

この結果はわずかな労力で強化できます。凸集合$\mathcal{X}_i$が与えられると、それらの交点$\cap_{i} \mathcal{X}_i$は凸状になります。逆が真でないことを確認するために、2 つの分断された集合 $\mathcal{X} \cap \mathcal{Y} = \emptyset$ を考えてみましょう。今度は $a \in \mathcal{X}$ と $b \in \mathcal{Y}$ を選んでください。$\mathcal{X} \cap \mathcal{Y} = \emptyset$ と仮定したので、$a$ と $b$ を結ぶ :numref:`fig_nonconvex` の線分には $\mathcal{X}$ にも $\mathcal{Y}$ にもない部分が含まれている必要があります。したがって、線分は$\mathcal{X} \cup \mathcal{Y}$にもないため、一般に凸集合の和集合は凸である必要はないことが証明されています。 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

通常、深層学習の問題は凸集合で定義されます。たとえば、$d$ 次元の実数ベクトルの集合である $\mathbb{R}^d$ は凸集合です ($\mathbb{R}^d$ の任意の 2 点間の線は $\mathbb{R}^d$ に残ります)。場合によっては、$\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$ で定義される半径 $r$ のボールなど、制限された長さの変数を使用します。 

### 凸関数

凸集合ができたので、*凸関数* $f$ を導入できます。凸集合 $\mathcal{X}$ が与えられた場合、関数 $f: \mathcal{X} \to \mathbb{R}$ は、すべての $x, x' \in \mathcal{X}$ とすべての $\lambda \in [0, 1]$ の関数が*凸* になります。 

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

これを説明するために、いくつかの関数をプロットし、どの関数が要件を満たすかを確認します。以下では、凸型と非凸型の関数をいくつか定義します。

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

予想どおり、余弦関数は*nonconvex* ですが、放物線と指数関数はです。$\mathcal{X}$ が凸集合であるという要件は、条件が意味をなすために必要であることに注意してください。そうしないと、$f(\lambda x + (1-\lambda) x')$ の結果が明確に定義されない可能性があります。 

### ジェンセンの不等式

凸関数 $f$ が与えられた場合、最も有用な数学ツールの 1 つは*ジェンセンの不等式* です。これは、凸の定義の一般化に相当します。 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

$\alpha_i$ は $\sum_i \alpha_i = 1$ と $X$ が確率変数になるような非負の実数です。言い換えれば、凸関数の期待値は期待値の凸関数以上であり、凸関数は通常より単純な式です。最初の不等式を証明するために、凸性の定義を一度に合計の1つの項に繰り返し適用します。 

ジェンセンの不等式の一般的な応用例の一つは、より複雑な表現をより単純な表現で束縛することです。たとえば、部分的に観測された確率変数の対数尤度に関して適用できます。つまり、 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

$\int P(Y) P(X \mid Y) dY = P(X)$年以来。これは変分法で使用できます。ここで、$Y$ は一般に観測されない確率変数で、$P(Y)$ は分布の最適な推測、$P(X)$ は $Y$ を積分した分布です。たとえば、クラスタリングでは $Y$ がクラスタラベルになり、$P(X \mid Y)$ がクラスタラベルを適用する場合の生成モデルになります。 

## プロパティ

凸関数には多くの有用な特性があります。以下では、一般的に使用されるものをいくつか説明します。 

### 局所的最小値は大域的最小値である

まず、凸関数の局所的最小値も大域的最小値です。次のように、矛盾によってそれを証明できます。 

凸集合 $\mathcal{X}$ に定義された凸関数 $f$ について考えてみます。$x^{\ast} \in \mathcal{X}$ が局所的最小値であると仮定します。$0 < |x - x^{\ast}| \leq p$ を満たす $x \in \mathcal{X}$ では $f(x^{\ast}) < f(x)$ になるように、小さい正の値が存在します。 

局所的最小値 $x^{\ast}$ はグローバル最小値 $f$ ではないと仮定します。$f(x') < f(x^{\ast})$ の $x' \in \mathcal{X}$ が存在します。$\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ のような $\lambda \in [0, 1)$ も存在するため、$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$ となります。  

しかし、凸関数の定義によれば、 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

これは $x^{\ast}$ が局所的最小値であるという私たちの声明と矛盾します。したがって、$f(x') < f(x^{\ast})$の$x' \in \mathcal{X}$は存在しません。局所的最小値 $x^{\ast}$ もグローバル最小値です。 

たとえば、凸関数 $f(x) = (x-1)^2$ の局所的最小値は $x=1$ で、これはグローバル最小値でもあります。

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

凸関数の局所的最小値が大域的最小値でもあることは非常に便利です。つまり、関数を最小化すると「行き詰まる」ことはできません。ただし、これは、グローバル最小値が複数存在し得ない、あるいは存在する可能性さえあるという意味ではないことに注意してください。たとえば、関数 $f(x) = \mathrm{max}(|x|-1, 0)$ は $[-1, 1]$ の間隔で最小値に達します。逆に、関数 $f(x) = \exp(x)$ は $\mathbb{R}$ では最小値に達しません。$x \to -\infty$ の場合は $0$ まで漸近しますが、$f(x) = 0$ の $x$ はありません。 

### 以下の凸関数の集合は凸である

凸関数の *below sets* を使って凸集合を簡単に定義できます。具体的には、凸集合 $\mathcal{X}$ に定義された凸関数 $f$ が与えられた場合、それより下の集合 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

凸状です。  

これをすぐに証明しよう。どの $x, x' \in \mathcal{S}_b$ についても $\lambda \in [0, 1]$ である限り $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ を示す必要があることを思い出してください。$f(x) \leq b$と$f(x') \leq b$以降、凸の定義により、  

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### 凸性と二次導関数

関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ の 2 次導関数が存在する場合は常に $f$ が凸かどうかを調べるのは非常に簡単です。必要なのは、$f$ のヘッシアンが半正定値であるかどうかをチェックすることだけです。つまり $\nabla^2f \succeq 0$、つまり $\mathbf{H}$ でヘッセ行列$\nabla^2f$、$\mathbf{x} \in \mathbb{R}^n$ すべてに対して $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ を表します。たとえば、関数 $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ は $\nabla^2 f = \mathbf{1}$ から凸です。つまり、ヘッシアンは単位行列です。 

形式的には、2 次微分可能な 1 次元関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ は、その 2 次導関数 $f'' \geq 0$ の場合にのみ凸になります。2 次微分可能な多次元関数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ では、ヘッシアン $\nabla^2f \succeq 0$ の場合にのみ凸になります。 

まず、一次元のケースを証明する必要があります。$f$ の凸状が $f'' \geq 0$ を意味することを確認するために、次の事実を使用します。 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

二次導関数は有限差分の極限によって与えられるので、次のようになります。 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

$f'' \geq 0$ が $f$ が凸であることを暗示することを確認するには、$f'' \geq 0$ が $f'$ が単調に非減少する関数であることを暗示しているという事実を使用します。$a < x < b$ を $\mathbb{R}$ の 3 つのポイントとします。$x = (1-\lambda)a + \lambda b$ と $\lambda \in (0, 1)$ です。平均値定理によると、$\alpha \in [a, x]$ と $\beta \in [x, b]$ が存在し、 

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

単調性によって $f'(\beta) \geq f'(\alpha)$、それゆえ 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

$x = (1-\lambda)a + \lambda b$ 以来、私達は 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

こうして凸状を証明する。 

次に、多次元のケースを証明する前に補題が必要です。$f: \mathbb{R}^n \rightarrow \mathbb{R}$ は、すべての $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ の場合に限り凸です。 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

凸状です。 

$f$ の凸が $g$ が凸であることを意味することを証明するために、すべての $a, b, \lambda \in [0, 1]$ (したがって $0 \leq \lambda a + (1-\lambda) b \leq 1$) についてそれを示すことができます。 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

逆を証明するために、すべての$\lambda \in [0, 1]$についてそれを示すことができます  

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

最後に、上記の補題と一次元ケースの結果を用いて、多次元ケースは次のように証明できます。多次元関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ は、$\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$ ($z \in [0,1]$) がすべて凸である場合に限り凸になります。一次元の場合によると、これはすべての $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ に対して $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) の場合にのみ成り立ちます。これは、正の半正定値行列の定義では $\mathbf{H} \succeq 0$ に相当します。 

## 制約

凸最適化の優れた特性の 1 つは、制約を効率的に処理できることです。つまり、以下の形式の*制約付き最適化*問題を解くことができます。 

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

$f$ は目的関数で、関数 $c_i$ は制約関数です。$c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$の場合を考えてみましょう。この場合、パラメータ $\mathbf{x}$ はユニットボールに拘束されます。2 番目の制約が $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$ の場合、これは半空間に配置されたすべての $\mathbf{x}$ に相当します。両方の制約を同時に満たすことは、ボールのスライスを選択することになります。 

### ラグランジアン

一般に、制約付き最適化問題を解くのは困難です。それに対処する1つの方法は、かなり単純な直感による物理学に由来します。箱の中にボールが入っているところを想像してみてください。ボールは最も低い場所に転がり、ボックスの側面がボールにかけることができる力と重力のバランスが取られます。つまり、目的関数の勾配 (つまり、重力) は、制約関数の勾配によってオフセットされます (壁が「押し戻される」ため、ボールはボックスの内側に留まる必要があります)。コンストレイントの中にはアクティブでないものもあることに注意してください。ボールが接触していない壁はボールに力を加えられません。 

*Lagrangian* $L$ の導出をスキップすると、上記の推論は次のサドルポイント最適化問題で表現できます。 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

ここで、変数 $\alpha_i$ ($i=1,\ldots,n$) は、制約が適切に適用されることを保証する、いわゆる*ラグランジュ乗数* です。すべての$i$に対して$c_i(\mathbf{x}) \leq 0$を確保するのに十分な大きさで選択されています。たとえば、$c_i(\mathbf{x}) < 0$ が自然と存在する $\mathbf{x}$ については、$\alpha_i = 0$ を選択することになります。さらに、これはサドルポイント最適化問題であり、$\alpha_i$ すべてに対して $L$ を *最大化* すると同時に $\mathbf{x}$ に対して *最小化* したいと考えています。関数$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$に到達する方法を説明する豊富な文献があります。この目的のためには、$L$ のサドルポイントが、元の制約付き最適化問題が最適に解かれる場所であることを知っていれば十分です。 

### 罰則

制約付き最適化問題を少なくとも *近似* 満たす方法の 1 つは、ラグランジアン $L$ を適応させることです。$c_i(\mathbf{x}) \leq 0$ を満たすのではなく、$\alpha_i c_i(\mathbf{x})$ を単に目的関数 $f(x)$ に加算します。これにより、制約が過度に違反されることがなくなります。 

実際、私たちはずっとこのトリックを使ってきました。:numref:`sec_weight_decay` での体重減少を考えてみましょう。その中で $\frac{\lambda}{2} \|\mathbf{w}\|^2$ を目的関数に追加して、$\mathbf{w}$ が大きくなりすぎないようにします。制約付き最適化の観点から、これにより半径$r$に対して$\|\mathbf{w}\|^2 - r^2 \leq 0$が保証されることがわかります。$\lambda$ の値を調整すると、$\mathbf{w}$ のサイズを変えることができます。 

一般に、ペナルティを追加することは、拘束を近似的に満たすのに適した方法です。実際には、これは正確な満足度よりもはるかに堅牢であることがわかります。さらに、非凸問題では、凸の場合に正確なアプローチを魅力的にする多くの特性 (最適性など) はもはや成り立たない。 

### [投影]

制約を満たすためのもう 1 つの方法として、投影法があります。繰り返しになりますが、例えば :numref:`sec_rnn_scratch` でグラデーションクリッピングを扱っているときなどに、このような現象に遭遇しました。そこで、勾配の長さが$\theta$で囲まれていることを確認しました。 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

これは、半径$\theta$のボールへの$\mathbf{g}$の*投影*であることが判明しました。より一般的には、凸集合 $\mathcal{X}$ 上の投影は次のように定義されます。 

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

$\mathcal{X}$の$\mathbf{x}$に最も近い地点はどれですか。  

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

投影法の数学的な定義は、少し抽象的に聞こえるかもしれません。:numref:`fig_projections` は幾分明確に説明しています。その中には、円とダイヤモンドの2つの凸セットがあります。両方のセット (黄色) の内側にあるポイントは、投影中も変化しません。両方のセットの外側 (黒) のポイントは、元のポイント (黒) に一番近いセット (赤) の内側のポイント (赤) に投影されます。$L_2$ボールの場合、方向は変わりませんが、ダイヤモンドの場合に見られるように、一般的にはそうである必要はありません。 

凸投影の用途の 1 つに、スパースな重みベクトルの計算があります。この例では、:numref:`fig_projections` のダイヤモンドケースの一般化されたバージョンである $L_1$ ボールに重量ベクトルを投影します。 

## [概要

ディープラーニングのコンテキストでは、凸関数の主な目的は、最適化アルゴリズムを動機付けて、最適化アルゴリズムを詳細に理解できるようにすることです。以下では、それに応じて勾配降下法と確率的勾配降下法がどのように導出されるかを見ていきます。 

* 凸集合の交点は凸状です。組合はそうではありません。
* 凸関数の期待値は、期待値の凸関数 (ジェンセンの不等式) 以上です。
* 2 倍微分可能な関数は、ヘッセシアン (2 次導関数の行列) が半正定値の場合に限り凸になります。
* 凸コンストレイントはラグランジュ関数を使って追加できます。実際には、目的関数にペナルティを付けて単純に追加できます。
* 投影は、元のポイントに最も近い凸集合内のポイントにマップされます。

## 演習

1. 集合内の点の間にすべての線を描き、線が含まれているかどうかをチェックして、集合の凸状性を検証すると仮定します。
    1. 境界上の点だけをチェックすれば十分であることを証明する。
    1. セットの頂点だけをチェックするだけで十分であることを証明する。
1. $p$ ノルムを使用して $r$ の半径のボールを $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ で表します。$\mathcal{B}_p[r]$がすべての$p \geq 1$に対して凸であることを証明してください。
1. 凸関数 $f$ と $g$ が与えられた場合、$\mathrm{max}(f, g)$ も凸関数であることがわかります。$\mathrm{min}(f, g)$が凸でないことを証明してください。
1. softmax関数の正規化が凸であることを証明する。具体的には、$f(x) = \log \sum_i \exp(x_i)$の凸状性を証明してください。
1. 線形部分空間、すなわち $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$ が凸集合であることを証明する。
1. $\mathbf{b} = \mathbf{0}$ の線形部分空間の場合、投影 $\mathrm{Proj}_\mathcal{X}$ は行列 $\mathbf{M}$ に対して $\mathbf{M} \mathbf{x}$ と書けることを証明します。
1. 二微分可能な凸関数 $f$ に対して $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ を書けることを示してください。
1. $\|\mathbf{w}\|_1 > 1$ をもつベクトル $\mathbf{w} \in \mathbb{R}^d$ が与えられたら、$L_1$ 単位球の投影を計算します。
    1. 中間ステップとして、ペナルティを課された目標 $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$ を書き出し、与えられた $\lambda > 0$ の解を計算します。
    1. 多くの試行錯誤をせずに$\lambda$の「正しい」価値を見つけることができますか？
1. 凸集合 $\mathcal{X}$ と 2 つのベクトル $\mathbf{x}$ と $\mathbf{y}$ を指定して、投影によって距離が決して増えないこと、つまり $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$ が証明されます。

[Discussions](https://discuss.d2l.ai/t/350)

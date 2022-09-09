```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 微積分
:label:`sec_calculus`

長い間、円の面積を計算する方法は謎のままでした。その後、古代ギリシャの数学者アルキメデスは、円の内側に頂点の数が増える一連の多角形を内接させるという巧妙なアイデアを思いつきました（:numref:`fig_circle_area`）。$n$の頂点を持つ多角形の場合、$n$の三角形が得られます。円をより細かく分割するにつれて、各三角形の高さは半径 $r$ に近づきます。同時に、円弧と割線の比が多数の頂点に対して1に近づくので、その底辺は$2 \pi r/n$に近づきます。したがって、三角形の面積は$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$に近づきます。  

![Finding the area of a circle as a limit procedure.](../img/polygon-circle.svg)
:label:`fig_circle_area`

この制限手順は両方につながります 
*微分計算* と*積分* 
(:numref:`sec_integral_calculus`)。前者は、引数を操作することで関数の値を増減する方法を教えてくれます。これは、損失関数を減らすためにパラメーターを繰り返し更新するディープラーニングで直面する「最適化問題」に役立ちます。最適化は、モデルをトレーニングデータに適合させる方法を扱い、微積分はその重要な前提条件です。しかし、私たちの最終的な目標は、*これまで見られなかった*データでうまく機能することであることを忘れないでください。この問題は*一般化*と呼ばれ、他の章の主要な焦点となるでしょう。 

## デリバティブと差別化

簡単に言えば、*微分*は、引数の変化に対する関数の変化率です。デリバティブは、各パラメータを無限に少しだけ*増加*または*減少*した場合、損失関数がどれだけ速く増加または減少するかを教えてくれます。正式には、スカラーからスカラーにマップする関数$f: \mathbb{R} \rightarrow \mathbb{R}$の場合、[**$f$の*微分* $x$は**として定義されます] 

(**$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$ドル**) :eqlabel:`eq_derivative` 

右側のこの用語は*リミット*と呼ばれ、指定された変数が特定の値に近づくと式の値がどうなるかを示します。この制限は、摂動 $h$ と関数値 $f(x + h) - f(x)$ の変化との比が、サイズをゼロに縮小したときに収束する割合を示します。 

$f'(x)$が存在する場合、$f$は$x$で*微分可能*と言われ、$f'(x)$がセットのすべての$x$に対して存在する場合、$f$はこのセットで微分可能であると言います。精度や受信動作特性（AUC）の下の領域など、最適化したい多くの機能を含め、すべての機能が差別化できるわけではありません。しかし、損失の微分を計算することは、ディープニューラルネットワークを学習するためのほぼすべてのアルゴリズムにおいて重要なステップであるため、代わりに微分可能な*サロゲート*を最適化することがよくあります。 

微分$f'(x)$は、$x$に対する$f(x)$の*瞬間的な*変化率として解釈できます。例を挙げて直感を身につけましょう。(**$u = f(x) = 3x^2-4x$.の定義を挙げてください**)

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$、$\frac{f(x+h) - f(x)}{h}$**]（**$h$が$0$に近づくと、$2$に近づきます**）この実験は数学的な証明の厳密さを欠いていますが、すぐに$f'(1) = 2$であることがわかります。

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

デリバティブには同等の表記規則がいくつかあります。$y = f(x)$を考えると、次の式は同等です。 

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

ここで、記号$\frac{d}{dx}$と$D$は*微分演算子*です。以下に、いくつかの一般的な関数の派生物を示します。 

$$\begin{aligned} \frac{d}{dx} C & = 0 && \text{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \text{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1} \end{aligned}$$

微分可能な関数から構成される関数は、しばしばそれ自体が微分可能です。次のルールは、微分可能な関数 $f$ と $g$、および定数 $C$ のコンポジションを扱う場合に便利です。 

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \text{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \text{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \text{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \text{Quotient rule} \end{aligned}$$

これを使用して、規則を適用して$3 x^2 - 4x$の微分を求めることができます。 

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

$x = 1$を接続すると、この位置では微分が$2$であることがわかります。微分は、特定の位置における関数の*傾き*を教えてくれることに注意してください。   

## ビジュアル化ユーティリティ

[**`matplotlib`ライブラリを使用して関数の傾きを可視化できます**]。いくつかの関数を定義する必要があります。その名前が示すように、`use_svg_display`は`matplotlib`に、より鮮明な画像のためにSVG形式でグラフィックを出力するように指示します。コメント `# @save `は特別な修飾子で、関数、クラス、その他のコードブロックを`d2l`パッケージに保存して、コードを繰り返さずに後で呼び出せるようにする (例:`d2l.use_svg_display()`)。

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

便利なことに、`set_figsize`でフィギュアサイズを設定できます。インポート文`from matplotlib import pyplot as plt`は `# @save` in the `d2l` package, we can call `d2l .plt` でマークされていたので。

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` 関数は、軸をラベル、範囲、スケールなどのプロパティに関連付けることができます。

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

これら 3 つの関数を使用して、複数の曲線をオーバーレイする `plot` 関数を定義できます。ここでのコードの多くは、入力のサイズと形状が一致することを保証するだけです。

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None: axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

ここで、[**$u = f(x)$とその接線$y = 2x - 3$を$x=1$にプロット**] できます。ここで、係数$2$は接線の傾きです。

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏微分と勾配
:label:`subsec_calculus-grad`

これまで、私たちはただ一つの変数の関数を区別してきました。ディープラーニングでは、*多くの*変数の関数も扱う必要があります。このような*多変量*関数に適用される微分の概念を簡単に紹介します。 

$y = f(x_1, x_2, \ldots, x_n)$を$n$の変数を持つ関数とします。$i^\mathrm{th}$ パラメータ $x_i$ に対する $y$ の*偏微分* は次のようになります。 

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

$\frac{\partial y}{\partial x_i}$を計算するために、$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$を定数として扱い、$x_i$に対する$y$の微分を計算することができます。偏導関数の次の表記規則はすべて共通で、すべて同じ意味です。 

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

多変量関数の偏導関数をそのすべての変数に対して連結して、関数の*勾配*と呼ばれるベクトルを得ることができます。関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ の入力が $n$ 次元ベクトル $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ で、出力がスカラーであるとします。$\mathbf{x}$ に対する関数 $f$ の勾配は、$n$ の偏微分のベクトルです。 

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

あいまいさがない場合、$\nabla_{\mathbf{x}} f(\mathbf{x})$ は通常 $\nabla f(\mathbf{x})$ に置き換えられます。次のルールは、多変量関数を区別するのに便利です。 

* すべての $\mathbf{A} \in \mathbb{R}^{m \times n}$ には $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ と $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$ があります。
* 正方行列 $\mathbf{A} \in \mathbb{R}^{n \times n}$ には $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ があり、特に
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$。 

同様に、どのマトリックス $\mathbf{X}$ にも $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$ があります。  

## 連鎖規則

ディープラーニングでは、深くネストされた関数（関数（関数の...））を扱っているため、関心の勾配を計算するのが難しいことがよくあります。幸いなことに、*チェーンルール*がこれを処理します。単一変数の関数に戻り、$y = f(g(x))$ と、基礎となる関数 $y=f(u)$ と $u=g(x)$ の両方が微分可能であると仮定します。連鎖規則には、  

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

多変量関数に戻ると、$y = f(\mathbf{u})$には変数$u_1, u_2, \ldots, u_m$があり、各$u_i = g_i(\mathbf{x})$には変数$x_1, x_2, \ldots, x_n$、つまり$\mathbf{u} = g(\mathbf{x})$があるとします。そして、連鎖規則は次のように述べています 

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \text{ and thus } \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

ここで、$\mathbf{A} \in \mathbb{R}^{n \times m}$ は、ベクトル $\mathbf{x}$ に対するベクトル $\mathbf{u}$ の微分を含む*行列* です。したがって、勾配を評価するには、ベクトルマトリックスの積を計算する必要があります。これが、線形代数がディープラーニングシステムの構築において不可欠な構成要素である主な理由の1つです。  

## ディスカッション

ディープトピックの表面をスクラッチしたばかりですが、すでにいくつかの概念に焦点が当てられています。1つ目は、差別化のための構成ルールを無意識に適用でき、勾配を*自動*で計算できることです。このタスクは創造性を必要としないため、認知力を他の場所に集中させることができます。第2に、ベクトル値関数の導関数を計算するには、出力から入力までの変数の依存グラフをトレースするときに、行列を乗算する必要があります。特に、このグラフは、関数を評価するときは*順方向*方向に、勾配を計算するときは*後方*方向にトラバースされます。後の章では、連鎖規則を適用するための計算手順であるバックプロパゲーションを正式に紹介します。 

最適化の観点から、勾配を使用すると、損失を減らすためにモデルのパラメーターをどのように移動するかを決定できます。この本全体で使用されている最適化アルゴリズムの各ステップでは、勾配を計算する必要があります。 

## 演習

1. これまでのところ、デリバティブのルールは当然のことと考えていました。定義と制限を使用すると、(i) $f(x) = c$、(ii) $f(x) = x^n$、(iii) $f(x) = e^x$、(iv) $f(x) = \log x$ のプロパティが証明されます。
1. 同じように、第一原理から積、和、商の法則を証明します。 
1. 積則の特殊なケースとして、定数倍則が続くことを証明します。 
1. $f(x) = x^x$ の微分を計算します。 
1. $f'(x) = 0$が一部の$x$にとってどういう意味ですか？関数$f$と、これが当てはまる可能性のある場所$x$の例を挙げてください。 
1. 関数 $y = f(x) = x^3 - \frac{1}{x}$ をプロットし、その接線を $x = 1$ にプロットします。
1. 関数 $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ の勾配を求めます。
1. 関数$f(\mathbf{x}) = \|\mathbf{x}\|_2$の勾配は何ですか？$\mathbf{x} = \mathbf{0}$はどうなりますか？
1. $u = f(x, y, z)$と$x = x(a, b)$、$y = y(a, b)$、$z = z(a, b)$の場合の連鎖ルールを書けますか？
1. 可逆関数$f(x)$が与えられると、その逆関数$f^{-1}(x)$の微分を計算します。ここにその$f^{-1}(f(x)) = x$があり、逆に$f(f^{-1}(y)) = y$があります。ヒント:これらのプロパティを派生に使用してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:

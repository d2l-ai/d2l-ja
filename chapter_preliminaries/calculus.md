# 微積分
:label:`sec_calculus`

多角形の面積を見つけることは、少なくとも2500年前、古代ギリシア人が多角形を三角形に分割して面積を合計するまで不思議なままでした。円などの湾曲した形状の領域を見つけるために、古代ギリシア人はそのような形状のポリゴンを内接しました。:numref:`fig_circle_area` に示すように、辺の長さが等しい内接多角形は、円の近似がよくなります。このプロセスは「枯渇方法」とも呼ばれています。 

![Find the area of a circle with the method of exhaustion.](../img/polygon-circle.svg)
:label:`fig_circle_area`

実際、枯渇法は*積分計算* (:numref:`sec_integral_calculus` で説明される) の由来です。2000年以上経った今後、微積分学のもうひとつの分野、*微分積分* が発明されました。微分積分の最も重要な応用例の中でも、最適化問題では「最良」なことをどう行うかが考慮されます。:numref:`subsec_norms_and_objectives` で説明したように、このような問題は深層学習では広く見られます。 

ディープラーニングでは、モデルを「トレーニング」し、連続的に更新することで、見るデータが増えていくにつれてモデルがどんどん良くなるようにします。通常、より良くなるということは、「私たちのモデルがどれほど悪い*？」という質問に答えるスコアである*損失関数*を最小化することを意味します。この質問は見た目よりも微妙です。最終的に、私たちが本当に気にかけているのは、これまでに見たことのないデータに対して優れたパフォーマンスを発揮するモデルを作成することです。しかし、実際に見ることができるデータにしかモデルをあてはめられません。したがって、モデルをフィッティングするタスクを、(i) *最適化*: 観測されたデータにモデルをフィッティングするプロセス、(ii) *一般化*: 正確なデータセットを超える妥当性を持つモデルの作成方法を導く数学的原理と実践者の知恵に分解できます。彼らを訓練するのに使われた例。 

後の章で最適化の問題と手法を理解しやすくするために、ここではディープラーニングで一般的に使用される微分計算について簡単に説明します。 

## デリバティブと微分

まず、ほとんどすべてのディープラーニング最適化アルゴリズムにおいて重要なステップである微分の計算に取り組みます。ディープラーニングでは、通常、モデルのパラメーターに関して微分可能な損失関数を選択します。簡単に言うと、各パラメータについて、そのパラメータを極小の「増加」または「減少」した場合に、損失がどれだけ急速に増減するかを判断できるということです。 

入力と出力の両方がスカラーである関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ があるとします。[**$f$ の*微分* は次のように定義されます**] 

(** $f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$ドル**) :eqlabel:`eq_derivative` 

この制限が存在する場合。$f'(a)$ が存在する場合、$a$ では $f$ は*微分可能* であると言われます。$f$ が区間の数ごとに微分可能である場合、この関数はこの区間で微分可能です。:eqref:`eq_derivative` の導関数 $f'(x)$ は、$x$ に対する $f(x)$ の*瞬間的な*変化率として解釈できます。いわゆる瞬時変化率は、$x$ の $h$ の変動 $h$ に基づいており、$0$ に近づいています。 

導関数を説明するために、例を挙げて実験してみましょう。(** $u = f(x) = 3x^2-4x$ を定義してください**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$ を設定し $h$ を $0$ に近づけると $\frac{f(x+h) - f(x)}{h}$ の数値結果**] :eqref:`eq_derivative` (** $2$ に近づく**) この実験は数学的な証明ではありませんが、$x=1$ のときに導関数 $u'$ が $2$ であることが後でわかります。

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

デリバティブの同等の表記法をいくつか理解しておきましょう。$y = f(x)$ を指定すると、$x$ と $y$ はそれぞれ関数 $f$ の独立変数と従属変数です。次の式は同等です。 

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

シンボル $\frac{d}{dx}$ と $D$ は、*微分* の演算を示す*微分演算子* です。一般的な機能を区別するために、次のルールを使用できます。 

* $DC = 0$ ($C$ は定数です)
* $Dx^n = nx^{n-1}$ (*べき乗則*、$n$ は任意の実数です)
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

上記の共通関数のようないくつかのより単純な関数から形成される関数を区別するために、以下のルールが役に立ちます。関数 $f$ と $g$ が両方とも微分可能で、$C$ が定数であると仮定すると、*定数の倍数規則* があります。 

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*sumルール* 

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*製品ルール* 

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

そして*商の法則* 

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

これで $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$ を見つけるために、上記の規則のいくつかを適用できます。したがって、$x = 1$ を設定すると $u' = 2$ が得られます。これは、このセクションの以前の実験でサポートされており、数値結果は $2$ に近づきます。この微分は、$x = 1$ のときの曲線 $u = f(x)$ に対する接線の傾きでもあります。 

[**このような微分の解釈を視覚化するために、Python でよく使われるプロットライブラリである `matplotlib`, **] を使います。`matplotlib` で生成される Figure のプロパティを設定するには、いくつかの関数を定義する必要があります。次の例では、`use_svg_display` 関数は `matplotlib` パッケージを指定して、より鮮明なイメージのために svg Figure を出力します。コメント `# @save `は、以下の関数、クラス、文を `d2l` パッケージに保存する特別なマークなので、あとで再定義することなく直接呼び出せる (`d2l.use_svg_display()` など) ことができます。

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

`set_figsize` 関数を定義して Figure のサイズを指定します。ここでは `d2l.plt` を直接使用することに注意してください。これは、インポートステートメント `from matplotlib import pyplot as plt` が、序文で `d2l` パッケージに保存されるようマークされているためです。

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

次の `set_axes` 関数は `matplotlib` によって生成される図形座標軸のプロパティを設定します。

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Figure コンフィギュレーション用のこれら 3 つの関数を使用して、本書全体で多くの曲線を視覚化する必要があるため、複数の曲線を簡潔にプロットする関数 `plot` を定義します。

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

これで、[**関数 $u = f(x)$ とその接線 $y = 2x - 3$ を $x=1$ にプロット**] できます。ここで、係数 $2$ は接線の傾きです。

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏微分

これまで、1つの変数の関数の微分を扱ってきました。ディープラーニングでは、関数は多くの場合、*多数* 個の変数に依存しています。したがって、微分の概念をこれらの「多変量関数」にまで拡張する必要があります。 

$y = f(x_1, x_2, \ldots, x_n)$ を $n$ 個の変数をもつ関数とします。$i^\mathrm{th}$ パラメーター $x_i$ に対する $y$ の *偏微分* は次のようになります。 

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

$\frac{\partial y}{\partial x_i}$ を計算するには、$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ を定数として扱い、$x_i$ に対する $y$ の導関数を計算します。偏微分の表記法では、以下は同等です。 

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## グラデーション
:label:`subsec_calculus-grad`

多変量関数の偏導関数をそのすべての変数に対して連結して、関数の*gradient* ベクトルを求めることができます。関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ の入力が $n$ 次元のベクトル $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ で、出力がスカラーであるとします。$\mathbf{x}$ に対する関数 $f(\mathbf{x})$ の勾配は $n$ 偏微分のベクトルです。 

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

$\nabla_{\mathbf{x}} f(\mathbf{x})$ は、あいまいさがなければ $\nabla f(\mathbf{x})$ に置き換えられることがよくあります。 

$\mathbf{x}$ を $n$ 次元のベクトルとすると、多変量関数を微分するときには次の規則がよく使われます。 

* すべての$\mathbf{A} \in \mathbb{R}^{m \times n}$、$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$、
* すべての$\mathbf{A} \in \mathbb{R}^{n \times m}$、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$、
* すべての$\mathbf{A} \in \mathbb{R}^{n \times n}$、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$、
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$。

同様に、マトリックス $\mathbf{X}$ については $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$ があります。後で説明するように、勾配はディープラーニングにおける最適化アルゴリズムの設計に役立ちます。 

## 連鎖規則

しかし、そのようなグラデーションは見つけにくい場合があります。これは、ディープラーニングの多変量関数は*合成*であることが多いため、これらの関数を区別するために前述のルールを適用しない可能性があるためです。幸いなことに、*chainルール*によって複合関数を区別することができます。 

まず、単一変数の関数について考えてみましょう。関数 $y=f(u)$ と $u=g(x)$ がどちらも微分可能であると仮定すると、連鎖規則は次のようになります。 

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

ここで、関数が任意の数の変数をもつ、より一般的なシナリオに注目しましょう。微分可能関数 $y$ に変数 $u_1, u_2, \ldots, u_m$ があり、各微分可能関数 $u_i$ には変数 $x_1, x_2, \ldots, x_n$ があるとします。$y$ は $x_1, x_2, \ldots, x_n$ の関数であることに注意してください。そして、連鎖規則は次のようになります。 

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

どんな$i = 1, 2, \ldots, n$にも合います。 

## [概要

* 微分積分学と積分微積分は微積分学の2つの分岐であり、前者は深層学習におけるユビキタス最適化問題に適用できます。
* 微分は、その変数に対する関数の瞬間的な変化率として解釈できます。これは、関数の曲線に対する接線の傾きでもあります。
* 勾配は、そのすべての変数に対する多変量関数の偏導関数を成分とするベクトルです。
* 連鎖則により、複合関数を区別することができます。

## 演習

1. $x = 1$ の場合、関数 $y = f(x) = x^3 - \frac{1}{x}$ とその接線をプロットします。
1. 関数 $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ の勾配を求めます。
1. 関数$f(\mathbf{x}) = \|\mathbf{x}\|_2$の勾配は何ですか？
1. $u = f(x, y, z)$ と $x = x(a, b)$、$y = y(a, b)$、$z = z(a, b)$ の場合のチェーンルールを書き出せますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:

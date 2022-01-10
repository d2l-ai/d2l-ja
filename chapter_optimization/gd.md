# グラデーションディセント
:label:`sec_gd`

このセクションでは、*勾配降下* の基本概念を紹介します。ディープラーニングで直接使用されることはほとんどありませんが、確率的勾配降下法アルゴリズムを理解するには、勾配降下法を理解することが重要です。たとえば、学習率が高すぎると最適化問題が発散することがあります。この現象は勾配降下法ですでに見られます。同様に、前処理は勾配降下法で一般的な手法であり、より高度なアルゴリズムにも引き継がれます。単純な特殊なケースから始めましょう。 

## 1 次元勾配降下法

1 次元の勾配降下法は、勾配降下法アルゴリズムによって目的関数の値が減少する理由を説明する優れた例です。連続微分可能な実数値関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ について考えてみましょう。テイラー展開を使用すると、 

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

つまり、一次近似では $f(x+\epsilon)$ は関数値 $f(x)$ と一次導関数 $f'(x)$ によって $x$ で与えられます。小さい$\epsilon$の場合、負の勾配の方向に移動すると$f$が減少すると仮定するのは不合理ではありません。単純にするために、固定ステップサイズ $\eta > 0$ を選択し、$\epsilon = -\eta f'(x)$ を選択します。これを上のTaylor拡張に差し込むと、 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

導関数 $f'(x) \neq 0$ が消えなければ $\eta f'^2(x)>0$ 以降は進歩する。さらに、高次の項が無関係になるほど小さな $\eta$ をいつでも選択できます。したがって、我々は到着する 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

これは、もし我々が 

$$x \leftarrow x - \eta f'(x)$$

$x$ を反復すると、関数 $f(x)$ の値が減少する可能性があります。したがって、勾配降下法では、最初に初期値 $x$ と定数 $\eta > 0$ を選択し、停止条件に達するまで (勾配 $|f'(x)|$ の大きさが十分に小さい場合や、反復回数が一定の値。 

簡単にするために、勾配降下法の実装方法を示す目的関数 $f(x)=x^2$ を選択します。$x=0$ が $f(x)$ を最小化する解であることはわかっていますが、この単純な関数を使用して $x$ がどのように変化するかを観察します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

次に、$x=10$ を初期値として使用し、$\eta=0.2$ と仮定します。勾配降下法を使用して $x$ を 10 回反復すると、最終的に $x$ の値が最適解に近づくことがわかります。

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$ を超える最適化の進行状況は、次のようにプロットできます。

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 学習率
:label:`subsec_gd-learningrate`

学習率 $\eta$ は、アルゴリズム設計者が設定できます。学習率が小さすぎると、$x$ の更新が非常に遅くなり、より適切な解を得るためにより多くの反復が必要になります。このような場合に何が起こるかを示すために、$\eta = 0.05$ の同じ最適化問題の進捗状況を考えてみましょう。ご覧のとおり、10ステップ後でも、最適なソリューションにはまだほど遠いです。

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

逆に、過度に高い学習率を使用すると、$\left|\eta f'(x)\right|$ は 1 次のテイラー展開式には大きすぎる可能性があります。つまり、:eqref:`gd-taylor-2` の $\mathcal{O}(\eta^2 f'^2(x))$ という用語が意味を持つようになるかもしれません。この場合、$x$ の反復が $f(x)$ の値を下げることは保証できません。たとえば、学習率を $\eta=1.1$ に設定すると、$x$ は最適解 $x=0$ をオーバーシュートし、徐々に発散します。

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 局所的極小値

非凸関数で何が起こるかを説明するために、定数 $c$ の $f(x) = x \cdot \cos(cx)$ の場合を考えてみましょう。この関数は無限に多くの局所的最小値をもちます。学習率の選択と問題がどの程度適切に調整されているかによって、多くの解決策のうちの1つになることがあります。以下の例は、(非現実的に) 学習率が高いと局所的最小値が低くなる原因を示しています。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 多変量グラデーションディセント

単変量ケースをより直感的に理解できたので、$\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$ の状況を考えてみましょう。つまり、目的関数 $f: \mathbb{R}^d \to \mathbb{R}$ はベクトルをスカラーにマッピングします。それに対応して、その勾配も多変量です。これは $d$ の偏微分で構成されるベクトルです。 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

勾配内の各偏微分要素 $\partial f(\mathbf{x})/\partial x_i$ は、入力 $x_i$ に対する $\mathbf{x}$ における $f$ の変化率を示します。一変量の場合と同様に、多変量関数に対応するテイラー近似を使用して、何をすべきかのアイデアを得ることができます。特に、私たちはそれを持っています 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

言い換えると、$\boldsymbol{\epsilon}$ の 2 次の項まで、最も急な降下方向は負の勾配 $-\nabla f(\mathbf{x})$ によって与えられます。適切な学習率 $\eta > 0$ を選択すると、典型的な勾配降下法アルゴリズムが生成されます。 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

アルゴリズムが実際にどのように動作するかを確認するために、2 次元ベクトル $\mathbf{x} = [x_1, x_2]^\top$ を入力し、スカラーを出力とする目的関数 $f(\mathbf{x})=x_1^2+2x_2^2$ を構築します。勾配は $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ で与えられます。$\mathbf{x}$ の軌跡を初期位置 $[-5, -2]$ からの勾配降下により観測します。  

まず、ヘルパー関数がもう2つ必要です。1 つ目は更新関数を使用し、それを初期値に 20 回適用します。2 番目のヘルパーは $\mathbf{x}$ の軌跡を視覚化します。

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

次に、学習率 $\eta = 0.1$ に対する最適化変数 $\mathbf{x}$ の軌跡を観察します。20 ステップ後、$\mathbf{x}$ の値が最小値である $[0, 0]$ に近づくことがわかります。進歩はかなり遅いですが、かなり行儀が良いです。

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## アダプティブメソッド

:numref:`subsec_gd-learningrate` でわかるように、学習率 $\eta$ を「ちょうどいい」ものにするのは難しいです。小さすぎると、ほとんど進歩しません。大きすぎると、解は振動し、最悪の場合は発散することさえあります。$\eta$ を自動的に決定したり、学習率を選択しなくても済むとしたらどうでしょうか。この場合、目的関数の値と勾配だけでなく、その*曲率*も調べる二次法が役立ちます。これらの手法は、計算コストがかかるためディープラーニングに直接適用することはできませんが、以下に概説するアルゴリズムの望ましい特性の多くを模倣する高度な最適化アルゴリズムの設計方法を直感的に理解できます。 

### ニュートン法

ある関数 $f: \mathbb{R}^d \rightarrow \mathbb{R}$ のテイラー展開を見直すと、最初の項の後で停止する必要はない。実際、次のように書くことができます 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

煩雑な表記を避けるため、$\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$ を $f$ のヘッシアンと定義します。これは $d \times d$ 行列です。$d$ の小さい問題や単純な問題では、$\mathbf{H}$ は簡単に計算できます。一方、ディープニューラルネットワークでは $\mathcal{O}(d^2)$ エントリを格納するコストがかかるため、$\mathbf{H}$ は非常に大きくなる可能性があります。さらに、バックプロパゲーションによる計算にはコストがかかりすぎる可能性があります。今のところ、そのような考慮事項を無視して、どのようなアルゴリズムが得られるかを見てみましょう。 

結局のところ、$f$の最小値は$\nabla f = 0$を満たしています。:numref:`subsec_calculus-grad`の微積分法に従い、$\boldsymbol{\epsilon}$に関して:eqref:`gd-hot-taylor`の導関数をとり、高次の項を無視することによって、我々は到達する 

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

つまり、最適化問題の一部として、ヘッシアン $\mathbf{H}$ を反転する必要があります。 

簡単な例として、$f(x) = \frac{1}{2} x^2$ には $\nabla f(x) = x$ と $\mathbf{H} = 1$ があります。したがって、どの $x$ についても $\epsilon = -x$ が得られます。言い換えれば、*シングル*ステップで調整しなくても完全に収束できます。悲しいかな、ここで少しラッキーになりました。Taylorの拡張は$f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$以来正確でした。  

他の問題で何が起こるか見てみましょう。ある定数 $c$ に対して凸双曲線余弦関数 $f(x) = \cosh(cx)$ が与えられた場合、$x=0$ のグローバル最小値に数回の反復で到達することがわかります。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

ここで、定数 $c$ に対する $f(x) = x \cos(c x)$ のような、*非凸* 関数について考えてみましょう。結局、ニュートンの方法ではヘッシアンで割ることになることに注意してください。つまり、2 次導関数が*負*の場合、$f$ の値を「増やす」方向に進む可能性があります。これはアルゴリズムの致命的な欠陥です。実際に何が起こるか見てみましょう。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

これは見事に間違っていた。どうすれば直せる？1 つの方法は、ヘッシアンを絶対値として「固定」することです。もう一つの戦略は、学習率を取り戻すことです。これは目的を打ち負かすようですが、それほどではありません。二次情報があると、曲率が大きい場合は注意深く、目的関数が平坦な場合は常に長いステップを取ることができます。$\eta = 0.5$ のように、少し小さい学習率でこれがどのように機能するか見てみましょう。ご覧のとおり、非常に効率的なアルゴリズムがあります。

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 収束解析

ニュートン法の収束率を解析するのは、2 次導関数が非ゼロ、つまり $f'' > 0$ である凸および 3 倍微分可能な目的関数 $f$ についてのみです。多変量証明は、以下の一次元の議論を単純に拡張したもので、直感的にはあまり役に立たないので省略されています。 

$k^\mathrm{th}$ の反復での $x$ の値を $x^{(k)}$ で表し、$k^\mathrm{th}$ の反復での $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ を最適性からの距離とします。テイラー展開により、条件 $f'(x^*) = 0$ は次のように記述できます。 

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

これは$\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$に当てはまる上記の拡張を$f''(x^{(k)})$で割ると、 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

更新プログラム $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$ があることを思い出してください。この更新方程式をプラグインし、両辺の絶対値を取ると、 

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

したがって、境界のある $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ の領域にいるときは常に、誤差が二次的に減少します。  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

余談ですが、最適化の研究者はこれを*線形*収束と呼んでいますが、$\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ のような条件は*一定*収束率と呼ばれます。この分析にはいくつかの注意点があります。第一に、急速な収束の領域に到達する時期はあまり保証されていません。その代わり、一度到達すると、収束が非常に速くなることが分かります。第 2 に、この分析では $f$ が高次の導関数まで正常に動作することが必要です。$f$ には、その値がどのように変更されるかという点で「驚くべき」特性がないことを確認する必要があります。 

### プレコンディショニング

ヘッシアン全体を計算して格納するのは当然のことながら非常にコストがかかります。したがって、代替案を見つけることが望ましい。問題を改善する一つの方法は、*前処理*です。ヘッシアン全体の計算は避け、*diagonal* エントリのみを計算します。これにより、次の形式のアルゴリズムが更新されます。 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

これは完全なニュートン法ほど良くはありませんが、使用しないよりはるかによいです。この方法がよい理由を理解するために、1 つの変数が高さをミリメートルで表し、もう 1 つの変数が高さをキロメートルで表す場合を考えてみましょう。両方の自然スケールがメートル単位であると仮定すると、パラメータ化にはひどいミスマッチがあります。幸いなことに、前処理を使用するとこれが取り除かれます。勾配降下法による効果的な前処理は、変数 (ベクトル $\mathbf{x}$ の座標) ごとに異なる学習率を選択することになります。後で説明するように、プリコンディショニングは確率的勾配降下最適化アルゴリズムの革新の一部を推進しています。  

### ラインサーチによる勾配降下

勾配降下法における重要な問題の一つは、ゴールをオーバーシュートさせたり、進行が不十分になったりする可能性があることです。この問題を簡単に解決するには、ラインサーチを勾配降下法と組み合わせて使用します。つまり、$\nabla f(\mathbf{x})$ で指定された方向を使用して、$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$ を最小にする学習率 $\eta$ について二分探索を実行します。 

このアルゴリズムは急速に収束します (解析と証明については :cite:`Boyd.Vandenberghe.2004` を参照)。ただし、ディープラーニングの目的では、ライン探索の各ステップでデータセット全体で目的関数を評価する必要があるため、これはあまり実現可能ではありません。これはコストがかかりすぎて達成できません。 

## [概要

* 学習率が重要です。大きすぎると発散し、小さすぎて進歩しません。
* 勾配降下は局所的最小値で動けなくなることがあります。
* 高次元では、学習率の調整は複雑です。
* プリコンディショニングはスケールの調整に役立ちます。
* ニュートン法は、凸問題で正しく動作するようになると、はるかに高速になります。
* 非凸問題を調整せずにニュートン法を使用することに注意してください。

## 演習

1. 勾配降下法について、さまざまな学習率と目的関数を試します。
1. $[a, b]$ の間隔で凸関数を最小化するためにラインサーチを実装します。
    1. 二分探索のために導関数が必要ですか、つまり $[a, (a+b)/2]$ または $[(a+b)/2, b]$ のどちらを選ぶかを決めるために必要ですか。
    1. アルゴリズムの収束率はどれくらい速いですか？
    1. アルゴリズムを実装し、$\log (\exp(x) + \exp(-2x -3))$ の最小化に適用します。
1. $\mathbb{R}^2$ で定義された、勾配降下が非常に遅い目的関数を設計します。ヒント:異なる座標を異なる方法でスケーリングします。
1. 前処理を使用して、軽量版のニュートン法を実装します。
    1. 対角ヘッシアンを前処理行列として使用します。
    1. 実際の (符号付きの場合もある) 値ではなく、その絶対値を使用します。
    1. これを上記の問題に適用してください。
1. 上記のアルゴリズムをいくつかの目的関数 (凸か否か) に適用します。座標を $45$ 度回転するとどうなりますか？

[Discussions](https://discuss.d2l.ai/t/351)

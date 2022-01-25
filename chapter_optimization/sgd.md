# 確率的勾配降下法
:label:`sec_sgd`

以前の章では、トレーニング手順で確率的勾配降下法を使用し続けましたが、なぜそれが機能するのか説明しませんでした。これに光を当てるために、:numref:`sec_gd`で勾配降下法の基本原理を説明しました。このセクションでは、議論を続けます。
*確率的勾配降下* より詳細に。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 確率的勾配の更新

ディープラーニングでは、通常、目的関数はトレーニングデータセット内の各例の損失関数の平均になります。$n$ の例を含むトレーニングデータセットを想定して、インデックス $i$ のトレーニング例に対して $f_i(\mathbf{x})$ が損失関数であると仮定します。$\mathbf{x}$ はパラメーターベクトルです。次に、目的関数に到達します。 

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

$\mathbf{x}$ における目的関数の勾配は次のように計算されます。 

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

勾配降下法を使用すると、独立変数の各反復の計算コストは $\mathcal{O}(n)$ となり、$n$ に比例して増加します。したがって、トレーニングデータセットが大きいほど、反復ごとの勾配降下法のコストが高くなります。 

確率的勾配降下法 (SGD) は、各反復で計算コストを削減します。確率的勾配降下法の各反復で、データ例のインデックス $i\in\{1,\ldots, n\}$ をランダムに一様にサンプリングし、勾配 $\nabla f_i(\mathbf{x})$ を計算して $\mathbf{x}$ を更新します。 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

$\eta$ は学習率です。各反復の計算コストは、勾配降下の $\mathcal{O}(n)$ から定数 $\mathcal{O}(1)$ に低下することがわかります。さらに、確率的勾配 $\nabla f_i(\mathbf{x})$ は完全勾配 $\nabla f(\mathbf{x})$ の偏りのない推定値であることを強調しておきます。 

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

これは、平均して、確率的勾配が勾配の適切な推定値であることを意味します。 

ここで、平均が 0、分散 1 のランダムノイズを勾配に追加して、確率的勾配降下法をシミュレートすることで、勾配降下法と比較します。

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

ご覧のとおり、確率的勾配降下における変数の軌跡は、:numref:`sec_gd`で観測された勾配降下法よりもはるかにノイズが多いです。これは、勾配の確率的性質によるものです。つまり、最小値に近づいても、$\eta \nabla f_i(\mathbf{x})$ を介した瞬間的な勾配によって注入される不確実性の影響を受けます。50ステップ経っても品質はあまり良くありません。さらに悪いことに、追加の手順を実行しても改善されません（これを確認するには、より多くの手順を試すことをお勧めします）。これにより、唯一の選択肢が残ります。学習率を変更する$\eta$。しかし、これを小さすぎると、当初は意味のある進歩を遂げません。一方、大きすぎると、上記のように適切な解決策が得られません。これらの矛盾する目標を解決する唯一の方法は、最適化が進むにつれて学習率を「動的に」減らすことです。 

これは、学習率関数 `lr` をステップ関数 `sgd` に追加する理由でもあります。上記の例では、関連する `lr` 関数を定数に設定しているため、学習率スケジューリングのすべての機能が休止しています。 

## 動的学習率

$\eta$ を時間依存の学習率 $\eta(t)$ に置き換えると、最適化アルゴリズムの収束制御が複雑になります。特に、$\eta$ がどれだけ急速に減衰するかを把握する必要があります。速すぎる場合は、最適化を早めに停止します。減らすのが遅すぎると、最適化に時間がかかりすぎます。$\eta$ を長期的に調整する際に使用されるいくつかの基本的な戦略を以下に示します (より高度な戦略については後で説明します)。 

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polynomial decay}
\end{aligned}
$$

最初の*区分的定数*シナリオでは、最適化の進行が止まるたびに学習率を下げます。これは、ディープネットワークに学習させるための一般的な戦略です。あるいは、*指数関数的減衰*によって、もっと積極的に減らすこともできます。残念ながら、アルゴリズムが収束する前に早期に停止することがよくあります。一般的な選択肢は、$\alpha = 0.5$ の「多項式減衰」です。凸最適化の場合、このレートが適切に動作していることを示す証明が数多くあります。 

指数関数的減衰が実際にどのように見えるか見てみましょう。

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

予想どおり、パラメーターの分散は大幅に減少します。ただし、これは最適解 $\mathbf{x} = (0, 0)$ に収束しないという犠牲を払います。1000回の反復ステップを経た後でも、最適解からは程遠いままです。実際、アルゴリズムはまったく収束しません。一方、ステップ数の逆平方根で学習率が減衰する多項式減衰を使用すると、わずか50ステップで収束が良くなります。

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

学習率の設定方法には、他にも多くの選択肢があります。たとえば、小さなレートから始めて、ゆっくりとはいえ、急速に上昇し、再び下げることができます。小さい学習率と大きい学習率を交互に繰り返すこともできます。そのようなスケジュールは多種多様です。ここでは、包括的な理論分析が可能な学習率スケジュール、つまり凸状の設定での学習率に焦点を当ててみましょう。一般的な非凸問題では、非線形非凸問題の最小化は NP が難しいため、意味のある収束保証を得ることは非常に困難です。調査については、Tibshirani 2015の優秀な [講義ノート](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) を参照してください。 

## 凸型対物レンズの収束解析

凸目的関数の確率的勾配降下法の次の収束解析は任意であり、主に問題についてより直感的に伝えるのに役立ちます。私たちは自分自身を最も単純な証明:cite:`Nesterov.Vial.2000`の1つに限定します。目的関数が特に適切に振る舞う場合など、はるかに高度な証明手法が存在します。 

目的関数 $f(\boldsymbol{\xi}, \mathbf{x})$ がすべての $\boldsymbol{\xi}$ に対して $\mathbf{x}$ で凸であると仮定します。より具体的には、確率的勾配降下法の更新について考えます。 

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

$f(\boldsymbol{\xi}_t, \mathbf{x})$ は、ステップ $t$ で何らかの分布から抽出された学習例 $\boldsymbol{\xi}_t$ に対する目的関数で、$\mathbf{x}$ はモデルパラメーターです。で示す 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

予想されるリスクと$R^*$までに$\mathbf{x}$に関しては最低限になります。最後に $\mathbf{x}^*$ をミニマイザーとします ($\mathbf{x}$ が定義されているドメイン内に存在すると仮定します)。この場合、時間 $t$ の現在のパラメーター $\mathbf{x}_t$ とリスクミニマイザー $\mathbf{x}^*$ の間の距離を追跡し、時間が経つにつれて改善するかどうかを確認できます。 

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

確率勾配 $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ の $L_2$ ノルムが何らかの定数 $L$ で囲まれていると仮定します。 

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

私たちは主に、$\mathbf{x}_t$と$\mathbf{x}^*$の間の距離が*期待どおりにどのように変化するかに関心があります。実際、特定の一連のステップでは、$\boldsymbol{\xi}_t$ のいずれかに応じて、距離が大きくなる可能性があります。したがって、内積を束縛する必要があります。どの凸関数でも $f$ は $\mathbf{x}$ と $\mathbf{y}$ すべてに対して $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$ を保持するので、凸性によっては次のようになります。 

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

:eqref:`eq_sgd-L` と :eqref:`eq_sgd-f-xi-xstar` の両方の不等式を :eqref:`eq_sgd-xt+1-xstar` に差し込むと、時間 $t+1$ におけるパラメータ間の距離の境界が次のように得られます。 

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

これは、電流損失と最適損失の差が$\eta_t L^2/2$を上回る限り、進歩することを意味します。この差は必ずゼロに収束するため、学習率 $\eta_t$ も「消失*」する必要があります。 

次に、:eqref:`eqref_sgd-xt-diff`を超える期待値を取ります。これによって得られる 

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

最後のステップでは、$t \in \{1, \ldots, T\}$ の不等式を合計します。和は望遠鏡なので、下限を落とすことで得られます 

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

$\mathbf{x}_1$ が与えられていることを悪用したため、期待値を落とす可能性があることに注意してください。最終定義 

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

以来 

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

ジェンセンの不等式 (:eqref:`eq_jensens-inequality` で $i=t$、$\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ を設定) と $R$ の凸度によって、$E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$ という結果になります。 

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

これを不等式 :eqref:`eq_sgd-x1-xstar` に差し込むと、境界が生成されます。 

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

$r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ は、パラメーターの初期選択と最終結果の間の距離の境界です。つまり、収束の速度は、確率的勾配のノルムがどのように制限されるか ($L$) と、初期パラメーター値が最適性からどれだけ離れているか ($r$) に依存します。境界は $\mathbf{x}_T$ ではなく $\bar{\mathbf{x}}$ で表されることに注意してください。$\bar{\mathbf{x}}$ は最適化パスの平滑化バージョンであるため、このようになります。$r, L$ と $T$ がわかっているときはいつでも、学習率 $\eta = r/(L \sqrt{T})$ を選択できます。これは上限の $rL/\sqrt{T}$ として得られます。つまり、レート $\mathcal{O}(1/\sqrt{T})$ で最適解に収束します。 

## 確率的勾配と有限標本

これまでのところ、確率的勾配降下について話すときは、少し速くて緩いプレーをしてきました。インスタンス $x_i$ を描画し、通常は分布 $p(x, y)$ からラベル $y_i$ を付けて描画し、これを使用してモデルパラメーターを何らかの方法で更新すると仮定しました。特に、有限の標本サイズについては、一部の関数 $\delta_{x_i}$ および $\delta_{y_i}$ の離散分布 $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ により、確率的勾配降下法を実行できると単純に主張しました。 

しかし、これは私たちがやったことではありません。現在のセクションのおもちゃの例では、それ以外は非確率的勾配にノイズを追加しただけです。つまり、ペア$(x_i, y_i)$があるふりをしました。これはここで正当化されることが判明しました（詳細な議論については演習を参照）。さらに厄介なのは、これまでのすべての議論で、明らかにこれを行わなかったことです。代わりに、すべてのインスタンスを *正確に 1 回* 反復しました。これが望ましい理由を理解するために、逆を考えてみましょう。つまり、離散分布 (*置換あり) から $n$ 個の観測値をサンプリングしているということです。要素 $i$ をランダムに選択する確率は $1/n$ です。したがって、それを*少なくとも*1回選択するのは 

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

同様の推論により、あるサンプル（トレーニング例）をピッキングする確率は*正確に1回*で与えられることが分かります。 

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

これにより、*置換なし*のサンプリングに比べて、分散が大きくなり、データ効率が低下します。したがって、実際には後者を実行します (これは本書全体のデフォルトの選択です)。最後に、トレーニングデータセットを繰り返し通過すると、そのデータセットが*異なる*ランダムな順序でトラバースされることに注意してください。 

## [概要

* 凸問題では、学習率の幅広い選択肢に対して、確率的勾配降下法が最適解に収束することを証明できます。
* ディープラーニングでは、一般的にはそうではありません。しかし、凸問題の分析は、最適化へのアプローチ方法、つまり学習率をそれほど速くではなく徐々に減らす方法についての有益な洞察を与えてくれます。
* 学習率が小さすぎたり大きすぎたりすると、問題が発生します。実際には、適切な学習率は多くの場合、複数回の実験を経て初めて見つかります。
* トレーニングデータセットに例が多い場合は、勾配降下法の各反復を計算するコストが高くなるため、このような場合は確率的勾配降下法が優先されます。
* 確率的勾配降下法に対する最適性の保証は、通常、凸でない場合には利用できません。これは、チェックが必要な局所的最小値の数が指数関数的である可能性があるためです。

## 演習

1. 確率的勾配降下法と反復回数を変えて、異なる学習率スケジュールを試します。特に、最適解 $(0, 0)$ からの距離を反復回数の関数としてプロットします。
1. 関数 $f(x_1, x_2) = x_1^2 + 2 x_2^2$ について、勾配に正規ノイズを加えることは、損失関数 $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ を最小化することと等価であることを証明します。$\mathbf{x}$ は正規分布から引き出されます。
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ から置換ありでサンプリングする場合と、置換なしでサンプリングした場合の、確率的勾配降下法の収束を比較します。
1. ある勾配（またはそれに関連する座標）が他のすべての勾配よりも一貫して大きい場合、確率的勾配降下法ソルバーをどのように変更しますか？
1. $f(x) = x^2 (1 + \sin x)$ と仮定します。$f$には局所的最小値がいくつありますか？$f$ を最小化するために、すべての局所的最小値を評価する必要があるような方法で変更できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:

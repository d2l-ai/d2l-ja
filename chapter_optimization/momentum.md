# モメンタム
:label:`sec_momentum`

:numref:`sec_sgd` では、確率的勾配降下を実行するとき、すなわち、ノイズの多い勾配バリアントしか利用できない最適化を実行するとどうなるかを検討しました。特に、ノイズの多い勾配では、ノイズに直面した場合の学習率の選択には特に注意が必要であることに気付きました。急激に減らしすぎると、収束が停止します。私たちが寛大すぎると、ノイズが私たちを最適性から遠ざけ続けるので、十分に良い解に収束できません。 

## 基本

このセクションでは、特に実際に一般的である特定のタイプの最適化問題に対して、より効果的な最適化アルゴリズムを探ります。 

### リーキー平均

前のセクションでは、計算を高速化する手段としてミニバッチ SGD について説明しました。また、グラデーションを平均化すると分散量が減少するという素晴らしい副作用もありました。ミニバッチ確率的勾配降下法は次のように計算できます。 

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

表記を単純にするために、ここでは $t-1$ で更新された重みを使用して、サンプル $i$ の確率的勾配降下法として $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ を使用しました。ミニバッチの勾配の平均化を超えて、分散削減の効果から利益を得ることができればいいですね。このタスクを実行する 1 つのオプションは、勾配計算を「漏出平均」に置き換えることです。 

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

いくつかの $\beta \in (0, 1)$これにより、瞬間的なグラデーションが、複数の*過去*グラデーションで平均化されたグラデーションに置き換えられます。$\mathbf{v}$ は*momentum* と呼ばれます。これは、目的関数のランドスケープを転がる重いボールが過去の力を積分するのと同じように、過去の勾配を蓄積します。何が起こっているのかをもっと詳しく見るために $\mathbf{v}_t$ を再帰的に展開して 

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ が大きいと長距離の平均になりますが、$\beta$ が小さいと、勾配法に比べてわずかな補正に過ぎません。新しいグラデーションの置き換えは、特定のインスタンスで最も急な降下方向を指すのではなく、過去のグラデーションの加重平均の方向を指すようになりました。これにより、バッチの勾配を実際に計算するコストをかけずに、バッチの平均化のメリットのほとんどを実現できます。この平均化手順については、後ほど詳しく説明します。 

上記の推論は、運動量のある勾配など、現在*加速*勾配法として知られている方法の基礎を形成しました。最適化問題が条件が悪い (つまり、狭い峡谷のように進行が他の方向よりもはるかに遅い方向がある) 場合に、より効果的になるという追加の利点を享受します。さらに、後続の勾配を平均化して、より安定した降下方向を得ることができます。実際、ノイズのない凸問題でも加速の側面は、運動量が働き、運動量がうまく機能する主な理由の一つです。 

予想通り、その有効性のために、勢いはディープラーニング以降の最適化において十分に研究されたテーマです。詳細な分析とインタラクティブなアニメーションについては、美しい [解説記事](https://distill.pub/2017/momentum/) by :cite:`Goh.2017`) を参照してください。:cite:`Polyak.1964` によって提案されました。:cite:`Nesterov.2018` では、凸最適化のコンテキストで詳細な理論的議論が行われています。ディープラーニングのモメンタムは、古くから有益であることが知られていました。詳細については、:cite:`Sutskever.Martens.Dahl.ea.2013` の議論を参照してください。 

### 悪条件の問題

運動量法の幾何学的性質をよりよく理解するために、目的関数はそれほど快適ではありませんが、勾配降下法を再検討します。:numref:`sec_gd` では $f(\mathbf{x}) = x_1^2 + 2 x_2^2$、つまり中程度に歪んだ楕円体対物レンズを使用したことを思い出してください。この関数を $x_1$ の方向に引き伸ばすことで、この関数をさらに歪めます。 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

以前と同様に $f$ の最小値は $(0, 0)$ です。この関数は $x_1$ の方向に「非常に」フラットです。この新しい関数で以前のように勾配降下を実行するとどうなるか見てみましょう。ここでは $0.4$ の学習率を選びます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

構造上、$x_2$ 方向の勾配は、水平方向の $x_1$ よりもずっと高く、急激に変化します。したがって、2 つの望ましくない選択の間に立ち往生しています。学習率を小さくすると、解が $x_2$ 方向に発散しないようになりますが、$x_1$ 方向の収束が遅くなります。逆に、学習率が大きいと $x_1$ の方向には急速に進みますが、$x_2$ では発散します。次の例は、学習率が $0.4$ から $0.6$ にわずかに増加した後でも何が起こるかを示しています。$x_1$ 方向の収束は向上しますが、全体的な解析品質は大幅に低下します。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### モメンタム・メソッド

運動量法により、上述した勾配降下問題を解くことができます。上記の最適化トレースを見ると、過去の勾配の平均化がうまくいくと直感的に理解できるかもしれません。結局のところ、$x_1$ 方向では、適切に整列されたグラデーションが集約されるため、すべてのステップでカバーする距離が長くなります。逆に、勾配が振動する $x_2$ 方向では、互いに打ち消し合う振動により、集約勾配によってステップサイズが小さくなります。勾配 $\mathbf{g}_t$ の代わりに $\mathbf{v}_t$ を使用すると、次の更新方程式が生成されます。 

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$\beta = 0$ では、通常の勾配降下法が回復することに注意してください。数学的性質を深く掘り下げる前に、アルゴリズムが実際にどのように動作するかを簡単に見てみましょう。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

ご覧のとおり、以前と同じ学習率でも、勢いはうまく収束します。運動量パラメータを小さくするとどうなるか見てみましょう。$\beta = 0.25$ に半分にすると軌道がほとんど収束しません。それでも、運動量がない（解が発散する）場合よりもはるかに優れています。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

モーメンタムと確率的勾配降下法、特にミニバッチ確率的勾配降下法を組み合わせることができることに注意してください。唯一の変更点は、この場合のグラデーション $\mathbf{g}_{t, t-1}$ を $\mathbf{g}_t$ に置き換えることです。最後に、便宜上 $\mathbf{v}_0 = 0$ を時刻 $t=0$ に初期化します。リーキーアベレージングが実際に更新に与える影響を見てみましょう。 

### 有効サンプル重量

$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$ということを思い出してください。この制限では、項の合計は $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$ になります。言い換えれば、勾配降下法または確率的勾配降下法でサイズ $\eta$ のステップを取るのではなく、サイズ $\frac{\eta}{1-\beta}$ のステップを取ると同時に、潜在的にはるかに優れた降下方向を処理します。これらは1つに2つのメリットがあります。$\beta$ のさまざまな選択肢に対して重み付けがどのように動作するかを説明するために、次の図を考えてみましょう。

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 実践的実験

実際にモメンタムがどのように機能するか、つまり適切なオプティマイザのコンテキスト内で使用された場合を見てみましょう。そのためには、もう少しスケーラブルな実装が必要です。 

### ゼロからの実装

(ミニバッチ) 確率的勾配降下法と比較すると、モーメンタム法は一連の補助変数、すなわち速度を維持する必要があります。勾配 (および最適化問題の変数) と同じ形をしています。以下の実装では、これらの変数を `states` と呼んでいます。

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

これが実際にどのように機能するか見てみましょう。

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

運動量ハイパーパラメーター `momentum` を 0.9 に増やすと、実効サンプルサイズ $\frac{1}{1 - 0.9} = 10$ が大幅に大きくなります。問題を管理するために、学習率をわずかに$0.01$に下げます。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

学習率を下げると、スムーズでない最適化問題の問題がさらに解決されます。$0.005$ に設定すると、良好な収束特性が得られます。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 簡潔な実装

標準の `sgd` ソルバーには既に運動量が組み込まれているため、Gluon でやることはほとんどありません。マッチングパラメータを設定すると、非常に似た軌跡が得られます。

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 理論的解析

これまでのところ、$f(x) = 0.1 x_1^2 + 2 x_2^2$の2D例はかなり不自然に思えました。これは、少なくとも凸二次目的関数を最小化する場合には、実際に遭遇する可能性のある問題のタイプをかなり代表していることがわかります。 

### 二次凸関数

関数を考えてみましょう 

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

これは一般的な二次関数です。正定値行列 $\mathbf{Q} \succ 0$、つまり正の固有値をもつ行列の場合、最小化は $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ で、最小値は $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$ です。したがって、$h$を次のように書き直すことができます。 

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

勾配は $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ で与えられます。つまり、$\mathbf{x}$ とミニマイザーの間の距離に $\mathbf{Q}$ を掛けたものになります。したがって、運動量も項$\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$の線形結合になります。 

$\mathbf{Q}$ は正定値であるため、直交 (回転) 行列 $\mathbf{O}$ と正の固有値の対角行列 $\boldsymbol{\Lambda}$ に対して $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ を介して固有系に分解できます。これにより、変数を $\mathbf{x}$ から $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ に変更して、非常に簡略化された式を得ることができます。 

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

ここに$b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$。$\mathbf{O}$ は直交行列にすぎないため、勾配が意味のある方法で摂動することはありません。$\mathbf{z}$で表すと、勾配降下は次のようになります。 

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

この式の重要な事実は、勾配降下は異なる固有空間間で*混在*しないということです。つまり、$\mathbf{Q}$ の固有系で表すと、最適化問題は座標的に進行します。これは勢いにも当てはまります。 

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

これを行うことで、次の定理を証明しました。凸二次関数の運動量がある場合とない場合の勾配降下は、二次行列の固有ベクトルの方向で座標方向の最適化に分解されます。 

### スカラー関数

上記の結果を考えると、関数 $f(x) = \frac{\lambda}{2} x^2$ を最小化するとどうなるか見てみましょう。勾配降下には 

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

$|1 - \eta \lambda| < 1$ ステップの後 $x_t = (1 - \eta \lambda)^t x_0$ になるので、この最適化は指数関数的に収束します。これは、学習率 $\eta$ を $\eta \lambda = 1$ まで上げると、収束率が最初にどのように向上するかを示しています。それを超えて物事は発散し、$\eta \lambda > 2$では最適化問題が発散します。

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

モーメンタムの場合の収束を解析するために、まず、更新方程式を 2 つのスカラー (1 つは $x$ 用、もう 1 つはモメンタム $v$) で書き直します。これにより、次の結果が得られます。 

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

$\mathbf{R}$ を使用して、収束動作を制御する $2 \times 2$ を示しました。$t$ ステップの後、$[v_0, x_0]$ の初期選択は $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$ になります。したがって、収束速度を決定するのは $\mathbf{R}$ の固有値までです。素晴らしいアニメーションについては [蒸留ポスト]（https://distill.pub/2017/momentum/) of :cite:`Goh.2017`）を、詳細な分析については:cite:`Flammarion.Bach.2015`を参照してください。$0 < \eta \lambda < 2 + 2 \beta$ モメンタムが収束することを示すことができます。これは、勾配降下法の $0 < \eta \lambda < 2$ と比較すると、実行可能なパラメーターの範囲が広くなります。また、一般的には$\beta$という大きな値が望ましいことも示唆しています。さらなる詳細にはかなりの技術的詳細が必要であり、興味のある読者は元の出版物を調べることをお勧めします。 

## [概要

* Momentum は、勾配を過去の勾配の漏れやすい平均に置き換えます。これにより、収束が大幅に高速化されます。
* ノイズのない勾配降下法と (ノイズの多い) 確率的勾配降下法のどちらにも適しています。
* Momentum は、確率的勾配降下法で発生する可能性がはるかに高い最適化プロセスの停止を防ぎます。
* 勾配の有効数は $\frac{1}{1-\beta}$ で与えられます。これは、過去のデータを累乗的に下げたためです。
* 凸二次問題の場合、これを明確に詳細に分析できます。
* 実装は非常に簡単ですが、追加の状態ベクトル (momentum $\mathbf{v}$) を保存する必要があります。

## 演習

1. 運動量ハイパーパラメーターと学習率の他の組み合わせを使用して、さまざまな実験結果を観察し、分析します。
1. 複数の固有値、つまり $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$、例えば $\lambda_i = 2^{-i}$ をもつ二次問題に対して GD とモーメンタムを試してみてください。$x_i = 1$ の初期化で $x$ の値がどのように減少するかをプロットします。
1. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$ の最小値と最小値を導出する。
1. 勢いのある確率的勾配降下法を実行すると何が変わるのですか？運動量のあるミニバッチ確率的勾配降下法を使用するとどうなりますか？パラメータを試してみる？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:

# rmsProp
:label:`sec_rmsprop`

:numref:`sec_adagrad` の重要な問題の 1 つは、事前定義されたスケジュール $\mathcal{O}(t^{-\frac{1}{2}})$ で学習率が低下することです。これは一般に凸問題には適していますが、深層学習で発生するような凸でない問題には適さない場合があります。しかし、アダグラッドの座標的適応性は前処理行列として非常に望ましい。 

:cite:`Tieleman.Hinton.2012` は、レートスケジューリングを座標適応学習レートから切り離す簡単な修正として、rmsProp アルゴリズムを提案しました。問題は、アダグラッドが勾配 $\mathbf{g}_t$ の二乗を状態ベクトル $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ に累積していることです。その結果、$\mathbf{s}_t$ は正規化がないため、アルゴリズムが収束するにつれて本質的に直線的に増加し続けます。 

この問題を解決する 1 つの方法は $\mathbf{s}_t / t$ を使用することです。$\mathbf{g}_t$ の分布が妥当であれば、これは収束します。残念ながら、プロシージャーは値の完全な軌跡を記憶しているため、制限の動作が問題になり始めるまでには非常に長い時間がかかる場合があります。別の方法として、モメンタム法で使用したのと同じ方法で漏れ平均を使用する方法があります。つまり、一部のパラメーター $\gamma > 0$ に $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ を使用します。他のすべてのパーツを変更しないままにすると、rmsProp が生成されます。 

## アルゴリズム

方程式を詳しく書いてみましょう。 

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

通常、定数 $\epsilon > 0$ は $10^{-6}$ に設定されます。これは、ゼロ除算や過度に大きいステップサイズに悩まされないようにするためです。この拡張により、座標ごとに適用されるスケーリングとは無関係に学習率 $\eta$ を自由に制御できるようになりました。リーキー平均に関しては、モメンタム法の場合に以前に適用されたのと同じ推論を適用できます。$\mathbf{s}_t$ イールドの定義を広げる 

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

:numref:`sec_momentum` では以前と同様に $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$ を使います。したがって、重みの合計は $1$ に正規化され、観測値の半減期は $\gamma^{-1}$ になります。$\gamma$ のさまざまな選択肢について、過去 40 個のタイムステップの重みを視覚化してみましょう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## ゼロからの実装

前と同じように、二次関数 $f(\mathbf{x})=0.1x_1^2+2x_2^2$ を使用して rmsProp の軌跡を観察します。:numref:`sec_adagrad` では、学習率 0.4 で Adagrad を使用した場合、学習率の低下が早すぎたため、アルゴリズムの後の段階で変数の移動が非常に遅くなったことを思い出してください。$\eta$ は個別に制御されるため、rmsProp ではこのようなことは起こりません。

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

次に、ディープネットワークで使用する rmsProp を実装します。これも同様に簡単です。

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

初期学習率を 0.01 に設定し、重み付け項 $\gamma$ を 0.9 に設定します。つまり、$\mathbf{s}$ は、過去の $1/(1-\gamma) = 10$ 個の二乗勾配の観測値の平均を集計します。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 簡潔な実装

rmsProp はかなり一般的なアルゴリズムなので、`Trainer` インスタンスでも使用できます。必要なのは、`rmsprop` という名前のアルゴリズムを使用してインスタンスを作成し、$\gamma$ をパラメーター `gamma1` に割り当てることだけです。

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## [概要

* rmsProp は、両方とも勾配の 2 乗を使用して係数をスケーリングする点で Adagrad と非常によく似ています。
* rmsProp はリーキー平均化を勢いで共有します。ただし、rmsProp はこの手法を使用して係数単位の前処理行列を調整します。
* 学習率は、実験者が実際にスケジュールする必要があります。
* 係数 $\gamma$ は、座標単位のスケールを調整するときのヒストリの長さを決定します。

## 演習

1. $\gamma = 1$ を設定すると実験的にどうなりますか？なぜ？
1. 最適化問題を回転して $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ を最小化します。コンバージェンスはどうなりますか？
1. Fashion-MNIST でのトレーニングなど、実際の機械学習の問題で rmsProp がどうなるかを試してみてください。学習率を調整するために、さまざまな選択肢を試してみてください。
1. 最適化の進行に合わせて $\gamma$ を調整しますか?rmsProp はこれに対してどの程度敏感ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:

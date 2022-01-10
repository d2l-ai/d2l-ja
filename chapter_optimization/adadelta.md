# アダデルタ
:label:`sec_adadelta`

アダデルタはアダグラッド (:numref:`sec_adagrad`) のもう一つの変種です。主な違いは、学習率が座標に適応する量が減少するという事実にあります。また、従来は、変化の量自体を将来の変化のためのキャリブレーションとして使用するため、学習率を持たないと呼ばれていました。このアルゴリズムは :cite:`Zeiler.2012` で提案されました。これまでのアルゴリズムの議論を考えると、これはかなり簡単です。  

## アルゴリズム

簡単に言うと、Adadelta は 2 つの状態変数 $\mathbf{s}_t$ を使用して勾配の 2 番目のモーメントの漏れ平均を保存し、$\Delta\mathbf{x}_t$ を使用して、モデル自体にパラメーターの変化の 2 番目のモーメントの漏洩平均を保存します。他の出版物や実装との互換性を保つために、著者のオリジナルの表記法と命名法を使っていることに注意してください (momentum、Adagrad、rmsProp、および Adadelta で同じ目的を果たすパラメータを示すために異なるギリシャ語の変数を使うべき本当の理由は他にありません)。  

アダデルタの技術的な詳細はこちらです。パラメーター du jour が $\rho$ であるとすると、:numref:`sec_rmsprop` と同様に、次のリークのある更新が取得されます。 

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop` との違いは、再スケーリングされたグラデーション $\mathbf{g}_t'$ で更新を実行することです。 

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

では、再スケーリングされたグラデーション $\mathbf{g}_t'$ は何ですか？次のように計算できます。 

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

$\Delta \mathbf{x}_{t-1}$ は、再スケーリングされた勾配 $\mathbf{g}_t'$ の 2 乗の漏れやすい平均です。$\Delta \mathbf{x}_{0}$ を $0$ に初期化し、各ステップで $\mathbf{g}_t'$ で更新します。つまり、 

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

数値の安定性を維持するために $\epsilon$ ($10^{-5}$ などの小さい値) が追加されます。 

## 実装

アダデルタは、変数ごとに $\mathbf{s}_t$ と $\Delta\mathbf{x}_t$ の 2 つの状態変数を維持する必要があります。これにより、以下のような実装になります。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$ を選択すると、パラメーターの更新ごとに半減期が 10 になります。これはかなりうまく機能する傾向があります。次のような動作になります。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

簡潔な実装のために、`Trainer` クラスの `adadelta` アルゴリズムを使用するだけです。これにより、呼び出しがはるかにコンパクトになるように、次のワンライナーが生成されます。

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## [概要

* Adadelta には学習率パラメーターがありません。代わりに、パラメータ自体の変化率を使用して学習率を調整します。 
* Adadelta では、勾配の 2 番目のモーメントとパラメーターの変化を保存するために 2 つの状態変数が必要です。 
* Adadelta は漏出平均を使用して、適切な統計量の実行中の推定値を保持します。 

## 演習

1. $\rho$ の値を調整します。何が起きる？
1. $\mathbf{g}_t'$ を使用せずにアルゴリズムを実装する方法を示します。なぜこれが良いアイデアなのでしょうか？
1. アダデルタは本当に学習率は無料ですか？アダデルタを壊す最適化問題を見つけられますか？
1. Adadelta と Adagrad および RMS prop を比較して、それらの収束動作について議論します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:

# アダム
:label:`sec_adam`

このセクションに先立つ議論の中で、効率的な最適化のためのいくつかの手法に出会いました。ここで詳しくまとめましょう。 

* 最適化問題を解く場合、:numref:`sec_sgd` は勾配降下法よりも効果的であることがわかりました。たとえば、冗長データに対する固有の復元力があるためです。 
* :numref:`sec_minibatch_sgd` では、1 つのミニバッチでより大きな観測値セットを使用することで、ベクトル化による効率が大幅に向上することがわかりました。これは、マルチマシン、マルチ GPU、および全体的な並列処理を効率的に行うための鍵となります。 
* :numref:`sec_momentum` では、過去の勾配の履歴を集約して収束を高速化するメカニズムが追加されました。
* :numref:`sec_adagrad` では、座標単位のスケーリングを使用して、計算効率の良い前処理行列を実現しました。 
* :numref:`sec_rmsprop` 座標単位のスケーリングを学習率調整から切り離しました。 

Adam :cite:`Kingma.Ba.2014` は、これらすべての手法を 1 つの効率的な学習アルゴリズムにまとめました。予想通り、このアルゴリズムは、深層学習で使用できる、より堅牢で効果的な最適化アルゴリズムの 1 つとして人気が高まっています。しかし、問題がないわけではありません。特に :cite:`Reddi.Kale.Kumar.2019` は、分散制御が不十分なために Adam が発散できる状況があることを示しています。フォローアップ作業で :cite:`Zaheer.Reddi.Sachan.ea.2018` は、これらの問題に対処する Yogi という修正プログラムを Adam に提案しました。これについては後で詳しく説明します。とりあえず、Adam アルゴリズムを見ていきましょう。  

## アルゴリズム

Adam の重要な要素の 1 つは、指数加重移動平均 (漏れ平均化とも呼ばれる) を使用して、勾配のモーメンタムと第 2 モーメントの両方の推定値を取得することです。つまり、状態変数を使用します。 

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

ここで $\beta_1$ と $\beta_2$ は非負の重み付けパラメータです。一般的な選択肢は $\beta_1 = 0.9$ と $\beta_2 = 0.999$ です。つまり、分散推定値はモメンタム項よりも*はるかにゆっくり*移動します。$\mathbf{v}_0 = \mathbf{s}_0 = 0$ を初期化すると、最初は小さい値に対してかなりのバイアスがあることに注意してください。これは、$\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ が項を再正規化するという事実を利用することで対処できます。これに対応して、正規化された状態変数は次の式で与えられます。  

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

適切な見積もりを準備して、更新方程式を書き出すことができます。まず、rmsProp と非常によく似た方法で勾配を再スケーリングして取得します。 

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

rmsProp とは異なり、今回のアップデートではグラデーションそのものではなくモメンタム $\hat{\mathbf{v}}_t$ を使用しています。また、$\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$ ではなく $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ を使用して再スケーリングが行われるため、外観上のわずかな違いがあります。前者は実際にはおそらくわずかに優れているため、rmsProp からの逸脱です。通常 $\epsilon = 10^{-6}$ は、数値の安定性と忠実度のトレードオフを考慮して選びます。  

これで、更新を計算するためのすべての要素が整いました。これは少し反気候的で、フォームの簡単な更新があります 

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

アダムのデザインを見直すと、そのインスピレーションは明らかです。モーメンタムとスケールは状態変数ではっきりとわかります。それらのかなり独特な定義は、用語をデバイアスすることを強制します (これは、初期化と更新の条件が少し違えば修正できます)。第二に、rmsProp を考えると、両方の用語の組み合わせは非常に簡単です。最後に、明示的学習率 $\eta$ により、収束の問題に対処するためにステップ長を制御できます。  

## 実装 

Adamをゼロから実装するのはそれほど難しいことではありません。便宜上、タイムステップカウンター $t$ は `hyperparams` ディクショナリに格納されています。それを超えてすべてが簡単です。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Adam を使ってモデルをトレーニングする準備が整いました。ここでは $\eta = 0.01$ の学習率を使用します。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

`adam` は Gluon `trainer` 最適化ライブラリの一部として提供されているアルゴリズムの 1 つなので、より簡潔な実装は簡単です。したがって、Gluon の実装に対して設定パラメータを渡すだけで済みます。

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## ヨギ

Adamの問題の一つは、$\mathbf{s}_t$の第2モーメント推定値が爆発すると、凸状の設定でも収束に失敗する可能性があることです。修正として :cite:`Zaheer.Reddi.Sachan.ea.2018` は $\mathbf{s}_t$ の改良された更新 (および初期化) を提案しました。何が起こっているのかを理解するために、Adam の更新を次のように書き直してみましょう。 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

$\mathbf{g}_t^2$ の分散が大きい場合や更新がまばらである場合、$\mathbf{s}_t$ は過去の値をすぐに忘れてしまう可能性があります。この問題を解決するには、$\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ を $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$ に置き換えます。これで、更新の大きさは偏差の量に依存しなくなりました。これにより、Yogi の更新が生成されます 

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

著者らはさらに、初期のポイントワイズ推定だけでなく、より大きな初期バッチで運動量を初期化することを推奨している。詳細は議論にとって重要ではなく、この収束がなくてもかなり良いままであるため、詳細は省略します。

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## [概要

* Adam は、多くの最適化アルゴリズムの特徴をかなり堅牢な更新ルールにまとめています。 
* Adam は rmsProp に基づいて作成され、ミニバッチ確率勾配で EWMA も使用します。
* Adam は、バイアス補正を使用して、モーメンタムと 2 番目のモーメントを推定するときに、起動が遅くなるように調整します。 
* 有意な分散をもつ勾配では、収束の問題が発生することがあります。それらは、より大きなミニバッチを使用するか、$\mathbf{s}_t$ の改善された見積もりに切り替えることで修正できます。ヨギはそのような代替案を提供します。 

## 演習

1. 学習率を調整し、実験結果を観察して分析します。
1. バイアス補正を必要としないように、モーメンタムとセカンドモーメントの更新を書き換えることはできますか？
1. 収束するときに学習率 $\eta$ を下げる必要があるのはなぜですか。
1. アダムが発散しヨギが収束するケースを作ろうか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:

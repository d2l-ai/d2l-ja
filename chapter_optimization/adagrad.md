# アダグラッド
:label:`sec_adagrad`

まれにしか発生しない特徴に関する学習問題を検討することから始めましょう。 

## 疎な特徴量と学習率

言語モデルをトレーニングしているとします。精度を高めるために、通常は $\mathcal{O}(t^{-\frac{1}{2}})$ 以下のレートで、学習を続けながら学習率を下げます。次に、スパースな特徴、つまりまれにしか発生しない特徴についてのモデルトレーニングについて考えてみましょう。これは自然言語ではよくあることです。例えば、*学習*よりも*前処理*という言葉が出てくる可能性ははるかに低いです。ただし、コンピュテーショナルアドバタイズやパーソナライズされた協調フィルタリングなど、他の分野でも一般的です。結局のところ、少数の人々だけが関心を持つことがたくさんあります。 

頻度の低いフィーチャに関連付けられたパラメーターは、これらのフィーチャが発生するたびに意味のある更新のみを受け取ります。学習率が低下すると、共通の特徴のパラメーターが最適値にかなり早く収束する状況に陥る可能性があります。一方、頻度の低い特徴については、最適値を決定する前に十分な頻度で観測できない状況に陥る可能性があります。言い換えると、頻度の高い特徴では学習率の低下が遅すぎ、頻度の低い特徴では学習率の低下が速すぎます。 

この問題を解決する可能性のあるハックは、特定のフィーチャの検出回数をカウントし、これを学習率を調整するためのクロックとして使用することです。つまり、$\eta = \frac{\eta_0}{\sqrt{t + c}}$ という形式の学習率を選択するのではなく、$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ を使用できます。ここで $s(i, t)$ は、時点 $t$ までに観測されたフィーチャ $i$ の非ゼロの数をカウントしています。これは実際には意味のあるオーバーヘッドなしで実装するのが非常に簡単です。ただし、スパース性がまったくなく、勾配が非常に小さく、まれにしか大きくないデータだけがある場合は失敗します。結局のところ、観測された特徴とみなされるものの間に線を引く場所は不明です。 

Adagrad by :cite:`Duchi.Hazan.Singer.2011` は、かなり粗雑なカウンター $s(i, t)$ を、以前に観測された勾配の二乗の集合体で置き換えることでこれに対処しています。特に、学習率を調整する手段として $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ を使用しています。これには 2 つの利点があります。1 つは、グラデーションがいつ十分に大きくなるかを判断する必要がなくなったことです。2 つ目は、グラデーションの大きさに合わせて自動的にスケーリングされることです。大きなグラデーションに日常的に対応する座標は大幅に縮小されますが、グラデーションの小さい座標ではより緩やかに処理されます。実際には、これはコンピュテーショナルアドバタイズおよび関連する問題に対して非常に効果的な最適化手順につながります。しかし、これは前処理の文脈で最もよく理解されるアダグラッドに内在する追加の利点のいくつかを隠しています。 

## プレコンディショニング

凸最適化問題は、アルゴリズムの特性を解析するのに適しています。結局のところ、ほとんどの非凸問題では、意味のある理論的保証を導き出すことは困難ですが、*intuition* と*insight* はしばしば引き継がれます。$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$ を最小化する問題を見てみましょう。 

:numref:`sec_momentum` で見たように、この問題を固有分解 $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ で書き換えて、各座標を個別に解くことができる、非常に単純化された問題に到達することができます。 

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

ここでは $\mathbf{x} = \mathbf{U} \mathbf{x}$ を使用し、結果的に $\mathbf{c} = \mathbf{U} \mathbf{c}$ を使用しました。修正された問題は、最小値として $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$、最小値は $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$ です。$\boldsymbol{\Lambda}$ は $\mathbf{Q}$ の固有値を含む対角行列なので、計算がはるかに簡単です。 

$\mathbf{c}$ を少し混乱させると、$f$ のミニマイザーにわずかな変化しか見られないと期待します。残念ながらそうではありません。$\mathbf{c}$ のわずかな変更は $\bar{\mathbf{c}}$ でも同様にわずかな変化につながりますが、$f$ (および $\bar{f}$) のミニマイザーには当てはまりません。固有値 $\boldsymbol{\Lambda}_i$ が大きい場合は常に $\bar{x}_i$ と最小の $\bar{f}$ には小さな変化しか見られません。逆に、$\boldsymbol{\Lambda}_i$ が小さければ $\bar{x}_i$ の変更は劇的なものになる可能性があります。最大固有値と最小固有値の比率を最適化問題の条件数といいます。 

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

条件数値 $\kappa$ が大きい場合、最適化問題を正確に解くことは困難です。大きなダイナミックレンジの値を正しく得るには、注意が必要です。私たちの分析は、いくぶん素朴ではあるが明白な疑問につながる。すべての固有値が $1$ になるように空間を歪ませることで問題を単純に「修正」することはできないだろう。理論上、これは非常に簡単です。$\mathbf{x}$ から $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$ の 1 つに問題をスケール変更するには $\mathbf{Q}$ の固有値と固有ベクトルのみが必要です。新しい座標系では、$\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ を $\|\mathbf{z}\|^2$ に簡略化できます。悲しいかな、これはかなり実用的ではない提案です。固有値と固有ベクトルの計算は、一般に、実際の問題を解くよりも *はるかに*コストがかかります。 

固有値を正確に計算するのはコストがかかるかもしれませんが、それを推測して多少おおよそ計算することは、何もしないよりもすでにずっと優れているかもしれません。特に、$\mathbf{Q}$ の対角線エントリを使用し、それに応じてスケールを変更することができます。これは、固有値を計算するよりも「大いに」安価です。 

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

この場合、$\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$、特にすべての $i$ に対して $\tilde{\mathbf{Q}}_{ii} = 1$ があります。ほとんどの場合、これにより条件数が大幅に簡略化されます。たとえば、前に説明したケースでは、問題は軸に揃えられているため、当面の問題を完全に排除できます。 

深層学習では、通常、目的関数の 2 次導関数にアクセスすることすらできません。$\mathbf{x} \in \mathbb{R}^d$ の場合、ミニバッチでも 2 次導関数には $\mathcal{O}(d^2)$ の空間と計算作業が必要で、実際には実行不可能です。アダグラッドの独創的なアイデアは、計算が比較的安価で効果的なヘシアンのわかりにくい対角線 (勾配自体の大きさ) にプロキシを使用することです。 

これがなぜ機能するのかを見るために、$\bar{f}(\bar{\mathbf{x}})$ を見てみましょう。我々はそれを持っている 

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

$\bar{\mathbf{x}}_0$ は $\bar{f}$ の最小化です。したがって、勾配の大きさは $\boldsymbol{\Lambda}$ と最適性からの距離の両方に依存します。$\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ が変わらなければ、これで十分です。結局のところ、この場合、勾配$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$の大きさで十分です。AdaGradは確率的勾配降下法アルゴリズムなので、最適でも非ゼロ分散の勾配が見られます。その結果、勾配の分散をヘッシアンのスケールの安価な代用として安全に使用できます。徹底的な分析はこのセクションの範囲外です (数ページになります)。詳細については、読者の :cite:`Duchi.Hazan.Singer.2011` を参照してください。 

## アルゴリズム

上からの議論を形式化しましょう。変数 $\mathbf{s}_t$ を使用して、以下のように過去の勾配分散を累積します。 

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

ここで演算は座標的に適用されます。つまり、$\mathbf{v}^2$ には $v_i^2$ というエントリがあります。同様に、$\frac{1}{\sqrt{v}}$ にはエントリ $\frac{1}{\sqrt{v_i}}$ があり、$\mathbf{u} \cdot \mathbf{v}$ にはエントリ $u_i v_i$ があります。前のように $\eta$ は学習率で、$\epsilon$ は $0$ で除算されないようにする加法定数です。最後に $\mathbf{s}_0 = \mathbf{0}$ を初期化します。 

運動量の場合と同様に、補助変数を追跡する必要があります。この場合は、座標ごとに個別の学習率を考慮する必要があります。主なコストは通常 $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ とその導関数を計算することであるため、SGD と比較して Adagrad のコストが大幅に増加することはありません。 

$\mathbf{s}_t$ に 2 乗勾配を累積すると、$\mathbf{s}_t$ は基本的に線形速度で成長することを意味することに注意してください (勾配は最初は減少するため、実際の線形よりもやや遅くなります)。これにより $\mathcal{O}(t^{-\frac{1}{2}})$ の学習率が得られますが、座標ごとに調整されます。凸問題ではこれで十分です。ただし、ディープラーニングでは、学習率をゆっくり下げたい場合があります。これにより、後続の章で説明するいくつかのAdagradバリアントが生まれました。とりあえず、二次凸問題でどのように動作するか見てみましょう。以前と同じ問題を使用します。 

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Adagrad は、以前と同じ学習率、つまり $\eta = 0.4$ を使用して実装します。ご覧のとおり、独立変数の反復軌跡はより滑らかです。ただし、$\boldsymbol{s}_t$ の累積効果により、学習率は継続的に減衰するため、反復の後の段階で独立変数はあまり動きません。

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

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

学習率を $2$ に上げると、動作が大幅に改善されます。これは、ノイズのない場合でも学習率の低下がかなり積極的になる可能性があることを示しており、パラメータが適切に収束することを確認する必要があります。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## ゼロからの実装

モーメンタム法と同様に、Adagrad はパラメーターと同じ形状の状態変数を維持する必要があります。

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd` の実験と比較して、モデルの学習にはより大きな学習率を使用します。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 簡潔な実装

アルゴリズム `adagrad` の `Trainer` インスタンスを使用して、Gluon で Adagrad アルゴリズムを呼び出すことができます。

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## [概要

* Adagrad は座標ごとに学習率を動的に下げます。
* グラデーションの大きさは、進行がどれだけ早く達成されるかを調整する手段として使用します。グラデーションの大きい座標は、学習率を小さくして補正します。
* 深層学習の問題では、メモリと計算上の制約があるため、通常、厳密な 2 次導関数を計算することは不可能です。グラデーションは便利なプロキシになります。
* 最適化問題の構造がかなり不均一である場合、Adagrad は歪みを軽減するのに役立ちます。
* アダグラッドは、出現頻度の低い項に対して学習率をよりゆっくり低下させる必要があるスパースな特徴量に対して特に効果的です。
* ディープラーニングの問題では、Adagrad は学習率を下げるのに積極的すぎることがあります。:numref:`sec_adam` のコンテキストで、これを軽減するための戦略について説明します。

## 演習

1. 直交行列 $\mathbf{U}$ とベクトル $\mathbf{c}$ の場合、$\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$ が成り立つことを証明してください。変数が直交的に変化しても摂動の大きさが変化しないのはなぜでしょうか。
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ でアダグラッドを試してみてください。また、目的関数が 45 度回転した、つまり $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ についても試してください。動作が違うのですか？
1. 行列 $\mathbf{M}$ の固有値 $\lambda_i$ が $j$ の少なくとも 1 つの選択肢に対して $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ を満たすことを示す [ゲルシュゴリンの円定理](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) を証明する。
1. ゲルシュゴリンの定理は、対角的に前処理された行列 $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$ の固有値について何を教えてくれますか？
1. ファッション MNIST に適用すると :numref:`sec_lenet` のように、適切なディープネットワークの Adagrad を試してみてください。
1. 学習率の低下のアグレッシブな減衰を実現するには、どのようにAdagradを修正する必要がありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:

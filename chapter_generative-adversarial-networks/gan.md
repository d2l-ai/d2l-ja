# 敵対的生成ネットワーク
:label:`sec_basic_gan`

この本のほとんどを通して、私たちは予測をする方法について話しました。何らかの形で、データ例からラベルへのマッピングを学習したディープニューラルネットワークを使用しました。このような学習は差別的学習と呼ばれ、猫の写真と犬の写真を区別できるようにしたいと考えています。分類器とリグレッサーはどちらも識別学習の例です。そして、バックプロパゲーションによって訓練されたニューラルネットワークは、大規模で複雑なデータセットでの差別的学習について私たちが知っていると思っていたすべてを覆しました。高解像度画像の分類精度は、わずか 5 ～ 6 年で役に立たないものから人間レベルのものへと変化しました (いくつかの注意点があります)。ディープニューラルネットワークが驚くほどうまく機能する、他のすべての差別的なタスクについては、もう一回お話ししません。 

しかし、機械学習には、差別的なタスクを解決するだけでは不十分です。たとえば、ラベルのない大きなデータセットがある場合、このデータの特性を簡潔に捉えるモデルを学習したい場合があります。このようなモデルがあれば、トレーニングデータの分布に似た合成データの例をサンプリングできます。たとえば、顔の写真のコーパスが大きい場合、同じデータセットから取得されたと思われる新しいフォトリアリスティックなイメージを生成したい場合があります。このような学習をジェネレーティブモデリングといいます。 

最近まで、新しいフォトリアリスティックなイメージを合成できる方法はありませんでした。しかし、差別的学習のためのディープニューラルネットワークの成功は、新しい可能性を切り開いた。過去3年間の大きな傾向の1つは、教師あり学習の問題とは一般的に考えられない問題における課題を克服するために、差別的なディープネットを適用することでした。リカレントニューラルネットワーク言語モデルは、識別ネットワーク (次の文字を予測するように学習済み) を使用する例の 1 つです。この識別ネットワークは、一度学習すると生成モデルとして機能します。 

2014年、画期的な論文で敵対的生成ネットワーク (GAN) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`が紹介されました。これは、識別モデルの力を活用して優れた生成モデルを得るための巧妙な新しい方法です。GANの核心は、偽のデータと実際のデータを区別できない場合はデータジェネレーターが優れているという考えに依存しています。統計学では、これを 2 標本検定といいます。これは、データセット $X=\{x_1,\ldots, x_n\}$ と $X'=\{x'_1,\ldots, x'_n\}$ が同じ分布から抽出されたかどうかの質問に答える検定です。ほとんどの統計論文とGANの主な違いは、後者はこの考え方を建設的な方法で使用していることです。つまり、単に「ねえ、これら 2 つのデータセットは同じ分布から派生したとは思えない」と言うようにモデルをトレーニングするのではなく、[two-sample test](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) を使用して生成モデルにトレーニング信号を提供します。これにより、実際のデータに似たものが生成されるまで、データジェネレータを改善できます。少なくとも、分類器をだます必要があります。分類器が最先端のディープニューラルネットワークであっても。 

![Generative Adversarial Networks](../img/gan.svg)
:label:`fig_gan`

GAN アーキテクチャは :numref:`fig_gan` に示されています。ご覧のとおり、GAN アーキテクチャには 2 つの要素があります。まず、実物そっくりのデータを生成できる可能性のあるデバイス (たとえば、ディープネットワークですが、ゲームレンダリングエンジンなど何でもかまいません) が必要です。画像を扱う場合は、画像を生成する必要があります。音声を扱う場合は、音声シーケンスの生成などが必要です。これをジェネレータネットワークと呼びます。2 番目のコンポーネントはディスクリミネーターネットワークです。偽のデータと実際のデータを区別しようとします。両方のネットワークは互いに競合しています。ジェネレータネットワークは、ディスクリミネータネットワークを欺こうとします。この時点で、ディスクリミネーターネットワークは新しいフェイクデータに適応します。この情報は、発電機ネットワークの改善などに使用されます。 

ディスクリミネータは、入力 $x$ が実数 (実数データ) か擬似 (ジェネレータから) かを区別するバイナリ分類器です。通常、弁別器は、隠れサイズ 1 の密層を使用するなど、入力 $\mathbf x$ に対してスカラー予測 $o\in\mathbb R$ を出力し、シグモイド関数を適用して予測確率 $D(\mathbf x) = 1/(1+e^{-o})$ を取得します。実際のデータのラベル $y$ は $1$、フェイクデータには $0$ と仮定します。クロスエントロピー損失（*i.e.*）を最小化するようにディスクリミネーターをトレーニングします。 

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

ジェネレーターでは、まずランダム性のソースからパラメーター $\mathbf z\in\mathbb R^d$ を描画します。*例:*、正規分布 $\mathbf z \sim \mathcal{N} (0, 1)$。$\mathbf z$ を潜在変数と呼びます。その後、$\mathbf x'=G(\mathbf z)$ を生成する関数を適用します。ジェネレータの目標は、弁別器を騙して $\mathbf x'=G(\mathbf z)$ を真のデータ、*つまり*、$D( G(\mathbf z)) \approx 1$ として分類することです。言い換えると、所定の弁別器 $D$ について、$y=0$、*i.e.* のときにクロスエントロピー損失が最大になるように、ジェネレータ $G$ のパラメーターを更新します。 

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

ジェネレータが完璧な動作をする場合、$D(\mathbf x')\approx 1$ は上記の損失を 0 に近づけるため、勾配が小さすぎてディスクリミネータをうまく進めることができません。そのため、通常、次の損失を最小限に抑えます。 

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

これは$\mathbf x'=G(\mathbf z)$をディスクリミネータに送るだけですが、ラベル$y=1$を付けています。 

要約すると、$D$ と $G$ は包括的な目的関数を持つ「ミニマックス」ゲームをプレイしています。 

$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$

GaN アプリケーションの多くは、イメージのコンテキスト内にあります。デモンストレーションの目的として、まずはもっと単純なディストリビューションをフィッティングすることに満足します。GANを使用して、世界で最も非効率なガウス分布のパラメータ推定器を構築するとどうなるかを説明します。さっそく始めよう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## 「実際の」データを生成する

これは世界で最も怠惰な例になるので、単純にガウス分布から引き出されたデータを生成します。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

我々が得たものを見てみましょう。これは、平均 $b$ と共分散行列 $A^TA$ で、ある程度任意の方法でガウスシフトされたものでなければなりません。

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## ジェネレータ

ジェネレータネットワークは、可能な限りシンプルなネットワーク、つまり単層線形モデルになります。これは、ガウスデータジェネレーターを使用してその線形ネットワークを駆動するためです。したがって、文字通り、物事を完全に偽造するためのパラメーターを学習するだけで済みます。

```{.python .input}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## 弁別者

ディスクリミネーターについては、もう少し差別的になります。物事をもう少し面白くするために、3つのレイヤーを持つMLPを使用します。

```{.python .input}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## 訓練

まず、ディスクリミネータを更新する関数を定義します。

```{.python .input}
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) + 
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # Labels corresponding to real data
    zeros = tf.zeros((batch_size,)) # Labels corresponding to fake data
    # Do not need to compute gradient for `net_G`, so it's outside GradientTape
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogitsLoss
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

ジェネレータも同様に更新されます。ここでは、クロスエントロピー損失を再利用しますが、フェイクデータのラベルを $0$ から $1$ に変更します。

```{.python .input}
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        # We multiply the loss by batch_size to match PyTorch's BCEWithLogits loss
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

弁別器と発生器は両方とも、交差エントロピー損失をもつ2値ロジスティック回帰を実行します。トレーニングプロセスをスムーズにするために Adam を使います。各反復で、まずディスクリミネータを更新し、次にジェネレータを更新します。損失と生成された例の両方を視覚化します。

```{.python .input}
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # Visualize generated examples
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])
        
        # Show the losses
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
        
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

ここで、ガウス分布にあてはめるようにハイパーパラメーターを指定します。

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## [概要

* 敵対的生成ネットワーク (GAN) は、ジェネレーターとディスクリミネーターの 2 つのディープネットワークで構成されます。
* ジェネレータは、クロスエントロピー損失 (*i.e.*、$\max \log(D(\mathbf{x'}))$) を最大化することで、ディスクリミネーターをだますために可能な限り真のイメージに近いイメージを生成します。
* ディスクリミネータは、クロスエントロピー損失 (*i.e.*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$) を最小化することによって、生成されたイメージを真のイメージと区別しようとします。

## 演習

* 発生器が勝つところに平衡が存在しますか？つまり、弁別器は有限標本上の2つの分布を区別できなくなるのですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/408)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1082)
:end_tab:

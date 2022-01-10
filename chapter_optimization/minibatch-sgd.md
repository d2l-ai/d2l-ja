# ミニバッチ確率的勾配降下法
:label:`sec_minibatch_sgd`

これまで、勾配ベース学習のアプローチで 2 つの極端に遭遇しました。: :numref:`sec_gd` では、データセット全体を使用して勾配を計算し、パラメーターを一度に 1 パスずつ更新します。逆に、:numref:`sec_sgd` は一度に 1 つの観測値を処理して進行させます。それぞれに独自の欠点があります。グラデーションディセントは、データが非常によく似ている場合は特に「データ効率*」ではありません。確率的勾配降下法は、CPU と GPU がベクトル化の能力を最大限に活用できないため、特に「計算効率的」ではありません。これは、幸せな媒体があるかもしれないことを示唆しています。実際、これまで説明した例ではそれを使用してきました。 

## ベクタ変換とキャッシュ

ミニバッチを使用する決定の中心は、計算効率です。これは、複数の GPU と複数のサーバーへの並列化を検討する場合に最もわかりやすくなります。この場合、各 GPU に少なくとも 1 つのイメージを送信する必要があります。サーバーあたり8 GPU、サーバー16台で、すでに128のミニバッチサイズに達しています。 

シングルGPUやCPUに関しては、状況は少し微妙です。これらのデバイスには複数のタイプのメモリがあり、多くの場合、複数のタイプのコンピュートユニットがあり、デバイス間では帯域幅の制約が異なります。たとえば、CPUには少数のレジスタがあり、L1、L2、場合によってはL3キャッシュ（異なるプロセッサコア間で共有される）もあります。これらのキャッシュのサイズとレイテンシーは増加します (同時に帯域幅も減少します)。プロセッサは、メインメモリインターフェイスが提供できる操作よりもはるかに多くの操作を実行できます。 

* 16 コアと AVX-512 ベクトル化を備えた 2 GHz CPU は、1 秒あたり最大 $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ バイトを処理できます。GPU の能力は、この数を 100 倍簡単に上回ります。一方、ミッドレンジ・サーバ・プロセッサの帯域幅は 100 Gb/ 秒をはるかに超えない場合があります。つまり、プロセッサに電力を供給し続けるのに必要な帯域幅の 10 分の 1 未満です。さらに悪いことに、すべてのメモリアクセスが同等に作成されるわけではありません。まず、メモリー・インターフェースは通常 64 ビット幅またはそれより広い (例えば、GPU で 384 ビットまで) ため、1 バイトを読み取るとアクセスのコストが大きくなります。
* 最初のアクセスにはかなりのオーバーヘッドがありますが、シーケンシャルアクセスは比較的安価です (これはバーストリードと呼ばれることが多い)。複数のソケット、チップレット、その他の構造がある場合のキャッシングなど、留意すべき点は他にもたくさんあります。これについての詳細な説明は、このセクションでは扱いません。より詳細な議論については、この [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) を参照してください。

これらの制約を緩和する方法は、プロセッサーにデータを供給するのに十分な速さを持つ CPU キャッシュの階層を使用することです。これが、ディープラーニングのバッチ処理の原動力です。単純にするために、行列-行列の乗算 ($\mathbf{A} = \mathbf{B}\mathbf{C}$ など) を考えてみましょう。$\mathbf{A}$を計算するためのオプションがいくつかあります。たとえば、以下を試すことができます。 

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$ を計算できます。つまり、ドット積によって要素単位で計算できます。
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$ を計算できます。つまり、一度に 1 列ずつ計算できます。同様に、$\mathbf{A}$ を一度に 1 行ずつ $\mathbf{A}_{i,:}$ を計算することもできます。
1. $\mathbf{A} = \mathbf{B} \mathbf{C}$ を単純に計算できます。
1. $\mathbf{B}$ と $\mathbf{C}$ をより小さなブロック行列に分割し、$\mathbf{A}$ を一度に 1 ブロックずつ計算することができます。

最初のオプションに従うと、要素 $\mathbf{A}_{ij}$ を計算するたびに、1 つの行と 1 つの列ベクトルを CPU にコピーする必要があります。さらに悪いことに、行列要素は順次整列されるため、メモリから読み取るときに、2 つのベクトルのうちの 1 つの多数の分断された場所にアクセスする必要があります。2番目の選択肢ははるかに有利です。その中で、$B$ をトラバースし続ける間、列ベクトル $\mathbf{C}_{:,j}$ を CPU キャッシュに保持することができます。これにより、メモリ帯域幅要件が半分になり、それに応じてアクセスが高速化されます。もちろん、オプション3が最も望ましいです。残念ながら、ほとんどの行列はキャッシュに完全には収まらないかもしれません (これは結局私たちが議論していることです)。ただし、オプション 4 は実用的な代替手段を提供します。行列のブロックをキャッシュに移動して、ローカルで乗算することができます。最適化されたライブラリがこれを処理してくれます。これらの操作が実際にどれほど効率的であるかを見てみましょう。 

Python とディープラーニングフレームワーク自体によってもたらされるオーバーヘッドは、計算効率以外にも相当なものです。コマンドを実行するたびに Python インタプリタは MXNet エンジンにコマンドを送ります。MXNet エンジンは、コマンドをコンピュテーショナルグラフに挿入し、スケジューリング中に処理する必要があります。このようなオーバーヘッドは非常に有害です。つまり、可能な限りベクトル化 (および行列) を使用することを強くお勧めします。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np

timer = d2l.Timer()
A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

timer = d2l.Timer()
A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

要素単位の割り当てでは、$\mathbf{B}$ と $\mathbf{C}$ のすべての行と列に対して単純に反復処理され、値が $\mathbf{A}$ に代入されます。

```{.python .input}
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

より高速な方法は、列単位の代入を実行することです。

```{.python .input}
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

最後に、最も効果的な方法は、すべての操作を1つのブロックで実行することです。それぞれの操作の速度を見てみましょう。

```{.python .input}
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## ミニバッチ

:label:`sec_minibatches` 

以前は、パラメータを更新するために、単一の観測値ではなく、*ミニバッチ*のデータを読み取るのは当たり前のことでした。ここで、その理由を簡単に説明します。単一の観測値を処理するには、単一の行列-ベクトル (またはベクトル-ベクトル) の乗算を多数実行する必要があります。これは非常にコストがかかり、基盤となるディープラーニングフレームワークの代わりに大きなオーバーヘッドが発生します。これは、データに適用したときのネットワークの評価 (推論とも呼ばれる) と、勾配を計算してパラメーターを更新するときの両方に適用されます。つまり、これは $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ を実行するたびに当てはまります。 

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

この演算を一度に 1 つの観測値のミニバッチに適用することで、*計算*効率を高めることができます。つまり、1 つの観測値に対する勾配 $\mathbf{g}_t$ を、小さなバッチの 1 つに置き換えます。 

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

$\mathbf{g}_t$ の統計的特性にどのような影響があるかを見てみましょう。$\mathbf{x}_t$ とミニバッチ $\mathcal{B}_t$ のすべての要素はトレーニングセットからランダムに一様に描画されるため、勾配の期待値は変わりません。一方、分散は大幅に減少します。ミニバッチ勾配は平均化される $b := |\mathcal{B}_t|$ 個の独立した勾配で構成されているため、その標準偏差は $b^{-\frac{1}{2}}$ の係数だけ減少します。これは、更新が完全なグラデーションとより確実に一致することを意味するので、それ自体は良いことです。 

単純にこれは、大きなミニバッチ$\mathcal{B}_t$を選択することが普遍的に望ましいことを示しています。悲しいかな、ある時点の後、計算コストの線形増加と比較した場合、標準偏差の追加の減少は最小限です。実際には、GPU のメモリに適合しつつ、優れた計算効率を提供できる大きさのミニバッチを選びます。節約を説明するために、いくつかのコードを見てみましょう。その中で同じ行列-行列の乗算を実行しますが、今回は一度に64列の「ミニバッチ」に分割されます。

```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {2 / timer.times[3]:.3f}')
```

ご覧のとおり、ミニバッチでの計算は、基本的に完全行列での計算と同じくらい効率的です。注意の言葉は順調です。:numref:`sec_batch_norm` では、ミニバッチの分散量に大きく依存する正則化を使用しました。後者を大きくすると分散が小さくなり、バッチ正規化によるノイズインジェクションのメリットも得られます。適切な項を再スケーリングして計算する方法の詳細については、:cite:`Ioffe.2017` などを参照してください。 

## データセットの読み取り

データからミニバッチを効率的に生成する方法を見てみましょう。以下では、NASA が開発したデータセットを使用して Wing [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) をテストし、これらの最適化アルゴリズムを比較します。便宜上、最初の $1,500$ の例だけを使います。データは前処理のために白色化されます。つまり、平均を削除し、分散を座標あたり $1$ に再スケーリングします。

```{.python .input}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## ゼロからの実装

:numref:`sec_linear_scratch` のミニバッチ確率的勾配降下法の実装を思い出してください。以下では、もう少し一般的な実装を示します。便宜上、この章の後半で紹介する他の最適化アルゴリズムと同じコールシグネチャを使用します。具体的には、status 入力 `states` を追加し、ハイパーパラメータをディクショナリ `hyperparams` に配置します。さらに、学習関数内の各ミニバッチ例の損失を平均化するため、最適化アルゴリズムの勾配をバッチサイズで割る必要はありません。

```{.python .input}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

次に、この章の後半で紹介する他の最適化アルゴリズムを使用しやすくするために、汎用学習関数を実装します。線形回帰モデルを初期化し、ミニバッチ確率的勾配降下法やその後に導入される他のアルゴリズムでモデルをトレーニングするために使用できます。

```{.python .input}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Train
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

バッチ勾配降下法で最適化がどのように進行するか見てみましょう。これは、ミニバッチサイズを 1500 (例:サンプルの総数) に設定することで実現できます。その結果、モデルパラメータはエポックごとに 1 回だけ更新されます。進歩はほとんどない。実際、6ステップ後に進行が停止します。

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

バッチサイズが 1 の場合、確率的勾配降下法を使用して最適化します。実装を簡単にするために、一定の (小さいながらも) 学習率を選択しました。確率的勾配降下法では、例が処理されるたびにモデルパラメーターが更新されます。私たちの場合、これはエポックあたり1500回の更新に相当します。ご覧のとおり、目的関数の値の低下は 1 エポック後に減速します。どちらの手順も1つのエポックで1500例を処理しましたが、私たちの実験では確率的勾配降下法は勾配降下法よりも多くの時間を消費します。これは、確率的勾配降下法によりパラメーターがより頻繁に更新され、1 つの観測値を一度に 1 つずつ処理するほうが効率が悪いためです。

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

最後に、バッチサイズが 100 の場合、最適化にミニバッチ確率的勾配降下法を使用します。エポック当たりの所要時間は、確率的勾配降下法およびバッチ勾配降下法に要する時間よりも短い。

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

バッチサイズを 10 に減らすと、各バッチのワークロードの実行効率が低下するため、各エポックの時間が長くなります。

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

これで、前の 4 つの実験の時間と損失を比較できます。見て分かるように、確率的勾配降下法は処理されるサンプル数の点ではGDよりも速く収束しますが、例による勾配の計算はそれほど効率的ではないため、GDよりも同じ損失に到達するまでの時間が長くなります。ミニバッチ確率的勾配降下法は収束速度と計算効率をトレードオフすることができます。10 のミニバッチサイズは、確率的勾配降下法よりも効率的です。100 のミニバッチサイズは、ランタイムの点では GD よりも優れています。

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 簡潔な実装

Gluon では、`Trainer` クラスを使用して最適化アルゴリズムを呼び出すことができます。これは一般的なトレーニング関数を実装するために使用されます。この章ではこれを使います。

```{.python .input}
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)

    # Note: `MSELoss` computes squared error without the 1/2 factor
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    # Note: `MeanSquaredError` computes squared error without the 1/2 factor
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                r = (d2l.evaluate_loss(net, data_iter, loss),)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.avg():.3f} sec/epoch')
```

Gluon を使用して最後の実験を繰り返すと、同じ動作が示されます。

```{.python .input}
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.05}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## [概要

* ベクトル化により、ディープラーニングフレームワークに起因するオーバーヘッドが減少し、メモリーの局所性と CPU と GPU でのキャッシュが改善されるため、コードの効率が向上します。
* 確率的勾配降下法から生じる統計的効率と、一度に大量のデータを処理することから生じる計算効率との間にはトレードオフがあります。
* ミニバッチ確率的勾配降下法は、計算効率と統計効率という両方の長所を提供します。
* ミニバッチ確率的勾配降下法では、学習データのランダムな置換によって得られたデータのバッチを処理します (つまり、各観測値はランダムな順序ではあるが、エポックごとに1回だけ処理されます)。
* トレーニング中は学習率を下げることをお勧めします。
* 一般に、クロック時間で測定した場合、ミニバッチ確率的勾配降下法は確率的勾配降下法や勾配降下法よりも速く、収束してリスクを小さくすることができます。

## 演習

1. バッチサイズと学習率を変更し、目的関数の値に対する減少率と各エポックで消費された時間を観察します。
1. MXNet のドキュメントを読み、関数 `Trainer` クラス `set_learning_rate` を使用して、各エポック後にミニバッチ確率的勾配降下法の学習率を以前の値の 1/10 に下げます。
1. ミニバッチの確率的勾配降下法を、トレーニングセットから実際に*置換あり*をサンプリングするバリアントと比較します。何が起きる？
1. 邪悪な魔神は、あなたに告げることなくデータセットを複製します (つまり、各観測が 2 回発生し、データセットは元のサイズの 2 倍に拡大しますが、誰も教えてくれませんでした)。確率的勾配降下法、ミニバッチ確率的勾配降下法、勾配降下法の挙動はどのように変化するのか

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:

# 複数の GPU でのトレーニング
:label:`sec_multi_gpu`

ここまでは、CPU と GPU でモデルを効率的にトレーニングする方法について説明しました。:numref:`sec_auto_para` では、ディープラーニングフレームワークがどのようにして計算と通信を自動的に並列化できるかを示しました。:numref:`sec_use_gpu` では、`nvidia-smi` コマンドを使用して、コンピューターで使用可能なすべての GPU を一覧表示する方法も示しました。私たちが議論しなかったのは、ディープラーニングのトレーニングを実際にどのように並列化するかです。代わりに、データを何らかの形で複数のデバイスに分割して機能させることを暗示しました。このセクションでは、詳細を記入し、ゼロから開始するときにネットワークを並列に学習させる方法を示します。高レベル API の機能を利用する方法の詳細は :numref:`sec_multi_gpu_concise` に委ねられています。:numref:`sec_minibatch_sgd` で説明されているような、ミニバッチ確率的勾配降下法アルゴリズムに精通していることを前提としています。 

## 問題の分割

単純なコンピュータービジョンの問題と、畳み込み、プーリングの複数の層、そして場合によっては最終的にいくつかの完全に接続された層を含む、少し古風なネットワークから始めましょう。つまり、LeNet :cite:`LeCun.Bottou.Bengio.ea.1998`またはAlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`と非常によく似たネットワークから始めましょう。複数の GPU (デスクトップサーバーの場合は 2 個、AWS g4dn.12xlarge インスタンスには 4 個、p3.16xlarge インスタンスには 8 個、p2.16xlarge には 16 個) を考えると、高速化を達成すると同時に、シンプルで再現性のある設計の選択肢からメリットを得られるようにトレーニングを分割する必要があります。結局のところ、複数のGPUは*メモリ*と*計算*の両方の能力を向上させます。簡単に言うと、分類するトレーニングデータのミニバッチを考えると、次の選択肢があります。 

まず、複数の GPU にまたがってネットワークを分割できます。つまり、各 GPU は特定のレイヤーに流入するデータを入力として受け取り、複数の後続のレイヤーにわたってデータを処理し、そのデータを次の GPU に送信します。これにより、単一の GPU で処理できるものと比較して、大規模なネットワークでデータを処理できます。さらに、GPU あたりのメモリフットプリントは適切に制御できます (ネットワークフットプリント全体のほんの一部)。 

ただし、レイヤ (および GPU) 間のインターフェイスには、厳密な同期が必要です。これは特に、レイヤー間で計算ワークロードが適切に一致していない場合は注意が必要です。この問題は、GPU の数が多いほど悪化します。レイヤー間のインターフェースには、アクティベーションやグラデーションなどの大量のデータ転送も必要です。これにより、GPU バスの帯域幅が圧倒される可能性があります。さらに、計算集約的でありながらシーケンシャルな操作は、パーティショニングにとって自明ではありません。この点に関するベストエフォートについては、:cite:`Mirhoseini.Pham.Le.ea.2017` を参照してください。これは依然として困難な問題であり、非自明な問題に対して良好な (線形) スケーリングを実現できるかどうかは不明である。複数の GPU を連鎖させるための優れたフレームワークまたはオペレーティングシステムサポートがない限り、この方法はお勧めしません。 

次に、作業をレイヤーごとに分割できます。たとえば、単一の GPU で 64 チャネルを計算するのではなく、問題を 4 つの GPU に分割し、各 GPU で 16 チャネルのデータを生成することができます。同様に、完全に接続されたレイヤーでは、出力ユニット数を分割できます。:numref:`fig_alexnet_original` (:cite:`Krizhevsky.Sutskever.Hinton.2012` から取得) はこの設計を示しています。この設計では、この戦略を使用して、メモリフットプリントが非常に小さい (当時は 2 GB) の GPU を処理しました。これにより、チャンネル (またはユニット) の数が小さすぎない限り、計算の観点から適切なスケーリングが可能になります。さらに、使用可能なメモリは直線的に増大するため、複数の GPU はますます大規模なネットワークを処理できます。 

![Model parallelism in the original AlexNet design due to limited GPU memory.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

ただし、各レイヤーは他のすべてのレイヤーの結果に依存するため、*非常に多くの * 数の同期またはバリア操作が必要です。さらに、GPU 間でレイヤーを分散する場合よりも、転送する必要があるデータ量がさらに大きくなる可能性があります。そのため、帯域幅のコストと複雑さから、この方法はお勧めしません。 

最後に、複数の GPU にデータを分割できます。このようにして、観察結果は異なりますが、すべてのGPUが同じタイプの作業を実行します。勾配は、学習データの各ミニバッチの後に GPU 全体で集計されます。これは最も単純なアプローチであり、どのような状況でも適用できます。同期する必要があるのは、各ミニバッチの後だけです。とはいえ、グラデーションパラメータの交換は、他のパラメータがまだ計算されている間に開始することが非常に望ましいと言えます。さらに、GPU の数が多いほどミニバッチのサイズが大きくなり、トレーニングの効率が向上します。ただし、GPU を追加しても、より大きなモデルのトレーニングはできません。 

![Parallelization on multiple GPUs. From left to right: original problem, network partitioning, layerwise partitioning, data parallelism.](../img/splitting.svg)
:label:`fig_splitting`

:numref:`fig_splitting` には、複数の GPU でのさまざまな並列化方法の比較が示されています。概して、十分に大きなメモリを持つGPUにアクセスできるのであれば、データ並列処理が最も便利な方法です。分散学習のパーティショニングの詳細については、:cite:`Li.Andersen.Park.ea.2014` も参照してください。ディープラーニングの初期には、GPU メモリが問題になっていました。この問題は、最も珍しいケースを除くすべてのケースで解決されました。以下では、データの並列処理に注目します。 

## データ並列処理

マシンに $k$ GPU があると仮定します。トレーニングするモデルがある場合、各 GPU はモデルパラメーターの完全なセットを個別に維持しますが、GPU 全体のパラメーター値は同一で同期されます。例として、:numref:`fig_data_parallel` は $k=2$ の場合のデータ並列処理による学習を示しています。 

![Calculation of minibatch stochastic gradient descent using data parallelism on two GPUs.](../img/data-parallel.svg)
:label:`fig_data_parallel`

一般に、トレーニングは次のように進行します。 

* ランダムなミニバッチがある場合の学習の反復では、バッチ内の例を $k$ の部分に分割し、GPU 全体に均等に分散します。
* 各 GPU は、割り当てられたミニバッチサブセットに基づいて、モデルパラメーターの損失と勾配を計算します。
* 各 $k$ GPU のローカル勾配が集約され、現在のミニバッチ確率勾配が得られます。
* 集約勾配は各 GPU に再配分されます。
* 各 GPU は、このミニバッチの確率的勾配を使用して、保持するモデルパラメーターの完全なセットを更新します。

実際には、$k$ GPU でトレーニングする場合は、各 GPU が 1 つの GPU のみでトレーニングする場合と同じ量の作業を行うように、ミニバッチサイズを $k$ 倍に*増やします*。16 GPU サーバーでは、これによってミニバッチのサイズが大幅に増加する可能性があり、それに応じて学習率を上げる必要があります。また、:numref:`sec_batch_norm` のバッチ正規化は、GPU ごとに個別のバッチ正規化係数を保持するなどして調整する必要があることにも注意してください。以下では、おもちゃのネットワークを使ってマルチ GPU トレーニングを説明します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**トイネットワーク**]

:numref:`sec_lenet` で導入された LeNet を使用しています (若干の変更があります)。パラメーターの交換と同期を詳細に説明するために、これをゼロから定義します。

```{.python .input}
# Initialize model parameters
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# Initialize model parameters
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# Define the model
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Cross-entropy loss function
loss = nn.CrossEntropyLoss(reduction='none')
```

## データ同期

マルチ GPU トレーニングを効率的に行うには、2 つの基本操作が必要です。まず、[**パラメータのリストを複数のデバイスに配布する**] とグラデーション (`get_params`) を付ける機能が必要です。パラメーターがないと、GPU でネットワークを評価することは不可能です。次に、複数のデバイス間でパラメーターを合計する機能が必要です。つまり、`allreduce` 関数が必要です。

```{.python .input}
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

モデルパラメータを 1 つの GPU にコピーして試してみましょう。

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

まだ計算を行っていないので、バイアスパラメータに関する勾配はゼロのままです。ここで、ベクトルが複数の GPU に分散されていると仮定します。次の [**`allreduce` 関数は、すべてのベクトルを加算し、その結果をすべての GPU にブロードキャストします**]。これを機能させるには、データをデバイスにコピーして結果を蓄積する必要があることに注意してください。

```{.python .input}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

異なるデバイス上で異なる値を持つベクトルを作成し、それらを集約して、これをテストしてみましょう。

```{.python .input}
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## データの配布

[**ミニバッチを複数の GPU に均等に分散させる**] には、単純なユーティリティ関数が必要です。たとえば、2 つの GPU で、データの半分をどちらかの GPU にコピーしたいとします。より便利で簡潔なので、ディープラーニングフレームワークの組み込み関数を使用して $4 \times 5$ 行列で試してみます。

```{.python .input}
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

後で再利用するために、データとラベルの両方を分割する `split_batch` 関数を定義します。

```{.python .input}
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 訓練

これで、[**単一のミニバッチでマルチ GPU トレーニング**] を実装できます。その実装は、主にこのセクションで説明するデータ並列処理アプローチに基づいています。先ほど説明した補助関数 `allreduce` と `split_and_load` を使用して、複数の GPU 間でデータを同期します。並列処理を実現するために特定のコードを記述する必要はないことに注意してください。計算グラフはミニバッチ内のデバイス間の依存関係を持たないため、*自動的に*並列に実行されます。

```{.python .input}
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

これで、[**トレーニング関数**] を定義できます。前の章で使用したものとは少し異なります。GPUを割り当て、すべてのモデルパラメーターをすべてのデバイスにコピーする必要があります。明らかに、各バッチは `train_batch` 関数を使用して処理され、複数の GPU を処理します。利便性 (およびコードの簡潔さ) のために、単一の GPU で精度を計算しますが、他の GPU はアイドル状態なので、*非効率* です。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

これがどの程度うまく機能するか見てみましょう [**単一の GPU で**]。まず、バッチサイズを 256、学習率 0.2 を使用します。

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

バッチサイズと学習率を変更せず、[**GPU の数を 2 に増やす**] ことで、テストの精度は前の実験とほぼ同じままであることがわかります。最適化アルゴリズムに関しては、これらは同一です。残念ながら、ここで得られる意味のある高速化はありません。モデルは単純に小さすぎます。さらに、小さなデータセットしかなく、マルチ GPU トレーニングを実装するためのやや洗練されていないアプローチで、Python のオーバーヘッドが大きくなっていました。今後、より複雑なモデルとより洗練された並列化方法に遭遇するでしょう。それにもかかわらずFashion-MNISTに何が起こるか見てみましょう。

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## [概要

* ディープネットワーク学習を複数の GPU に分割する方法は複数あります。レイヤー間、レイヤー間、またはデータ間で分割できます。前者の2つは、厳密に振り付けられたデータ転送を必要とします。データ並列処理は最も単純な戦略です。
* データ並列学習は簡単です。ただし、効率を上げるために有効なミニバッチサイズが大きくなります。
* データ並列処理では、データは複数の GPU に分割され、各 GPU が独自の順方向および逆方向操作を実行し、続いて勾配が集約され、結果が GPU にブロードキャストされます。
* 大きいミニバッチには、学習率を少し上げて使用することがあります。

## 演習

1. $k$ GPU で学習する場合は、ミニバッチサイズを $b$ から $k \cdot b$ に変更します。つまり、GPU の数だけ拡大します。
1. 異なる学習率の精度を比較します。GPUの数に応じてどのようにスケーリングするのですか？
1. 異なるGPUで異なるパラメータを集約する、より効率的な`allreduce`関数を実装しますか?なぜより効率的ですか？
1. マルチ GPU テスト精度の計算を実装します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab:

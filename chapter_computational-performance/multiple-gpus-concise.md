# 複数の GPU の簡潔な実装
:label:`sec_multi_gpu_concise`

新しいモデルごとに並列処理をゼロから実装するのは楽しいことではありません。さらに、同期ツールを最適化してパフォーマンスを向上させることには、大きなメリットがあります。以下では、ディープラーニングフレームワークの高レベル API を使用してこれを行う方法を示します。数学とアルゴリズムは :numref:`sec_multi_gpu` と同じです。当然のことながら、このセクションのコードを実行するには、少なくとも 2 つの GPU が必要です。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**トイネットワーク**]

:numref:`sec_multi_gpu` の LeNet よりもやや意味のあるネットワークを使用してみましょう。このネットワークは、まだ十分に簡単かつ迅速に学習できます。私たちはResNet-18バリアント:cite:`He.Zhang.Ren.ea.2016`を選びます。入力画像は小さいので、少し修正します。特に :numref:`sec_resnet` との違いは、最初に小さい畳み込みカーネル、ストライド、パディングを使用する点です。さらに、最大プーリング層を削除します。

```{.python .input}
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## ネットワーク初期化

:begin_tab:`mxnet`
`initialize` 関数を使用すると、任意のデバイスでパラメーターを初期化できます。初期化メソッドの復習については、:numref:`sec_numerical_stability` を参照してください。特に便利なのは、*複数*のデバイスで同時にネットワークを初期化できることです。これが実際にどのように機能するのか試してみましょう。
:end_tab:

:begin_tab:`pytorch`
学習ループ内でネットワークを初期化します。初期化メソッドの復習については、:numref:`sec_numerical_stability` を参照してください。
:end_tab:

```{.python .input}
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu` で導入された `split_and_load` 関数を使用して、データのミニバッチを分割し、`devices` 変数によって提供されるデバイスのリストにその部分をコピーできます。ネットワークインスタンスは、適切な GPU を*自動的に* 使用して、順伝播の値を計算します。ここでは、4 つのオブザベーションを生成し、GPU に分割します。
:end_tab:

```{.python .input}
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
データがネットワークを通過すると、対応するパラメータが*データが通過したデバイス上で初期化されます*。つまり、初期化はデバイス単位で行われます。初期化に GPU 0 と GPU 1 を選択したため、ネットワークはそこでのみ初期化され、CPU 上では初期化されません。実際、パラメータはCPU上には存在しません。これを確認するには、パラメーターを出力し、発生する可能性のあるエラーを観察します。
:end_tab:

```{.python .input}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
次に、[**精度を評価する**] のコードを、動作するコード (**複数のデバイスで並行して**) に置き換えてみましょう。これは :numref:`sec_lenet` の `evaluate_accuracy_gpu` 関数に代わるものです。主な違いは、ネットワークを起動する前にミニバッチを分割することです。他のすべては本質的に同じです。
:end_tab:

```{.python .input}
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**トレーニング**]

前述のように、効率的な並列処理を行うには、トレーニングコードでいくつかの基本関数を実行する必要があります。 

* ネットワークパラメータは、すべてのデバイスで初期化する必要があります。
* データセットを反復処理する間、ミニバッチはすべてのデバイスに分割されます。
* 損失とその勾配をデバイス間で並列に計算します。
* グラデーションが集約され、それに応じてパラメータが更新されます。

最後に、ネットワークの最終的なパフォーマンスを報告するために、精度を (再び並行して) 計算します。トレーニングルーチンは前の章の実装とよく似ていますが、データを分割して集約する必要がある点が異なります。

```{.python .input}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

これが実際にどのように機能するか見てみましょう。ウォームアップとして [**単一の GPU でネットワークをトレーニングする**]

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

次に [**トレーニングには2つのGPUを使用する**]。:numref:`sec_multi_gpu` で評価された LeNet と比較すると、ResNet-18 のモデルはかなり複雑です。並列化が利点を示すのはこの点です。計算にかかる時間は、パラメーターの同期にかかる時間よりもかなり長くなります。これにより、並列化のオーバーヘッドが少なくなるため、スケーラビリティが向上します。

```{.python .input}
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## [概要

:begin_tab:`mxnet`
* Gluon は、コンテキストリストを提供することで、複数のデバイスにわたってモデルを初期化するためのプリミティブを提供します。
:end_tab:

* データは、データが検出されたデバイス上で自動的に評価されます。
* 各デバイスのパラメータにアクセスする前に、そのデバイスのネットワークを初期化してください。そうしないと、エラーが発生します。
* 最適化アルゴリズムは、複数の GPU にわたって自動的に集約されます。

## 演習

:begin_tab:`mxnet`
1. このセクションでは ResNet-18 を使用します。エポック、バッチサイズ、学習率を変えてみてください。計算にはより多くの GPU を使用します。16 個の GPU でこれを試した場合 (AWS p2.16xlarge インスタンスなど)、どうなりますか?
1. デバイスが異なれば、計算能力も異なる場合があります。GPUとCPUを同時に使用できました。仕事をどのように分けるべきか？努力する価値はありますか？なぜ？どうして？
1. `npx.waitall()`を落としたらどうなるの？並列処理のために最大2つのステップが重なるようにトレーニングをどのように修正しますか？
:end_tab:

:begin_tab:`pytorch`
1. このセクションでは ResNet-18 を使用します。エポック、バッチサイズ、学習率を変えてみてください。計算にはより多くの GPU を使用します。16 個の GPU でこれを試した場合 (AWS p2.16xlarge インスタンスなど)、どうなりますか?
1. デバイスが異なれば、計算能力も異なる場合があります。GPUとCPUを同時に使用できました。仕事をどのように分けるべきか？努力する価値はありますか？なぜ？どうして？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:

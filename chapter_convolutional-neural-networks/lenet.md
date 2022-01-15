# 畳み込みニューラルネットワーク (LeNet)
:label:`sec_lenet`

これで、完全に機能するCNNを組み立てるのに必要な材料がすべて揃いました。画像データとの以前の出会いでは、Fashion-MNIST データセット内の衣類の写真にソフトマックス回帰モデル (:numref:`sec_softmax_scratch`) と MLP モデル (:numref:`sec_mlp_scratch`) を適用しました。このようなデータをソフトマックス回帰と MLP に対応させるために、まず $28\times28$ 行列の各イメージを固定長の $784$ 次元ベクトルに平坦化し、その後、全結合層で処理しました。畳み込み層のハンドルができたので、イメージ内の空間構造を保持できます。全結合層を畳み込み層に置き換えることの追加の利点として、必要なパラメーターがはるかに少なくて済む、より簡潔なモデルが得られます。 

このセクションでは、コンピュータビジョンのタスクに対する性能が広く注目されている、最初に公開されたCNNのうち、*LeNet*を紹介します。このモデルは、画像 :cite:`LeCun.Bottou.Bengio.ea.1998` の手書き数字を認識する目的で、当時AT&T Bell LeCunの研究員だったYann LeCunによって紹介され、名前が付けられました。この研究は、この技術を開発する10年にわたる研究の集大成を表しています。1989年、LeCunはバックプロパゲーションによるCNNのトレーニングに成功した最初の研究を発表しました。 

当時、LeNetはサポートベクターマシンのパフォーマンスに匹敵する優れた結果を達成しました。これは、教師あり学習における主要なアプローチでした。LeNetは最終的に、ATMマシンで預金を処理するための数字を認識するように適合されました。今日まで、一部のATMは、1990年代にYannと彼の同僚Leon Bottouが書いたコードを実行しています。 

## LeNet

大まかに言うと、(**LeNet (LeNet-5) は、(i) 2 つの畳み込み層からなる畳み込み符号化器と (ii) 3 つの完全結合層からなる高密度ブロック の 2 つの部分から構成されます**)。アーキテクチャは :numref:`img_lenet` にまとめられています。 

![Data flow in LeNet. The input is a handwritten digit, the output a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

各畳み込みブロックの基本単位は、畳み込み層、シグモイド活性化関数、それに続く平均プーリング演算です。RelU と max-pooling はうまく機能しますが、これらの発見は 1990 年代にはまだ行われていなかったことに注意してください。各畳み込み層は $5\times 5$ カーネルとシグモイド活性化関数を使用します。これらのレイヤーは、空間的に配置された入力を多数の 2 次元フィーチャマップにマッピングし、通常はチャネル数を増やします。最初の畳み込み層には 6 つの出力チャネルがあり、2 番目の畳み込み層には 16 個の出力チャネルがあります。$2\times2$ プーリング演算 (ストライド 2) ごとに、空間的ダウンサンプリングによって次元が $4$ 倍減少します。畳み込みブロックは、(バッチサイズ、チャネル数、高さ、幅) で指定された形状を持つ出力を出力します。 

畳み込みブロックからの出力を密ブロックに渡すには、ミニバッチ内の各例を平坦化しなければなりません。言い換えると、この 4 次元の入力を受け取り、全結合層が期待する 2 次元の入力に変換します。念のため、望む 2 次元表現では、1 番目の次元を使用してミニバッチの例をインデックス化し、2 番目の次元をフラットベクトル表現に使用します。それぞれの例のLeNet の高密度ブロックには、それぞれ 120、84、10 の出力を持つ 3 つの完全に接続されたレイヤーがあります。まだ分類を行っているので、10 次元の出力層は可能な出力クラスの数に対応します。 

LeNetの内部で何が起こっているのかを真に理解するには少し手間がかかったかもしれませんが、次のコードスニペットで、このようなモデルを最新のディープラーニングフレームワークで実装するのは非常に簡単であることを納得させることができれば幸いです。`Sequential` ブロックをインスタンス化し、適切なレイヤーをチェーン化するだけで済みます。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # `Dense` will transform an input of the shape (batch size, number of
        # channels, height, width) into an input of the shape (batch size,
        # number of channels * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

元のモデルでは少し自由を取り、最終層のガウス活性化を取り除きました。それ以外は、このネットワークは元の LeNet-5 アーキテクチャと一致します。 

単一チャンネル (白黒) の $28 \times 28$ イメージをネットワーク経由で渡し、各レイヤーで出力形状を印刷することで、[**モデルを検査**] して、その操作が :numref:`img_lenet_vert` から期待されるものと一致していることを確認できます。 

![Compressed notation for LeNet-5.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

畳み込みブロック全体の各層での表現の高さと幅は (前の層と比較して) 小さくなっていることに注意してください。最初の畳み込み層は 2 ピクセルのパディングを使用して、$5 \times 5$ カーネルの使用による高さと幅の減少を補正します。一方、2 番目の畳み込みレイヤーではパディングが行われないため、高さと幅は両方とも 4 ピクセル縮小されます。層のスタックが上がるにつれて、チャネル数はレイヤーオーバーレイヤーの入力の 1 から 1 番目の畳み込み層のあと 6、2 番目の畳み込み層のあと 16 に増えます。ただし、各プーリング層の高さと幅は半分になります。最後に、全結合層ごとに次元が減少し、最終的にクラス数に一致する次元の出力が出力されます。 

## 訓練

モデルを実装したので、[**実験を実行して、LeNetがFashion-MNISTでどのように機能するかを確認する**]。

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

CNN のパラメーターは少なくなりますが、各パラメーターがより多くの乗算に関与するため、同様に深い MLP よりも計算コストが高くなる可能性があります。GPU にアクセスできる場合は、トレーニングを高速化するために GPU を実行に移すのに適したタイミングかもしれません。

:begin_tab:`mxnet, pytorch`
評価のためには、:numref:`sec_softmax_scratch` で説明した [**`evaluate_accuracy` 関数に若干の変更を加える**] 必要があります。データセット全体がメインメモリにあるため、モデルが GPU を使用してデータセットを計算する前に、データセットを GPU メモリにコピーする必要があります。
:end_tab:

```{.python .input}
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if not device:  # Query the first device where the first parameter is on
        device = list(net.collect_params().values())[0].list_ctx()[0]
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

また、[**GPU に対応するようにトレーニング関数を更新する**] :numref:`sec_softmax_scratch` で定義されている `train_epoch_ch3` とは異なり、順方向および逆方向の伝播を行う前に、データの各ミニバッチを指定のデバイス (できれば GPU) に移動する必要があります。 

トレーニング関数 `train_ch6` も :numref:`sec_softmax_scratch` で定義されている `train_ch3` と似ています。今後は多数のレイヤーを持つネットワークを実装する予定なので、主に高レベル API に依存することになります。次のトレーニング関数は、高レベル API から作成されたモデルを入力として想定し、それに応じて最適化されています。:numref:`subsec_xavier` で紹介された Xavier 初期化を使用して、`device` 引数で示されるデバイス上のモデルパラメーターを初期化します。MLPと同様に、損失関数はクロスエントロピーであり、ミニバッチ確率的勾配降下法によって損失関数を最小化します。各エポックの実行には数十秒かかるため、学習損失をより頻繁に視覚化します。

```{.python .input}
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the major difference from `d2l.train_epoch_ch3`
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

```{.python .input}
#@tab tensorflow
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """A callback to visiualize the training progress."""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6)."""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
```

[**それでは、LeNet-5 モデルをトレーニングして評価しましょう**]

```{.python .input}
#@tab all
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* CNN は畳み込み層を使用するネットワークです。
* CNN では、畳み込み、非線形性、および (しばしば) プーリング演算をインターリーブします。
* CNN では、畳み込み層は通常、表現の空間分解能を徐々に低下させながらチャネル数を増やすように配置されます。
* 従来のCNNでは、畳み込みブロックによって符号化された表現は、出力を出力する前に、1つ以上の全結合層によって処理されます。
* LeNetは、間違いなく、このようなネットワークの展開に成功した最初の企業です。

## 演習

1. 平均プーリングを最大プーリングに置き換えます。何が起きる？
1. LeNetをベースに、より複雑なネットワークを構築し、その精度を向上させよう。
    1. 畳み込みウィンドウのサイズを調整します。
    1. 出力チャンネル数を調整します。
    1. アクティベーション機能 (ReLU など) を調整します。
    1. 畳み込み層の数を調整します。
    1. 完全に接続されたレイヤーの数を調整します。
    1. 学習率やその他の学習の詳細 (初期化やエポック数など) を調整します。
1. 元の MNIST データセットで改善されたネットワークを試してみてください。
1. 異なる入力 (セーターやコートなど) に対する LeNet の第 1 レイヤーと第 2 レイヤーのアクティベーションを表示します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:

# 高密度に接続されたネットワーク (DenseNet)

ResNet は、ディープネットワークで関数をパラメーター化する方法の見方を大きく変えました。*DenseNet* (高密度畳み込みネットワーク) は、この :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` をある程度論理的に拡張したものです。それにたどり着く方法を理解するために、数学に少し回り道をしましょう。 

## ResNetからDenseNetへ

関数のテイラー展開を思い出してください。ポイント$x = 0$については、次のように書くことができます。 

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

重要なのは、関数を次第に高次の項に分解することです。同様に、ResNetは関数を次のように分解します。 

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

つまり、ResNet は $f$ を単純な線形項とより複雑な非線形項に分解します。2 つの用語を超える情報を取り込む (必ずしも追加する必要はない) 場合はどうなるでしょうか。一つの解決策はDenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`でした。 

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

:numref:`fig_densenet_block` に示されているように、ResNet と DenseNet の主な違いは、後者の場合、出力は加算されるのではなく、*連結* ($[,]$ で表される) になることです。その結果、ますます複雑になる一連の関数を適用した後に、$\mathbf{x}$ からその値へのマッピングを実行します。 

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

最終的には、これらすべての機能がMLPに組み合わされ、フィーチャの数が再び削減されます。実装の面では、これは非常に単純です。用語を追加するのではなく、連結します。DenseNet という名前は、変数間のディペンデンシーグラフが非常に密集しているという事実に由来しています。このようなチェーンの最後のレイヤーは、以前のすべてのレイヤーに密に接続されています。密な接続は :numref:`fig_densenet` に示されています。 

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`

DenseNetを構成する主なコンポーネントは、*denseBlocks* と*トランジションレイヤ*です。前者は入力と出力の連結方法を定義し、後者はチャンネル数が大きすぎないように制御します。 

## [**高密度ブロック**]

DenseNet は ResNet の修正された「バッチ正規化、アクティベーション、畳み込み」構造を使用します (:numref:`sec_resnet` の演習を参照)。まず、この畳み込みブロック構造を実装します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

*dense ブロック* は、それぞれ同じ数の出力チャネルを使用する複数の畳み込みブロックで構成されます。ただし、順伝播では、各畳み込みブロックの入力と出力をチャネル次元で連結します。

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

次の例では、10 個の出力チャンネルから成る 2 つの畳み込みブロックをもつ [**`DenseBlock` インスタンスを定義**] しています。3 チャンネルの入力を使用すると、$3+2\times 10=23$ チャンネルの出力が得られます。畳み込みブロックチャネルの数は、入力チャネル数に対する出力チャネル数の増加を制御します。これは*成長率*とも呼ばれます。

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## [**トランジションレイヤ**]

高密度ブロックごとにチャネル数が増えるため、追加しすぎるとモデルが過度に複雑になります。*トランジションレイヤ* は、モデルの複雑さを制御するために使用されます。畳み込み層 $1\times 1$ を使用することでチャネル数が削減され、平均プーリング層の高さと幅がストライド 2 で半分になり、モデルの複雑さがさらに軽減されます。

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

[**トランジションレイヤーを適用**] 前の例の Dense ブロックの出力に 10 チャンネルあります。これにより、出力チャンネル数が 10 に減り、高さと幅が半分になります。

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## [**DenseNet モデル**]

次に、DenseNet モデルを構築します。DenseNetはまず、ResNetと同じ単一の畳み込み層と最大プーリング層を使用します。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

その後、ResNetが使用する残留ブロックで構成される4つのモジュールと同様に、DenseNetは4つの高密度ブロックを使用します。ResNet と同様に、各密ブロックで使用する畳み込み層の数を設定できます。ここでは、:numref:`sec_resnet` の ResNet-18 モデルと一致する 4 に設定しました。さらに、Dense ブロック内の畳み込み層のチャネル数 (つまり、成長率) を 32 に設定して、各密ブロックに 128 個のチャネルが追加されるようにします。 

ResNet では、各モジュール間の高さと幅は、ストライド 2 の残差ブロック分だけ縮小されます。ここでは、トランジションレイヤーを使用して、高さと幅を半分にし、チャンネル数を半分にします。

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

ResNet と同様に、グローバルプーリング層と全結合層が最後に接続され、出力が生成されます。

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## [**トレーニング**]

ここではより深いネットワークを使用しているため、このセクションでは入力の高さと幅を 224 から 96 に減らして計算を簡略化します。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* クロスレイヤ接続に関しては、入力と出力が加算されるResNetとは異なり、DenseNetは入力と出力をチャネル次元で連結します。
* DenseNetを構成する主なコンポーネントは、高密度ブロックと遷移層です。
* ネットワークを構成する際には、チャネル数を再び縮小する遷移層を追加して、次元性を制御する必要があります。

## 演習

1. 遷移層で最大プーリングではなく平均プーリングを使用するのはなぜですか？
1. DenseNetの論文で述べた利点の1つは、モデルパラメータがResNetのものより小さいことです。なぜそうなのですか？
1. DenseNetが批判されている問題の1つは、メモリ消費量が多いことです。
    1. これは本当にそうですか？入力形状を $224\times 224$ に変更して、実際の GPU メモリ消費量を確認します。
    1. メモリ消費量を削減する代替手段を考えられますか？フレームワークをどのように変更する必要がありますか？
1. DenseNet ペーパー :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` の表 1 に示されている DenseNet のさまざまなバージョンを実装してください。
1. DenseNet のアイデアを適用して、MLP ベースのモデルを設計します。:numref:`sec_kaggle_house` の住宅価格予測タスクに適用します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:

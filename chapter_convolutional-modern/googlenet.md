# 並列連結を使用するネットワーク (GoogleNet)
:label:`sec_googlenet`

2014年、*GoogleNet*はImageNet Challengeで優勝し、NiNの強みと繰り返しブロック:cite:`Szegedy.Liu.Jia.ea.2015`のパラダイムを組み合わせた構造を提案しました。この論文の焦点の1つは、どのサイズの畳み込みカーネルが最適かという問題に対処することでした。結局のところ、以前の一般的なネットワークでは、$1 \times 1$ と $11 \times 11$ という大きさの選択肢が採用されていました。この論文の1つの見識は、さまざまなサイズのカーネルを組み合わせて使用すると有利になる場合があるということでした。このセクションでは GoogleNet を紹介し、元のモデルを少し簡略化したバージョンです。トレーニングを安定させるために追加されたアドホック機能はいくつか省略していますが、より優れたトレーニングアルゴリズムが利用可能になったため、現在は不要になっています。 

## (**インセプションブロック**)

GoogleNetの基本的な畳み込みブロックは*Inceptionブロック*と呼ばれ、バイラルミームを開始した映画「*Inception*」（「もっと深く行く必要がある」）からの引用にちなんで名付けられました。 

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

:numref:`fig_inception` に示すように、インセプションブロックは 4 つのパラレルパスで構成されます。最初の 3 つのパスでは、ウィンドウサイズが $1\times 1$、$3\times 3$、$5\times 5$ の畳み込み層を使用して、さまざまな空間サイズから情報を抽出します。真ん中の 2 つのパスは入力に対して $1\times 1$ 畳み込みを実行してチャネル数を減らし、モデルの複雑さを軽減します。4 番目のパスでは $3\times 3$ の最大プーリング層を使用し、その後に $1\times 1$ 畳み込み層を使用してチャネル数を変更します。4 つのパスはすべて、入力と出力に同じ高さと幅を与えるために適切なパディングを使用します。最後に、各パスに沿った出力はチャネル次元に沿って連結され、ブロックの出力を構成します。Inception ブロックでよく調整されるハイパーパラメーターは、レイヤーごとの出力チャンネル数です。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')


    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

このネットワークがなぜうまく機能するのかを直感的に理解するために、フィルタの組み合わせを考えてみましょう。さまざまなフィルターサイズでイメージを探索します。つまり、サイズの異なるフィルタによって、さまざまな範囲の詳細を効率的に認識できます。同時に、フィルターごとに異なる量のパラメーターを割り当てることができます。 

## [**GoogleNet モデル**]

:numref:`fig_inception_full` に示されているように、GoogleNet は合計 9 個のインセプションブロックとグローバル平均プーリングのスタックを使用して推定値を生成します。インセプションブロック間で最大プーリングを行うと、次元が減少します。最初のモジュールはAlexNetとLeNetに似ています。ブロックのスタックは VGG から継承され、グローバル平均プーリングにより、最後に完全に接続されたレイヤーのスタックが回避されます。 

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

これで、GoogleNetを1つずつ実装できるようになりました。最初のモジュールは 64 チャネル $7\times 7$ 畳み込み層を使用します。

```{.python .input}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

2 番目のモジュールは 2 つの畳み込み層を使用します。1 つ目は 64 チャネルの $1\times 1$ 畳み込み層、次に $3\times 3$ 畳み込み層でチャネル数を 3 倍にします。これは Inception ブロックの 2 番目のパスに相当します。

```{.python .input}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
       nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

3 番目のモジュールは、2 つの完全な Inception ブロックを直列に接続します。最初の Inception ブロックの出力チャンネル数は $64+128+32+32=256$ で、4 つのパスの出力チャンネル数比率は $64:128:32:32=2:4:1:1$ です。2 番目と 3 番目のパスは、最初に入力チャネル数をそれぞれ $96/192=1/2$ と $16/192=1/12$ に減らし、次に 2 番目の畳み込み層を接続します。2 番目の Inception ブロックの出力チャネル数は $128+192+96+64=480$ に増加し、4 つのパスの出力チャネル数比は $128:192:96:64 = 4:6:3:2$ です。2 番目と 3 番目のパスは、まず入力チャンネル数をそれぞれ $128/256=1/2$ と $32/256=1/8$ に減らします。

```{.python .input}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

4 番目のモジュールはもっと複雑です。5つのインセプションブロックを直列に接続し、$192+208+48+64=512$、$160+224+64+64=512$、$128+256+64+64=512$、$112+288+64+64=528$、$256+320+128+128=832$ の出力チャンネルをそれぞれ備えています。これらのパスに割り当てられるチャネル数は、3 番目のモジュールの場合と同様です。畳み込み層が $3\times 3$ の 2 番目のパスが最も多くのチャネルを出力し、続いて $1\times 1$ 畳み込み層のみを持つ最初のパス、$5\times 5$ の畳み込み層を持つ 3 番目のパスが続きます。$3\times 3$ 最大プーリング層を持つ 4 番目のパス。2 番目と 3 番目のパスは、最初に比率に従ってチャンネル数を減らします。これらの比率は、Inceptionブロックによって多少異なります。

```{.python .input}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

5 番目のモジュールには、$256+320+128+128=832$ と $384+384+128+128=1024$ の出力チャンネルを持つ 2 つのインセプションブロックがあります。各パスに割り当てられるチャネル数は、3 番目と 4 番目のモジュールと同じですが、特定の値が異なります。5番目のブロックの後に出力層が続くことに注意してください。このブロックは、グローバル平均プーリング層を使用して、NiN の場合と同様に、各チャネルの高さと幅を 1 に変更します。最後に、出力を 2 次元配列に変換し、その後にラベルクラスの数と同じ出力数をもつ完全接続レイヤーを作成します。

```{.python .input}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

```{.python .input}
#@tab pytorch
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```{.python .input}
#@tab tensorflow
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
```

GoogleNet モデルは計算が複雑なため、VGG ほどチャネル数を変更するのは簡単ではありません。[**Fashion-MNIST で妥当なトレーニング時間を確保するために、入力の高さと幅を 224 から 96 に減らします。**] これにより計算が簡単になります。さまざまなモジュール間での出力の形状の変化を以下に示します。

```{.python .input}
X = np.random.uniform(size=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## [**トレーニング**]

前と同じように、Fashion-MNIST データセットを使用してモデルをトレーニングします。トレーニングプロシージャを呼び出す前に、解像度を $96 \times 96$ ピクセルに変換します。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* Inception ブロックは、4 つのパスを持つサブネットワークに相当します。異なるウィンドウ形状と最大プーリング層の畳み込み層を介して情報を並列に抽出します。$1 \times 1$ 畳み込みは、ピクセル単位でチャネルの次元を減らします。最大プーリングは分解能を下げます。
* GoogleNetは、適切に設計された複数のInceptionブロックを他のレイヤーと直列に接続しています。Inception ブロックに割り当てられたチャネル数の比率は、ImageNet データセットに対する多数の実験によって得られます。
* GoogLeNet は、それに続くバージョンと同様、ImageNet で最も効率的なモデルの 1 つであり、計算の複雑さを抑えつつ、同様のテスト精度を実現しました。

## 演習

1. GoogLeNet にはいくつかのイテレーションがあります。実装して実行してみてください。その一部には次のものが含まれます。
    * :numref:`sec_batch_norm` で後述するように、バッチ正規化層 :cite:`Ioffe.Szegedy.2015` を追加します。
    * インセプションブロック :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` を調整します。
    * モデルの正則化 :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016` にはラベルスムージングを使用します。
    * :numref:`sec_resnet` で後述するように、残差接続 :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017` に含めます。
1. GoogleNetが動作する最小画像サイズはどれくらいですか？
1. AlexNet、VGG、および Nin のモデルパラメーターサイズを GoogleNet と比較します。後者の 2 つのネットワークアーキテクチャでは、モデルパラメーターのサイズが大幅に削減されるのはなぜですか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:

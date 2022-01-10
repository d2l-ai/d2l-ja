# ブロックを使用するネットワーク (VGG)
:label:`sec_vgg`

AlexNetは、深いCNNが良好な結果を達成できるという経験的証拠を提供しましたが、その後の研究者が新しいネットワークを設計する際にガイドする一般的なテンプレートを提供していませんでした。次のセクションでは、ディープネットワークの設計に一般的に使用されるヒューリスティックな概念をいくつか紹介します。 

この分野の進歩は、エンジニアがトランジスタの配置から論理素子、ロジックブロックへと移行したチップ設計における進歩を反映しています。同様に、ニューラルネットワークアーキテクチャの設計は次第に抽象的になり、研究者は個々のニューロンの観点から考えることから層全体、そして今やブロックや層の繰り返しのパターンへと移行していきました。 

ブロックを使用するというアイデアは、オックスフォード大学の [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG) の名を冠した*VGG* ネットワークで初めて生まれました。ループやサブルーチンを使用すれば、最新のディープラーニングフレームワークでこれらの繰り返し構造をコードに簡単に実装できます。 

## (**VGG ブロック**)
:label:`subsec_vgg-blocks`

古典的な CNN の基本的な構成要素は、(i) 分解能を維持するためのパディングのある畳み込み層、(ii) ReLU などの非線形性、(iii) 最大プーリング層などのプーリング層のシーケンスです。1 つの VGG ブロックは一連の畳み込み層で構成され、その後に空間的ダウンサンプリング用の最大プーリング層が続きます。元の VGG 論文 :cite:`Simonyan.Zisserman.2014` では、パディングが 1 のカーネル $3\times3$ (高さと幅を維持) とストライド 2 の最大プーリング $2 \times 2$ (ブロックごとに解像度を半減) の畳み込みを採用しています。以下のコードでは、1 つの VGG ブロックを実装する `vgg_block` という関数を定義します。

:begin_tab:`mxnet,tensorflow`
この関数は、畳み込み層の数 `num_convs` と出力チャネル数 `num_channels` に対応する 2 つの引数をとります。
:end_tab:

:begin_tab:`pytorch`
この関数は、畳み込み層の数 `num_convs`、入力チャネル数 `in_channels`、および出力チャネル数 `out_channels` に対応する 3 つの引数をとります。
:end_tab:

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## [**VGG ネットワーク**]

AlexNet や LeNet と同様に、VGG ネットワークは 2 つの部分に分割できます。1 つ目は主に畳み込み層とプーリング層で構成され、もう 1 つは完全接続された層で構成されます。これは:numref:`fig_vgg`に描かれています。 

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

ネットワークの畳み込み部分は、:numref:`fig_vgg` (`vgg_block` 関数でも定義) の複数の VGG ブロックを連続して接続します。次の変数 `conv_arch` はタプルのリスト (ブロックごとに 1 つ) で構成され、各タプルには畳み込み層の数と出力チャネル数の 2 つの値が含まれます。これらは `vgg_block` 関数を呼び出すのに必要な引数です。VGG ネットワークの完全に接続された部分は、AlexNet で説明されている部分と同じです。 

元の VGG ネットワークには 5 つの畳み込みブロックがあり、そのうち最初の 2 つにはそれぞれ 1 つの畳み込み層があり、後者の 3 つにはそれぞれ 2 つの畳み込み層が含まれています。最初のブロックには 64 個の出力チャネルがあり、後続の各ブロックは、その数が 512 に達するまで出力チャネル数を 2 倍にします。このネットワークは 8 つの畳み込み層と 3 つの全結合層を使用するため、VGG-11 と呼ばれることがよくあります。

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

次のコードは VGG-11 を実装しています。これは、`conv_arch` で for ループを実行するだけの単純な問題です。

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

次に、[**各レイヤーの出力形状を観察**] するために、高さと幅が 224 のシングルチャネルデータ例を構築します。

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

ご覧のとおり、各ブロックで高さと幅を半分にし、最終的に高さと幅を 7 にしてから、表現を平坦化してネットワークの完全に接続された部分で処理します。 

## 訓練

[**VGG-11 は AlexNet よりも計算量が多いので、より少ないチャネル数でネットワークを構築します。**] これは Fashion-MNist のトレーニングには十分すぎるほどです。

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Recall that this has to be a function that will be passed to
# `d2l.train_ch6()` so that model building/compiling need to be within
# `strategy.scope()` in order to utilize the CPU/GPU devices that we have
net = lambda: vgg(small_conv_arch)
```

少し大きい学習率を使用する以外は、[**モデルトレーニング**] のプロセスは :numref:`sec_alexnet` の AlexNet と同様です。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* VGG-11 は、再利用可能な畳み込みブロックを使用してネットワークを構築します。各ブロックの畳み込み層と出力チャネルの数の違いによって、異なる VGG モデルを定義できます。
* ブロックを使用すると、ネットワーク定義を非常にコンパクトに表現できます。これにより、複雑なネットワークの効率的な設計が可能になります。
* SimonyanとZisermanは、VGGの論文でさまざまなアーキテクチャを試しました。特に、深い畳み込みと狭い畳み込みのいくつかの層 (つまり、$3 \times 3$) が、より広い畳み込みの少ない層よりも効果的であることがわかった。

## 演習

1. レイヤーの寸法をプリントアウトすると、11個ではなく8個の結果しか見られませんでした。残りの3層情報はどこに行きましたか？
1. AlexNet と比較すると、VGG は計算速度がはるかに遅く、GPU メモリも必要になります。その理由を分析してください。
1. Fashion-MNISTの画像の高さと幅を224から96に変更してみてください。これは実験にどのような影響を与えますか？
1. VGG-16 や VGG-19 などの他の一般的なモデルを構築するには、VGG ペーパー :cite:`Simonyan.Zisserman.2014` の表 1 を参照してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:

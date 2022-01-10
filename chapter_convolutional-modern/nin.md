# ネットワーク内のネットワーク (NiN)
:label:`sec_nin`

LeNet、AlexNet、VGG はすべて共通の設計パターンを共有しています。一連の畳み込み層とプーリング層を介して*空間* 構造を利用して特徴を抽出し、完全に接続された層を介して表現を後処理します。AlexNetとVGGによるLeNetの改善は、主にこれらの後のネットワークがどのようにこれら2つのモジュールを広げ、深めるかにあります。あるいは、プロセスの早い段階で完全に接続されたレイヤーを使用することも想像できます。ただし、密度の高いレイヤーを不注意に使用すると、表現の空間構造が完全に失われる可能性があります。
*ネットワーク* (*nIN*) ブロック内のネットワークは代替手段を提供します。
それらは非常に単純な洞察に基づいて提案されました：各ピクセルのチャネルでMLPを別々に使用する :cite:`Lin.Chen.Yan.2013`。 

## (**nInブロック**)

畳み込み層の入力と出力は、例、チャネル、高さ、幅に対応する軸をもつ 4 次元のテンソルで構成されていることを思い出してください。また、全結合層の入力と出力は、通常、例と特徴に対応する 2 次元テンソルであることを思い出してください。NiN の背後にある考え方は、ピクセル位置ごとに (高さと幅ごとに) 完全に連結されたレイヤーを適用することです。各空間位置でウェイトを結び付けると、$1\times 1$ 畳み込みレイヤー (:numref:`sec_channels` を参照)、または各ピクセル位置で独立して動作する完全結合レイヤーと考えることができます。これを確認するもう 1 つの方法は、空間次元 (高さと幅) の各エレメントを例に相当し、チャネルはフィーチャと同等であると考えることです。 

:numref:`fig_nin` は、VGG と NiN、およびそれらのブロックの主な構造上の違いを示しています。NiN ブロックは、1 つの畳み込み層の後に 2 つの $1\times 1$ 畳み込み層で構成されます。この層は、ReLU アクティベーションによってピクセル単位の完全接続層として機能します。第 1 レイヤーの畳み込みウィンドウの形状は、通常、ユーザーが設定します。それ以降のウィンドウシェイプは $1 \times 1$ に固定されます。 

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## [**nIN モデル**]

オリジナルのNiNネットワークはAlexNetの直後に提案され、明らかにインスピレーションを得ています。NiN は $11\times 11$、$5\times 5$、$3\times 3$ のウィンドウ形状をもつ畳み込み層を使用し、対応する出力チャンネル数は AlexNet と同じです。各 NiN ブロックの後には、ストライド 2、ウィンドウ形状 $3\times 3$ を持つ最大プーリング層が続きます。 

NiN と AlexNet の大きな違いの 1 つは、NiN では完全に接続されたレイヤーを完全に回避できることです。代わりに、NiN は、ラベルクラスの数に等しい数の出力チャネルを持ち、その後に*global* 平均プーリング層が続く NiN ブロックを使用し、ロジットのベクトルを生成します。NiN の設計の利点の 1 つは、必要なモデルパラメータの数が大幅に削減されることです。ただし、実際には、この設計ではモデルトレーニング時間が長くなることがあります。

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

データ例を作成して [**各ブロックの出力形状**] を確認します。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**トレーニング**]

前と同じように、Fashion-MNISTを使ってモデルを訓練します。NiN のトレーニングは、AlexNet や VGG のトレーニングと似ています。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* NiN は、畳み込み層と複数の $1\times 1$ 畳み込み層で構成されるブロックを使用します。これを畳み込みスタック内で使用すると、ピクセル単位の非線形性を高めることができます。
* NiN は、チャネル数を必要な出力数に減らした後 (Fashion-MNIST では 10)、完全に接続されたレイヤーを削除し、グローバル平均プーリング (つまり、すべての場所の合計) に置き換えます。
* 完全に接続されたレイヤーを削除すると、オーバーフィットが減少します。NiN のパラメータは劇的に少なくなります。
* NiN設計は、その後の多くのCNN設計に影響を与えた。

## 演習

1. ハイパーパラメーターを調整して、分類の精度を向上させます。
1. NiN ブロックに 2 つの $1\times 1$ 畳み込み層があるのはなぜですか？そのうちの1つを取り除き、実験現象を観察して分析します。
1. NiN のリソース使用量を計算します。
    1. パラメータの数はいくつですか？
    1. 計算量はどれくらいですか？
    1. トレーニング中に必要な記憶量はどれくらいですか？
    1. 予測中に必要なメモリ容量はどれくらいですか？
1. $384 \times 5 \times 5$ 表現を $10 \times 5 \times 5$ 表現にワンステップで減らすことで考えられる問題は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:

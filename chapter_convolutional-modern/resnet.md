# 残留ネットワーク (ResNet)
:label:`sec_resnet`

ネットワークの設計がますます深くなっていくにつれて、レイヤを追加するとネットワークの複雑さと表現力がどのように増大するかを理解することが不可欠になります。さらに重要なのは、レイヤーを追加することで、ネットワークが単なる違いではなく、より厳密に表現力豊かになるようなネットワークを設計できることです。いくらか進歩するためには、少し数学が必要だ。 

## 関数クラス

$\mathcal{F}$ を考えてみましょう。これは、特定のネットワークアーキテクチャが (学習率やその他のハイパーパラメーター設定とともに) 到達できる関数のクラスです。つまり、すべての $f \in \mathcal{F}$ には、適切なデータセットでのトレーニングによって取得できるパラメーターのセット (重みやバイアスなど) が存在します。$f^*$ が、私たちが本当に見つけたい「真実」関数であると仮定しましょう。$\mathcal{F}$であれば、体調は良好ですが、通常はそれほど幸運ではありません。代わりに、$\mathcal{F}$内で最善の策である$f^*_\mathcal{F}$をいくつか見つけようとします。たとえば、フィーチャ $\mathbf{X}$、ラベル $\mathbf{y}$ をもつデータセットがある場合、次の最適化問題を解いて探すことができます。 

$$f^*_\mathcal{F} \stackrel{\mathrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

違った、より強力なアーキテクチャ $\mathcal{F}'$ を設計すれば、より良い結果が得られるはずだと仮定するのが妥当です。言い換えると、$f^*_{\mathcal{F}'}$ は $f^*_{\mathcal{F}}$ よりも「優れている」と予想されます。ただし、$\mathcal{F} \not\subseteq \mathcal{F}'$ の場合、これが起こるという保証はありません。実際、$f^*_{\mathcal{F}'}$ はもっと悪いかもしれません。:numref:`fig_functionclasses` で示されるように、ネストされていない関数クラスの場合、大きな関数クラスは必ずしも「真実」関数 $f^*$ に近づくとは限りません。たとえば、:numref:`fig_functionclasses` の左側では $\mathcal{F}_3$ は $\mathcal{F}_1$ よりも $f^*$ に近いですが、$\mathcal{F}_6$ は移動しなくなり、複雑さをさらに大きくしても $f^*$ から距離が短くなるという保証はありません。:numref:`fig_functionclasses` の右側にある $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$ のネストされた関数クラスを使用すると、ネストされていない関数クラスによる前述の問題を回避できます。 

![For non-nested function classes, a larger (indicated by area) function class does not guarantee to get closer to the "truth" function ($f^*$). This does not happen in nested function classes.](../img/functionclasses.svg)
:label:`fig_functionclasses`

したがって、大きな関数クラスに小さい関数クラスが含まれる場合にのみ、それらを大きくするとネットワークの表現力が厳密に増加することが保証されます。ディープニューラルネットワークでは、新しく追加された層を恒等関数 $f(\mathbf{x}) = \mathbf{x}$ に学習させることができれば、新しいモデルは元のモデルと同じ効果が得られます。新しいモデルではトレーニングデータセットに適合するより良いソリューションが得られる可能性があるため、レイヤーを追加するとトレーニングエラーを減らしやすくなる可能性があります。 

これは、彼らが非常に深いコンピュータービジョンモデル :cite:`He.Zhang.Ren.ea.2016` に取り組んでいるときに考慮した質問です。彼らが提案した*残差ネットワーク* (*ResNet*) の中心にあるのは、追加する層ごとに恒等関数をその要素の1つとしてより簡単に含めるべきだという考え方です。これらの考察はかなり深遠ですが、驚くほど単純な解法、つまり*残差ブロック*が生まれました。それにより、ResNetは2015年のImageNet大規模視覚認識チャレンジで優勝しました。この設計は、ディープニューラルネットワークの構築方法に大きな影響を与えました。 

## (**残留ブロック**)

:numref:`fig_residual_block` に示すように、ニューラルネットワークのローカル部分に注目してみましょう。入力を $\mathbf{x}$ で表します。学習によって取得したい基本となるマッピングは $f(\mathbf{x})$ で、上のアクティベーション関数への入力として使用すると仮定します。:numref:`fig_residual_block` の左側では、点線ボックス内の部分でマッピング $f(\mathbf{x})$ を直接学習する必要があります。右側の点線ボックス内の部分は、*残差マッピング* $f(\mathbf{x}) - \mathbf{x}$ を学習する必要があります。これは、残差ブロックの名前が由来する方法です。恒等マッピング $f(\mathbf{x}) = \mathbf{x}$ が望ましい基礎となるマッピングである場合、残差マッピングの方が学習しやすくなります。点線ボックス内の上位の重み層 (全結合層と畳み込み層など) の重みとバイアスをゼロにするだけで済みます。:numref:`fig_residual_block` の右図は ResNet の*残差ブロック* を示しています。レイヤー入力 $\mathbf{x}$ を加算演算子に運ぶ実線は、*残差接続* (または*ショートカット接続*) と呼ばれます。Residual ブロックを使用すると、入力は層間の残差接続を通じてより速く順伝播できます。 

![A regular block (left) and a residual block (right).](../img/residual-block.svg)
:label:`fig_residual_block`

ResNet は、VGG の完全な $3\times 3$ 畳み込み層設計に従っています。残差ブロックには、同じ数の出力チャネルをもつ 2 つの $3\times 3$ 畳み込み層があります。各畳み込み層の後には、バッチ正規化層と ReLU 活性化関数が続きます。次に、これら 2 つの畳み込み演算をスキップして、最後の ReLU アクティベーション関数の直前に入力を追加します。このような設計では、2 つの畳み込み層の出力を足し合わせることができるように、入力と同じ形状にする必要があります。チャンネル数を変更したい場合は、$1\times 1$ 畳み込み層を追加して、加算演算のために入力を目的の形状に変換する必要があります。以下のコードを見てみましょう。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

このコードは 2 種類のネットワークを生成します。1 つは `use_1x1conv=False` のときに ReLU 非線形性を適用する前に入力を出力に追加し、もう 1 つは加算前に $1 \times 1$ 畳み込みによってチャネルと解像度を調整するものです。:numref:`fig_resnet_block` は、このことを示しています。 

![ResNet block with and without $1 \times 1$ convolution.](../img/resnet-block.svg)
:label:`fig_resnet_block`

ここで、[**入力と出力が同じ形状である状況**] を見てみましょう。

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

また、[**出力の高さと幅を半分にし、出力チャンネル数を増やす**] というオプションもあります。

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

## [**ResNet モデル**]

ResNet の最初の 2 つの層は、前に説明した GoogLeNet の層と同じです。64 個の出力チャネルを持ち、ストライドが 2 の $7\times 7$ 畳み込み層に続いて、ストライドが 2 の $3\times 3$ 最大プーリング層が続きます。違いは、ResNet の各畳み込み層の後に追加されるバッチ正規化層です。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

GoogleNet は、Inception ブロックで構成された 4 つのモジュールを使用します。ただし、ResNet は残差ブロックで構成された 4 つのモジュールを使用し、各モジュールは同じ数の出力チャネルを持つ複数の残差ブロックを使用します。最初のモジュールのチャンネル数は入力チャンネル数と同じです。ストライドが 2 の最大プーリング層が既に使用されているため、高さと幅を小さくする必要はありません。後続の各モジュールの最初の残差ブロックでは、前のモジュールと比較してチャネル数が2倍になり、高さと幅が半分になります。 

ここで、このモジュールを実装します。最初のモジュールでは特別な処理が行われていることに注意してください。

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

次に、すべてのモジュールを ResNet に追加します。ここでは、各モジュールに 2 つの残差ブロックが使用されています。

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

最後に、GoogleNet と同様に、グローバル平均プーリング層を追加し、その後に全結合層出力を追加します。

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that we define this as a function so we can reuse later and run it
# within `tf.distribute.MirroredStrategy`'s scope to utilize various
# computational resources, e.g. GPUs. Also note that even though we have
# created b1, b2, b3, b4, b5 but we will recreate them inside this function's
# scope instead
def net():
    return tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

各モジュールには 4 つの畳み込み層があります ($1\times 1$ 畳み込み層を除く)。最初の $7\times 7$ 畳み込み層と最後の完全結合層とともに、全部で 18 層あります。したがって、このモデルは一般にResNet-18として知られています。モジュール内に異なる数のチャネルと残差ブロックを設定することで、より深い 152 層 ResNet-152 など、さまざまな ResNet モデルを作成できます。ResNetの主なアーキテクチャはGoogleNetのアーキテクチャと似ていますが、ResNetの構造はよりシンプルで変更が容易です。これらすべての要因により、ResNetは急速かつ広く使用されています。:numref:`fig_resnet18`はResNet-18のフルサイズを表しています。 

![The ResNet-18 architecture.](../img/resnet18.svg)
:label:`fig_resnet18`

ResNet を学習させる前に、[**ResNet の異なるモジュール間で入力形状がどのように変化するかを観察します**]。これまでのすべてのアーキテクチャと同様に、グローバル平均プーリング層がすべてのフィーチャを集約するまで、チャネル数が増加するにつれて解像度は低下します。

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
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## [**トレーニング**]

以前と同じように、ResNet を Fashion-MNIST データセットでトレーニングします。

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* 入れ子になった関数クラスが望ましいです。ディープニューラルネットワークの追加層を恒等関数として学習することは (極端なケースですが)、簡単になるはずです。
* 残差マッピングでは、ウェイトレイヤーのパラメーターを 0 にプッシュするなど、恒等関数をより簡単に学習できます。
* 残差ブロックを持つことで、効果的なディープニューラルネットワークに学習させることができます。入力は、層間の残差接続を通じてより速く順伝播できます。
* ResNetは、畳み込みとシーケンシャルの両方の性質のために、その後のディープニューラルネットワークの設計に大きな影響を与えました。

## 演習

1. :numref:`fig_inception` のインセプションブロックと残差ブロックの主な違いは何ですか？Inception ブロックでいくつかのパスを削除した後、それらのパスは相互にどのように関係していますか？
1. さまざまなバリアントを実装するには、ResNet ペーパー :cite:`He.Zhang.Ren.ea.2016` の表 1 を参照してください。
1. より深いネットワークでは、ResNet は「ボトルネック」アーキテクチャを導入し、モデルの複雑さを軽減します。実装してみてください。
1. ResNet の以降のバージョンでは、著者らは「畳み込み、バッチの正規化、および活性化」の構造を「バッチ正規化、活性化、畳み込み」の構造に変更しました。この改善を自分で行ってください。詳細については、:cite:`He.Zhang.Ren.ea.2016*1` の図1 を参照してください。
1. 関数クラスが入れ子になっていても、束縛されずに関数の複雑さを増すことができないのはなぜですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/333)
:end_tab:

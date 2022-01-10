# カスタムレイヤ

ディープラーニングの成功の要因の 1 つは、さまざまなタスクに適したアーキテクチャを設計するために、クリエイティブな方法で構成できる幅広いレイヤーを利用できることです。たとえば、研究者は、画像、テキストの処理、シーケンシャルデータのループ、動的プログラミングの実行に特化したレイヤーを考案しました。遅かれ早かれ、ディープラーニングフレームワークにまだ存在しない層に出会ったり、考案したりするでしょう。このような場合は、カスタム Layer を構築する必要があります。このセクションでは、その方法を説明します。 

## (**パラメータのないレイヤ**)

まず、独自のパラメーターを持たないカスタム Layer を作成します。:numref:`sec_model_construction` の block の導入を思い出せば、これはおなじみのように思えるでしょう。次の `CenteredLayer` クラスは、単純に入力から平均を減算します。それを構築するには、基本レイヤークラスから継承し、順伝播関数を実装するだけです。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

レイヤーにデータを入力して、レイヤーが意図したとおりに機能することを確認しましょう。

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

これで、[**レイヤーをコンポーネントとして組み込んで、より複雑なモデルを構築できるようになりました**]

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

追加のサニティチェックとして、ランダムデータをネットワーク経由で送信し、平均値が実際に 0 であることを確認できます。ここでは浮動小数点数を扱っているため、量子化によってゼロ以外の非常に小さい数値が表示されることがあります。

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## [**パラメータ付きの画層**]

単純層の定義方法がわかったところで、学習によって調整できるパラメーターを持つ層の定義に移りましょう。組み込み関数を使用して、基本的なハウスキーピング機能を提供するパラメーターを作成できます。特に、モデルパラメーターのアクセス、初期化、共有、保存、読み込みを制御します。これにより、他の利点の中でも、すべてのカスタム Layer に対してカスタムのシリアル化ルーチンを記述する必要がなくなります。 

それでは、完全接続されたレイヤーの独自のバージョンを実装しましょう。このレイヤーには 2 つのパラメーターが必要であることを思い出してください。1 つはウェイトを表し、もう 1 つはバイアス用です。この実装では、デフォルトとして ReLU アクティベーションをベイクインします。この層には、`in_units` と `units` の 2 つの入力引数が必要です。これらの引数は、それぞれ入力と出力の数を表します。

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow`
次に、`MyDense` クラスをインスタンス化し、そのモデルパラメーターにアクセスします。
:end_tab:

:begin_tab:`pytorch`
次に、`MyLinear` クラスをインスタンス化し、そのモデルパラメーターにアクセスします。
:end_tab:

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

[**カスタム層を使用して順方向伝播計算を直接実行できます**]

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

また、(**カスタムレイヤーを使用してモデルを構築**) できれば、組み込みの完全接続レイヤーと同じように使用できます。

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## [概要

* 基本レイヤクラスを介してカスタムレイヤを設計できます。これにより、ライブラリ内の既存のレイヤーとは異なる動作をする柔軟性のある新しいレイヤーを定義できます。
* カスタム Layer を定義すると、任意のコンテキストやアーキテクチャでカスタム Layer を呼び出すことができます。
* レイヤには、組み込み関数を使用して作成できるローカルパラメータを含めることができます。

## 演習

1. 入力を受け取り、テンソルリダクションを計算する、つまり $y_k = \sum_{i, j} W_{ijk} x_i x_j$ を返す層を設計します。
1. データのフーリエ係数の前半を返す層を設計します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:

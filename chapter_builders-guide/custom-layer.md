# カスタムレイヤー

ディープラーニングの成功の背後にある要因の 1 つは、さまざまなタスクに適したアーキテクチャを設計するために創造的な方法で構成できる幅広いレイヤーが利用できることです。たとえば、研究者は、画像、テキストの処理、順次データのループ、動的計画法の実行に特化したレイヤーを発明しました。遅かれ早かれ、ディープラーニングのフレームワークにはまだ存在しない層に出会ったり、発明したりするでしょう。このような場合は、カスタム Layer を構築する必要があります。このセクションでは、その方法を説明します。 

## (**パラメータなしのレイヤー**)

まず、独自のパラメータを持たないカスタム Layer を構築します。:numref:`sec_model_construction`のモジュールの紹介を思い出せば、これはおなじみのように思えます。次の `CenteredLayer` クラスは、入力から平均を単純に減算します。それを構築するには、基本レイヤークラスから継承し、フォワードプロパゲーション関数を実装するだけです。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
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
%%tab pytorch
from d2l import torch as d2l
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
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

データを通すことで、レイヤーが意図したとおりに機能することを確認しましょう。

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

これで [**より複雑なモデルを構築するためのコンポーネントとしてレイヤーを組み込むことができます。**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

追加の健全性チェックとして、ネットワークを介してランダムデータを送信し、平均が実際には0であることを確認できます。浮動小数点数を扱っているため、量子化によってゼロ以外の非常に小さい数値が表示される場合があります。

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## [**パラメータ付きレイヤー**]

単純なレイヤーの定義方法がわかったので、トレーニングを通じて調整できるパラメーターを持つレイヤーの定義に移りましょう。組み込み関数を使用して、基本的なハウスキーピング機能を提供するパラメーターを作成できます。特に、モデルパラメーターのアクセス、初期化、共有、保存、および読み込みを制御します。この方法では、他の利点の中でも、すべてのカスタム Layer に対してカスタムシリアル化ルーチンを記述する必要がなくなります。 

それでは、完全接続レイヤーの独自のバージョンを実装しましょう。この層には 2 つのパラメーターが必要であることを思い出してください。1 つは重みを表し、もう 1 つは偏りを表します。この実装では、ReLU アクティベーションをデフォルトとして組み込みます。この層には、それぞれ入力と出力の数を示す `in_units` と `units` の 2 つの入力引数が必要です。

```{.python .input}
%%tab mxnet
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
%%tab pytorch
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
%%tab tensorflow
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
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

[**カスタムレイヤーを使用してフォワードプロパゲーション計算を直接実行できます。**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

また、(**カスタムレイヤーを使用してモデルを構築する。**) それができれば、組み込みの完全接続レイヤーと同じように使用できます。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## まとめ

* 基本レイヤクラスを介してカスタムレイヤを設計できます。これにより、ライブラリ内の既存のレイヤーとは異なる動作をする柔軟な新しいレイヤーを定義できます。
* 一度定義すると、カスタム Layer は任意のコンテキストやアーキテクチャで呼び出すことができます。
* レイヤーには、組み込み関数を使用して作成できるローカルパラメーターを含めることができます。

## 演習

1. 入力を受け取り、テンソル削減を計算する層を設計します。つまり、$y_k = \sum_{i, j} W_{ijk} x_i x_j$を返します。
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

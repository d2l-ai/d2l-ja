# パラメータ管理

アーキテクチャを選択してハイパーパラメータを設定したら、学習ループに進みます。ここでは、損失関数を最小化するパラメータ値を見つけることが目標です。トレーニング後、将来の予測を行うためにこれらのパラメータが必要になります。さらに、パラメータを抽出して、他のコンテキストで再利用したり、モデルをディスクに保存して他のソフトウェアで実行したり、科学的な理解を得るために検討したりすることがあります。 

ほとんどの場合、重労働を行うためにディープラーニングフレームワークに依存して、パラメーターの宣言と操作方法の本質的な詳細を無視することができます。しかし、標準レイヤーを持つスタックアーキテクチャから離れると、パラメーターの宣言と操作の雑草に入る必要がある場合があります。このセクションでは、以下について説明します。 

* デバッグ、診断、および視覚化のためのパラメーターへのアクセス。
* 異なるモデルコンポーネント間でパラメータを共有する。

(**まず、隠れ層が1つあるMLPに焦点を当てることから始めます**)

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

## [**パラメータアクセス**]

まず、既に知っているモデルからパラメータにアクセスする方法から始めましょう。`Sequential` クラスを介してモデルが定義されている場合、リストであるかのようにモデルにインデックスを付けることで、まず任意のレイヤーにアクセスできます。各レイヤーのパラメーターは、その属性に便利に配置されています。次のように、第2の全結合層のパラメータを調べることができます。

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

この完全に接続されたレイヤーには、そのレイヤーの重みとバイアスにそれぞれ対応する 2 つのパラメーターが含まれていることがわかります。 

### [**ターゲットパラメータ**]

各パラメータは、パラメータクラスのインスタンスとして表されることに注意してください。パラメータで役に立つことをするには、まず基礎となる数値にアクセスする必要があります。これにはいくつかの方法があります。いくつかはより単純ですが、他のものはより一般的です。次のコードは、パラメータクラスインスタンスを返す 2 番目のニューラルネットワーク層からバイアスを抽出し、さらにそのパラメータの値にアクセスします。

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

:begin_tab:`mxnet,pytorch`
パラメータは、値、グラデーション、および追加情報を含む複雑なオブジェクトです。だからこそ、値を明示的に要求する必要があります。 

値に加えて、各パラメータでグラデーションにアクセスすることもできます。このネットワークに対してバックプロパゲーションをまだ呼び出していないため、初期状態です。
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**すべてのパラメータを一度に**]

すべてのパラメータに対して操作を実行する必要がある場合、それらに 1 つずつアクセスするのは面倒です。より複雑なモジュール (ネストされたモジュールなど) を扱う場合、各サブモジュールのパラメータを抽出するためにツリー全体を再帰的に処理する必要があるため、状況は特に扱いにくくなります。以下では、すべてのレイヤーのパラメーターにアクセスする方法を示します。

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

## [**結び付けられたパラメータ**]

多くの場合、複数のレイヤーでパラメーターを共有したいと考えています。これをエレガントに行う方法を見てみましょう。以下では、完全に接続されたレイヤーを割り当て、そのパラメーターを使用して別のレイヤーのパラメーターを設定します。ここでは、パラメータにアクセスする前に前方伝播`net(X)`を実行する必要があります。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])
net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

この例は、2番目と3番目のレイヤーのパラメーターが関連付けられていることを示しています。それらは等しいだけではなく、同じ正確なテンソルで表されます。したがって、パラメータの1つを変更すると、他のパラメータも変更されます。パラメータが関連付けられると、グラデーションはどうなるのだろうかと思うかもしれません。モデルパラメーターには勾配が含まれているため、2 番目の非表示レイヤーと 3 番目の非表示レイヤーのグラデーションは、バックプロパゲーション中に一緒に加算されます。 

## まとめ

モデルパラメータにアクセスして結び付ける方法はいくつかあります。 

## 演習

1. :numref:`sec_model_construction` で定義されている `NestMLP` モデルを使用して、さまざまなレイヤーのパラメーターにアクセスします。
1. 共有パラメーター層を含む MLP を構築し、学習させます。トレーニングプロセス中に、各レイヤーのモデルパラメーターと勾配を観察します。
1. なぜパラメータを共有するのが良いのですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:

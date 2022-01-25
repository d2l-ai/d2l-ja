# パラメータ管理

アーキテクチャを選択してハイパーパラメーターを設定したら、学習ループに進みます。ここでは、損失関数を最小化するパラメーター値を見つけることが目標です。トレーニング後、将来の予測を行うためにこれらのパラメータが必要になります。さらに、パラメーターを抽出して、他のコンテキストで再利用したり、モデルをディスクに保存して他のソフトウェアで実行できるようにしたり、科学的な理解を得るための調査のためにパラメーターを抽出したい場合があります。 

ほとんどの場合、ディープラーニングフレームワークに頼って重労働を行うことで、パラメーターの宣言と操作方法に関する重要な詳細を無視することができます。しかし、標準レイヤーを持つスタックアーキテクチャから離れると、パラメーターの宣言と操作の雑草に陥る必要が生じることがあります。このセクションでは、次の内容について説明します。 

* デバッグ、診断、可視化のためのパラメーターへのアクセス。
* パラメーターの初期化。
* 異なるモデルコンポーネント間でパラメータを共有する。

(**まず、隠れ層が1つあるMLPに着目します。**)

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## [**パラメータアクセス**]

既に知っているモデルからパラメータにアクセスする方法から始めましょう。`Sequential` クラスを介してモデルを定義すると、リストであるかのようにモデルにインデックスを付けることで、どのレイヤーにもまずアクセスできます。各レイヤのパラメータは、その属性に便利に配置されています。2 番目の全結合層のパラメーターを調べるには、次のようにします。

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

この出力から、いくつかの重要なことが分かります。まず、この完全に接続されたレイヤには、そのレイヤのウェイトとバイアスに対応する 2 つのパラメータが含まれています。どちらも単精度浮動小数点数 (float32) として格納されます。パラメーターの名前により、数百ものレイヤーを含むネットワーク内であっても、各レイヤーのパラメーターを一意に識別できます。 

### [**ターゲットパラメータ**]

各パラメータは、パラメータクラスのインスタンスとして表されることに注意してください。パラメータで何か役に立つことを行うには、まず基礎となる数値にアクセスする必要があります。これにはいくつかの方法があります。より単純なものもあれば、より一般的なものもあります。次のコードは、パラメータクラスインスタンスを返す 2 番目のニューラルネットワーク層からバイアスを抽出し、さらにそのパラメータの値にアクセスします。

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
パラメータは、値、グラデーション、追加情報を含む複雑なオブジェクトです。そのため、値を明示的に要求する必要があります。 

値に加えて、各パラメーターでグラデーションにアクセスすることもできます。このネットワークのバックプロパゲーションはまだ起動していないため、初期状態です。
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### [**すべてのパラメータを一度に**]

すべてのパラメータに対して操作を実行する必要がある場合、それらに 1 つずつアクセスするのは面倒です。より複雑なブロック (ネストされたブロックなど) を扱う場合、各サブブロックのパラメーターを抽出するためにツリー全体を再帰的に処理する必要があるため、状況は特に扱いにくくなります。以下では、最初に完全に接続されたレイヤーのパラメーターにアクセスする方法と、すべてのレイヤーにアクセスする方法について説明します。

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

これにより、次のようにネットワークのパラメータにアクセスする別の方法が提供されます。

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### [**ネストされたブロックからパラメータを収集する**]

複数のブロックを互いに入れ子にした場合、パラメーターの命名規則がどのように機能するかを見てみましょう。そのためには、まずブロックを生成する関数 (いわばブロックファクトリ) を定義し、さらに大きなブロック内でこれらを結合します。

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Nested here
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

[**ネットワークの設計が完了しました。ネットワークがどのように構成されているか見てみましょう**]

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

レイヤーは階層的にネストされているので、ネストされたリストでインデックスを作成するかのようにレイヤーにアクセスすることもできます。たとえば、最初のメジャーブロックにアクセスし、その中で2番目のサブブロックにアクセスし、その中で第1レイヤーのバイアスにアクセスするには、次のようにします。

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## パラメーターの初期化

パラメータへのアクセス方法がわかったところで、パラメータを正しく初期化する方法を見ていきましょう。:numref:`sec_numerical_stability` では、適切な初期化の必要性について説明しました。ディープラーニングフレームワークは、その層にデフォルトのランダム初期化を提供します。ただし、他のさまざまなプロトコルに従って重みを初期化したい場合がよくあります。このフレームワークは、最も一般的に使用されるプロトコルを提供し、カスタムイニシャライザを作成することもできます。

:begin_tab:`mxnet`
既定では、MXNet は一様分布 $U(-0.07, 0.07)$ からランダムに抽出し、バイアスパラメーターを 0 にクリアすることで、重みパラメーターを初期化します。MXNet の `init` モジュールは、さまざまなプリセット初期化方法を提供します。
:end_tab:

:begin_tab:`pytorch`
既定では、PyTorch は入力次元と出力次元に従って計算された範囲から描画することにより、重み行列とバイアス行列を一様に初期化します。PyTorch の `nn.init` モジュールは、さまざまなプリセット初期化メソッドを提供します。
:end_tab:

:begin_tab:`tensorflow`
デフォルトでは、Keras は入力次元と出力次元に従って計算された範囲から描画することで重み行列を均一に初期化し、バイアスパラメータはすべてゼロに設定されます。TensorFlow は、ルートモジュールと `keras.initializers` モジュールの両方でさまざまな初期化方法を提供します。
:end_tab:

### [**ビルトイン初期化**]

まず、組み込みイニシャライザを呼び出すことから始めましょう。以下のコードは、すべての重みパラメーターを標準偏差 0.01 のガウス確率変数として初期化し、バイアスパラメーターはゼロにクリアしています。

```{.python .input}
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

また、すべてのパラメータを指定された定数値 (1 など) に初期化することもできます。

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

[**特定のブロックに異なるイニシャライザを適用することもできます**] 例えば、以下では第1層をXavierイニシャライザで初期化し、第2層を定数42に初期化します。

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### [**カスタム初期化**]

必要な初期化方法が、ディープラーニングフレームワークによって提供されない場合があります。以下の例では、次の奇妙な分布を使用して、任意の加重パラメータ $w$ のイニシャライザを定義します。 

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
ここでは `Initializer` クラスのサブクラスを定義します。通常、実装する必要があるのはテンソル引数 (`data`) を取り、必要な初期化値を代入する `_init_weight` 関数のみです。
:end_tab:

:begin_tab:`pytorch`
ここでも `net` に適用する `my_init` 関数を実装します。
:end_tab:

:begin_tab:`tensorflow`
ここでは `Initializer` のサブクラスを定義し、形状とデータ型を指定して目的のテンソルを返す関数 `__call__` を実装します。
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor        

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

パラメーターを直接設定するオプションは常にあることに注意してください。

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

:begin_tab:`mxnet`
上級ユーザーへの注意:`autograd` スコープ内でパラメーターを調整する場合は、自動微分の仕組みを混乱させないように `set_data` を使用する必要があります。
:end_tab:

## [**同点パラメータ**]

多くの場合、複数のレイヤーにわたってパラメーターを共有する必要があります。これをエレガントに行う方法を見てみましょう。以下では、高密度レイヤーを割り当て、そのパラメーターを使用して別のレイヤーのパラメーターを設定します。

```{.python .input}
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
#@tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
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

:begin_tab:`mxnet,pytorch`
この例は、2 番目と 3 番目のレイヤーのパラメーターが結び付けられていることを示しています。それらは等しいだけでなく、同じ正確なテンソルで表されます。したがって、一方のパラメータを変更すると、もう一方のパラメータも変更されます。パラメータを結び付けると、グラデーションはどうなるのか疑問に思うかもしれません。モデルパラメーターには勾配が含まれているため、2 番目の隠れ層と 3 番目の隠れ層の勾配は逆伝播時に加算されます。
:end_tab:

## [概要

* モデルパラメーターへのアクセス、初期化、および結び付けを行う方法はいくつかあります。
* カスタム初期化を使用できます。

## 演習

1. :numref:`sec_model_construction` で定義されている `FancyMLP` モデルを使用して、さまざまなレイヤのパラメータにアクセスします。
1. 初期化モジュールのドキュメントを見て、さまざまなイニシャライザを調べてください。
1. 共有パラメーター層を含む MLP を構築し、学習させます。学習プロセス中に、各層のモデルパラメーターと勾配を観察します。
1. パラメーターの共有がなぜ良いアイデアなのですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:

# パラメーターの初期化

パラメータにアクセスする方法がわかったので、それらを正しく初期化する方法を見てみましょう。:numref:`sec_numerical_stability`では、適切な初期化の必要性について説明しました。ディープラーニングフレームワークは、そのレイヤーにデフォルトのランダム初期化を提供します。しかし、私たちはしばしば、他のさまざまなプロトコルに従って重みを初期化したいと考えています。このフレームワークは、最も一般的に使用されるプロトコルを提供し、カスタムイニシャライザを作成することもできます。

:begin_tab:`mxnet`
既定では、MXNet は一様分布 $U(-0.07, 0.07)$ からランダムに抽出して重みパラメーターを初期化し、バイアスパラメーターをゼロにクリアします。MXNetの`init`モジュールは、さまざまなプリセット初期化方法を提供します。
:end_tab:

:begin_tab:`pytorch`
デフォルトでは、PyTorch は、入力と出力の次元に従って計算された範囲から描画することにより、重みとバイアスの行列を均一に初期化します。PyTorch の `nn.init` モジュールは、さまざまなプリセット初期化メソッドを提供します。
:end_tab:

:begin_tab:`tensorflow`
デフォルトでは、Kerasは入力と出力の次元に従って計算された範囲から引き出すことによって重み行列を均一に初期化し、バイアスパラメータはすべてゼロに設定されます。TensorFlow は、ルートモジュールと `keras.initializers` モジュールの両方でさまざまな初期化方法を提供します。
:end_tab:

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
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

```{.python .input  n=3}
%%tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input  n=4}
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

## [**組み込みの初期化**]

組み込みのイニシャライザを呼び出すことから始めましょう。以下のコードは、すべての重みパラメータを標準偏差0.01のガウス確率変数として初期化し、バイアスパラメータはゼロにクリアされています。

```{.python .input  n=5}
%%tab mxnet
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=6}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=7}
%%tab tensorflow
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

また、すべてのパラメータを指定された定数値 (たとえば 1) に初期化することもできます。

```{.python .input  n=8}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=9}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=10}
%%tab tensorflow
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

[**特定のブロックに異なるイニシャライザを適用することもできます。**] たとえば、以下では、Xavier イニシャライザで最初のレイヤを初期化し、2 番目のレイヤを定数値 42 に初期化します。

```{.python .input  n=11}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input  n=12}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input  n=13}
%%tab tensorflow
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

必要な初期化方法が、ディープラーニングフレームワークによって提供されない場合があります。以下の例では、次の奇妙な分布を使用して、任意の重みパラメータ $w$ のイニシャライザを定義します。 

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
ここでは、`Initializer` クラスのサブクラスを定義します。通常は、テンソル引数 (`data`) を受け取り、必要な初期化値を代入する `_init_weight` 関数のみを実装する必要があります。
:end_tab:

:begin_tab:`pytorch`
ここでも、`net` に適用する `my_init` 関数を実装します。
:end_tab:

:begin_tab:`tensorflow`
ここでは、`Initializer`のサブクラスを定義し、形状とデータ型を指定して必要なテンソルを返す`__call__`関数を実装します。
:end_tab:

```{.python .input  n=14}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input  n=15}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input  n=16}
%%tab tensorflow
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

パラメータを直接設定するオプションが常にあることに注意してください。

```{.python .input  n=17}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input  n=18}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input  n=19}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## まとめ

組み込みのイニシャライザとカスタムイニシャライザを使用してパラメータを初期化できます。 

## 演習

その他の組み込みイニシャライザについては、オンラインドキュメントを参照してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:

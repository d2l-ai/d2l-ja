# リカレントニューラルネットワークの簡潔な実装
:label:`sec_rnn-concise`

:numref:`sec_rnn_scratch` は、RNN がどのように実装されているかを知るのに有益でしたが、これは便利でも高速でもありません。このセクションでは、ディープラーニングフレームワークの高レベル API によって提供される関数を使用して、同じ言語モデルをより効率的に実装する方法を説明します。前と同じように、タイムマシンデータセットを読み取ることから始めます。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## [**モデルの定義**]

高レベル API は、リカレントニューラルネットワークの実装を提供します。1 つの隠れ層と 256 個の隠れ単位をもつリカレントニューラルネットワーク層 `rnn_layer` を構築します。実際、複数のレイヤーを持つことの意味についてはまだ説明していません。これは :numref:`sec_deep_rnn` で発生します。今のところ、複数の層は、単にRNNの次の層の入力として使用されるRNNの1つの層の出力に相当すると言えば十分である。

```{.python .input}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

```{.python .input}
#@tab tensorflow
num_hiddens = 256
rnn_cell = tf.keras.layers.SimpleRNNCell(num_hiddens,
    kernel_initializer='glorot_uniform')
rnn_layer = tf.keras.layers.RNN(rnn_cell, time_major=True,
    return_sequences=True, return_state=True)
```

:begin_tab:`mxnet`
隠れ状態の初期化は簡単です。メンバー関数 `begin_state` を呼び出します。これにより、ミニバッチ内の各例の初期非表示状態を含むリスト (`state`) が返され、その形状は (非表示レイヤーの数、バッチサイズ、非表示ユニットの数) になります。後から導入されるモデル (長期短期記憶など) については、このようなリストには他の情報も含まれています。
:end_tab:

:begin_tab:`pytorch`
形状が（隠れ層の数、バッチサイズ、隠れユニットの数）である（**テンソルを使って隠れ状態を初期化します**）。
:end_tab:

```{.python .input}
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state = torch.zeros((1, batch_size, num_hiddens))
state.shape
```

```{.python .input}
#@tab tensorflow
state = rnn_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
state.shape
```

[**隠れ状態と入力があれば、更新された隠れ状態で出力を計算できます**] `rnn_layer` の「出力」(`Y`) には出力層の計算は含まれないことを強調しておく必要があります。これは*各*タイムステップでの隠れ状態を参照し、後続の出力レイヤー。

:begin_tab:`mxnet`
また、`rnn_layer` によって返される更新された隠れ状態 (`state_new`) は、ミニバッチの *最後* タイムステップにおける隠れ状態を参照します。シーケンシャル・パーティショニングでは、エポック内の次のミニバッチの隠れ状態を初期化するために使用できます。非表示レイヤーが複数ある場合、各レイヤーの非表示状態はこの変数 (`state_new`) に格納されます。後から導入されるモデル (長期短期記憶など) では、この変数には他の情報も含まれます。
:end_tab:

```{.python .input}
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

:numref:`sec_rnn_scratch` と同様、[**完全な RNN モデルには `RNNModel` クラスを定義しています。**] `rnn_layer` には隠れたリカレント層しか含まれていないので、別の出力層を作成する必要があります。

```{.python .input}
#@save
class RNNModel(nn.Block):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully-connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

```{.python .input}
#@tab pytorch
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

```{.python .input}
#@tab tensorflow
#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # Later RNN like `tf.keras.layers.LSTMCell` return more than two values
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)
```

## トレーニングと予測

モデルに学習をさせる前に、[**ランダムな重みをもつモデルで予測をしてみよう。**]

```{.python .input}
device = d2l.try_gpu()
net = RNNModel(rnn_layer, len(vocab))
net.initialize(force_reinit=True, ctx=device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab pytorch
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
```

```{.python .input}
#@tab tensorflow
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    net = RNNModel(rnn_layer, vocab_size=len(vocab))

d2l.predict_ch8('time traveller', 10, net, vocab)
```

明らかなように、このモデルはまったく機能しません。次に :numref:`sec_rnn_scratch` で定義されているのと同じハイパーパラメーターで `train_ch8` を呼び出し、[**高レベル API でモデルをトレーニング**] します。

```{.python .input}
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

このモデルは、ディープラーニングフレームワークの高レベル API によってコードが最適化されているため、前回のセクションと比較して、短期間ではあるが同等のパープレキシティを実現しています。 

## [概要

* ディープラーニングフレームワークの高レベル API は、RNN 層の実装を提供します。
* 高レベル API の RNN 層は、出力と更新された隠れ状態を返します。出力には出力層の計算は含まれません。
* 高レベル API を使用すると、実装をゼロから使用するよりも高速な RNN トレーニングが可能になります。

## 演習

1. 高レベル API を使用して RNN モデルをオーバーフィットさせることはできますか？
1. RNN モデルで隠れ層の数を増やすとどうなりますか？モデルを機能させることはできますか？
1. RNN を使用して :numref:`sec_sequence` の自己回帰モデルを実装します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:

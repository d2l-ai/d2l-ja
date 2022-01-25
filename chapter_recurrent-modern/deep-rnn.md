# ディープリカレントニューラルネットワーク

:label:`sec_deep_rnn` 

これまでは、単一方向隠れ層をもつRNNについてのみ議論してきました。その中で、潜在変数とオブザベーションがどのように相互作用するかという特定の関数形式はかなり恣意的です。さまざまなタイプの相互作用をモデル化するのに十分な柔軟性があれば、これは大きな問題ではありません。ただし、単一レイヤーでは、これは非常に困難な場合があります。線形モデルの場合、レイヤーを追加することでこの問題を修正しました。RNNでは、非線形性を追加する方法と場所を最初に決定する必要があるため、これは少しトリッキーです。 

実際、RNNの複数の層を積み重ねることができます。これにより、複数の単純なレイヤーの組み合わせにより、柔軟なメカニズムが実現します。特に、データはスタックのさまざまなレベルで関連性がある可能性があります。たとえば、金融市場の状況 (弱気相場または強気相場) に関する高レベルのデータを入手可能に保ちながら、より低いレベルでは短期的な時間的ダイナミクスのみを記録したい場合があります。 

上記の抽象的な議論以外にも、:numref:`fig_deep_rnn` を見直すことで、関心のあるモデル群を理解するのがおそらく最も簡単でしょう。$L$ の隠れ層をもつディープ RNN を記述します。各非表示状態は、カレントレイヤの次のタイムステップと次のレイヤのカレントタイムステップの両方に連続的に渡されます。 

![Architecture of a deep RNN.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

## 機能的な依存関係

:numref:`fig_deep_rnn` に示されている $L$ の隠れ層のディープアーキテクチャ内で、関数の依存関係を形式化できます。以下の議論は、主にバニラRNNモデルに焦点を当てていますが、他の配列モデルにも当てはまります。 

タイムステップ $t$ にミニバッチ入力 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (例の数:$n$、各例の入力数:$d$) があるとします。同じタイムステップで、$l^\mathrm{th}$ 隠れ層 ($l=1,\ldots,L$) の隠れ状態を $\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$ (隠れユニット数:$h$) とし、出力層変数を $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (出力数:$q$) とします。$\mathbf{H}_t^{(0)} = \mathbf{X}_t$ を設定すると、アクティベーション関数 $\phi_l$ を使用する隠れ層 $l^\mathrm{th}$ の隠れ状態は次のように表されます。 

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

ここで、重み $\mathbf{W}_{xh}^{(l)} \in \mathbb{R}^{h \times h}$ と $\mathbf{W}_{hh}^{(l)} \in \mathbb{R}^{h \times h}$ は、バイアス $\mathbf{b}_h^{(l)} \in \mathbb{R}^{1 \times h}$ と共に $l^\mathrm{th}$ 隠れ層のモデルパラメーターです。 

最終的には、出力層の計算は、最終的な $L^\mathrm{th}$ 隠れ層の隠れ状態のみに基づいて行われます。 

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q,$$

重み $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ とバイアス $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ は出力層のモデルパラメーターです。 

MLP と同様に、隠れ層の数 $L$ と隠れユニットの数 $h$ はハイパーパラメータです。言い換えれば、私たちがチューニングしたり指定したりすることができます。さらに、:eqref:`eq_deep_rnn_H` の隠れ状態の計算を GRU または LSTM からの計算に置き換えることで、ディープゲート RNN を容易に得ることができます。 

## 簡潔な実装

幸いなことに、RNN の複数のレイヤーを実装するために必要なロジスティクスの詳細の多くは、高レベル API ですぐに利用できます。物事を単純にするために、このような組み込み関数を使用した実装のみを説明します。LSTM モデルを例に挙げてみましょう。このコードは、以前 :numref:`sec_lstm` で使用したものと非常によく似ています。実際、唯一の違いは、デフォルトの 1 つのレイヤーを選択するのではなく、レイヤーの数を明示的に指定することです。いつものように、まずデータセットをロードします。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

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

ハイパーパラメータの選択などのアーキテクチャ上の決定は :numref:`sec_lstm` の場合とよく似ています。個別のトークンと同じ数の入力と出力を選択します。つまり `vocab_size` です。隠れユニットの数は256のままです。唯一の違いは、(** `num_layers` の値を指定して、自明ではない数の非表示レイヤーを選択**) したことです。

```{.python .input}
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
device = d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
rnn_cells = [tf.keras.layers.LSTMCell(num_hiddens) for _ in range(num_layers)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = tf.keras.layers.RNN(stacked_lstm, time_major=True,
                                 return_sequences=True, return_state=True)
with strategy.scope():
    model = d2l.RNNModel(lstm_layer, len(vocab))
```

## [**トレーニング**] と予測

今度は LSTM モデルで 2 つの層をインスタンス化しているので、このやや複雑なアーキテクチャではトレーニングが大幅に遅くなります。

```{.python .input}
#@tab mxnet, pytorch
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## [概要

* ディープ RNN では、隠れ状態の情報はカレントレイヤの次のタイムステップと次のレイヤのカレントタイムステップに渡されます。
* ディープRNには、LSTM、GRU、バニラRNなど、さまざまなフレーバーが存在します。便利なことに、これらのモデルはすべて、ディープラーニングフレームワークの高レベル API の一部として利用できます。
* モデルの初期化には注意が必要です。全体として、ディープ RNN は適切な収束を確保するためにかなりの量の作業 (学習率やクリッピングなど) を必要とします。

## 演習

1. :numref:`sec_rnn_scratch` で説明したシングルレイヤ実装を使用して、2 レイヤ RNN をゼロから実装してみます。
2. LSTM を GRU に置き換えて、精度と学習速度を比較します。
3. トレーニングデータを増やして、複数の本を含めます。パープレキシティスケールでどれくらい低くできますか？
4. テキストをモデリングするときに、異なる作者のソースを結合したいですか？なぜこれが良いアイデアなのですか？何が悪くなる可能性がありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1058)
:end_tab:

#  シーケンスからシーケンスへの学習
:label:`sec_seq2seq`

:numref:`sec_machine_translation` で見てきたように、機械翻訳では入力と出力の両方が可変長のシーケンスです。この種の問題に対処するために、:numref:`sec_encoder-decoder` では一般的なエンコーダ/デコーダアーキテクチャを設計しました。このセクションでは、2 つの RNN を使用してこのアーキテクチャのエンコーダとデコーダを設計し、機械翻訳 :cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014` の *sequence to sequence * 学習に適用します。 

エンコーダ/デコーダアーキテクチャの設計原理に従い、RNN エンコーダは可変長シーケンスを入力として受け取り、それを固定形状の隠れ状態に変換することができます。つまり、入力 (ソース) シーケンスの情報は、RNN エンコーダの隠れた状態で*encoded* されます。トークンごとに出力シーケンストークンを生成するために、別のRNNデコーダが、入力シーケンスの符号化された情報とともに、どのトークン (言語モデリングなど) または生成されたトークンに基づいて、次のトークンを予測できます。:numref:`fig_seq2seq` は、シーケンス間で2つのRNNを使用する方法を示しています。機械翻訳の学習。 

![Sequence to sequence learning with an RNN encoder and an RNN decoder.](../img/seq2seq.svg)
:label:`fig_seq2seq`

:numref:`fig_seq2seq` では、特殊な <eos>"" トークンがシーケンスの終わりを示します。このトークンが生成されると、モデルは予測を停止できます。RNN デコーダの初期タイムステップでは、2 つの特別な設計上の決定があります。まず、特別なシーケンスの始まり "<bos>" トークンが入力です。次に、RNN エンコーダの最終的な隠れ状態を使用して、デコーダの隠れ状態を開始します。:cite:`Sutskever.Vinyals.Le.2014` などの設計では、このようにエンコードされた入力シーケンス情報がデコーダに供給され、出力 (ターゲット) シーケンスが生成されます。:cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014` などの他の設計では、:numref:`fig_seq2seq` に示すように、エンコーダの最終的な隠れ状態もすべてのタイムステップで入力の一部としてデコーダに供給されます。:numref:`sec_language_model` での言語モデルのトレーニングと同様に、ラベルを 1 つのトークンだけシフトしたオリジナルの出力シーケンスにすることができます。」「, <bos>「Ils」,「regardent」,「.」$\rightarrow$「Ils」,「regardent」,」。「," <eos>"。 

以下では、:numref:`fig_seq2seq` の設計について詳しく説明します。:numref:`sec_machine_translation` で導入された英仏データセットで、このモデルを機械翻訳用にトレーニングします。

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## エンコーダ

技術的には、エンコーダは可変長の入力シーケンスを固定形状の*context 変数* $\mathbf{c}$ に変換し、入力シーケンス情報をこのコンテキスト変数にエンコードします。:numref:`fig_seq2seq` に示すように、RNN を使用してエンコーダを設計できます。 

シーケンスの例 (バッチサイズ:1) を考えてみましょう。入力シーケンスが $x_1, \ldots, x_T$ で、$x_t$ が入力テキストシーケンスの $t^{\mathrm{th}}$ トークンであるとします。タイムステップ $t$ で、RNN は $x_t$ の入力特徴ベクトル $\mathbf{x}_t$、前のタイムステップの隠れ状態 $\mathbf{h} _{t-1}$ を現在の隠れ状態 $\mathbf{h}_t$ に変換します。関数 $f$ を使って、RNN のリカレント層の変換を表現できます。 

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

一般に、エンコーダは、カスタマイズされた関数 $q$ を使用して、すべてのタイムステップの隠れ状態をコンテキスト変数に変換します。 

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

たとえば、:numref:`fig_seq2seq` のように $q(\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$ を選択すると、コンテキスト変数は最後のタイムステップでの入力シーケンスの隠れ状態 $\mathbf{h}_T$ になります。 

これまで、単方向 RNN を使用してエンコーダを設計しました。隠れ状態は、隠れ状態のタイムステップとその前の入力部分シーケンスにのみ依存します。また、双方向 RNN を使用してエンコーダを構築することもできます。この場合、隠れ状態はタイムステップの前後のサブシーケンス (現在のタイムステップでの入力を含む) に依存し、シーケンス全体の情報がエンコードされます。 

それでは [**RNNエンコーダを実装**] してみましょう。*embedding layer* を使用して、入力シーケンスの各トークンの特徴ベクトルを取得することに注意してください。埋め込み層の重みは、行数が入力語彙のサイズ (`vocab_size`) に等しく、列数が特徴ベクトルの次元 (`embed_size`) に等しい行列です。入力トークンインデックス $i$ の場合、埋め込み層は重み行列の $i^{\mathrm{th}}$ 行 (0 から始まる) をフェッチして、その特徴ベクトルを返します。また、ここではエンコーダを実装するためにマルチレイヤGRUを選択します。

```{.python .input}
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        output, state = self.rnn(X, state)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab tensorflow
#@save
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs): 
        super().__init__(*kwargs)
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
    
    def call(self, X, *args, **kwargs):
        # The input `X` shape: (`batch_size`, `num_steps`)
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        output = self.rnn(X, **kwargs)
        state = output[1:]
        return output[0], state
```

リカレントレイヤの返される変数は :numref:`sec_rnn-concise` で説明されています。まだ具体的な例を挙げて [**上記のエンコーダの実装を例示します**] 以下では、隠れユニットの数が 16 の 2 層 GRU エンコーダをインスタンス化します。シーケンス入力のミニバッチ `X` (バッチサイズ:4、タイムステップ数:7) を指定すると、すべてのタイムステップにおける最後の層の隠れ状態 (符号化器のリカレント層から `output` が返される) は形状のテンソル (タイムステップ数、バッチサイズ、隠れ単位数) になります。

```{.python .input}
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = d2l.zeros((4, 7))
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab pytorch
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = d2l.zeros((4, 7), dtype=torch.long)
output, state = encoder(X)
output.shape
```

```{.python .input}
#@tab tensorflow
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
X = tf.zeros((4, 7))
output, state = encoder(X, training=False)
output.shape
```

ここでは GRU が使用されるため、最終タイムステップでの多層隠れ状態の形状は (隠れ層の数、バッチサイズ、隠れユニットの数) になります。LSTM を使用すると、メモリセル情報も `state` に格納されます。

```{.python .input}
len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
state.shape
```

```{.python .input}
#@tab tensorflow
len(state), [element.shape for element in state]
```

## [**デコーダ**]
:label:`sec_seq2seq_decoder`

先ほど述べたように、エンコーダの出力のコンテキスト変数 $\mathbf{c}$ は入力シーケンス $x_1, \ldots, x_T$ 全体をエンコードします。トレーニングデータセットの出力シーケンス $y_1, y_2, \ldots, y_{T'}$ が与えられた場合、タイムステップ $t'$ (シンボルは入力シーケンスまたはエンコーダーのタイムステップ $t$ とは異なる) ごとに、デコーダー出力 $y_{t'}$ の確率は前の出力サブシーケンス $y_1, \ldots, y_{t'-1}$ とコンテキスト変数によって条件付けられます。$\mathbf{c}$、つまり$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$。 

この条件付き確率をシーケンスでモデル化するために、別の RNN をデコーダーとして使用できます。出力シーケンスの任意のタイムステップ $t^\prime$ で、RNN は前のタイムステップからの出力 $y_{t^\prime-1}$ とコンテキスト変数 $\mathbf{c}$ を入力として取り、それらと前の隠れステート $\mathbf{s}_{t^\prime-1}$ を現在のタイムステップで隠れステート $\mathbf{s}_{t^\prime}$ に変換します。その結果、関数 $g$ を使用して、デコーダの隠れ層の変換を表現できます。 

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

復号化器の隠れ状態を取得した後、出力層と softmax 演算を使用して、タイムステップ $t^\prime$ での出力の条件付き確率分布 $P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c})$ を計算できます。 

:numref:`fig_seq2seq` 以降では、デコーダを次のように実装すると、エンコーダの最後のタイムステップで隠れ状態を直接使用して、デコーダの隠れ状態を初期化します。そのためには、RNN エンコーダと RNN デコーダのレイヤ数と隠れ単位が同じであることが必要です。符号化された入力シーケンス情報をさらに組み込むために、コンテキスト変数はすべてのタイムステップでデコーダ入力と連結される。出力トークンの確率分布を予測するために、全結合層を使用して、RNN デコーダの最終層で隠れ状態を変換します。

```{.python .input}
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).swapaxes(0, 1)
        # `context` shape: (`batch_size`, `num_hiddens`)
        context = state[0][-1]
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = np.broadcast_to(context, (
            X.shape[0], context.shape[0], context.shape[1]))
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).swapaxes(0, 1)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state[0]` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = d2l.concat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state
```

```{.python .input}
#@tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        
    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def call(self, X, state, **kwargs):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=X.shape[1], axis=1)
        X_and_context = tf.concat((X, context), axis=2)
        rnn_output = self.rnn(X_and_context, state, **kwargs)
        output = self.dense(rnn_output[0])
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` is a list with `num_layers` entries. Each entry has shape:
        # (`batch_size`, `num_hiddens`)
        return output, rnn_output[1:]
```

[**実装されたデコーダを説明する**] として、前述のエンコーダと同じハイパーパラメータでデコーダをインスタンス化します。ご覧のとおり、デコーダの出力形状は (バッチサイズ、タイムステップ数、語彙サイズ) になり、テンソルの最後の次元には予測されたトークン分布が格納されます。

```{.python .input}
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, len(state), state[0].shape
```

```{.python .input}
#@tab pytorch
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
```

```{.python .input}
#@tab tensorflow
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
state = decoder.init_state(encoder(X))
output, state = decoder(X, state, training=False)
output.shape, len(state), state[0].shape
```

要約すると、上記の RNN エンコーダ/デコーダモデルのレイヤは :numref:`fig_seq2seq_details` に示されています。 

![Layers in an RNN encoder-decoder model.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

## 損失関数

各タイムステップで、デコーダは出力トークンの確率分布を予測します。言語モデリングと同様に、softmax を適用して分布を取得し、最適化のためのクロスエントロピー損失を計算できます。:numref:`sec_machine_translation` を思い出してください。特別なパディングトークンがシーケンスの最後に追加されるため、長さの異なるシーケンスを同じ形状のミニバッチに効率的にロードできます。ただし、パディングトークンの予測は損失計算から除外する必要があります。 

この目的のために、次の `sequence_mask` 関数を使用して [**無関係なエントリをゼロ値でマスク**] し、無関係な予測をゼロで乗算するとゼロになります。たとえば、パディングトークンを除く 2 つのシーケンスの有効な長さがそれぞれ 1 と 2 の場合、最初の 1 つと最初の 2 つのエントリの後の残りのエントリは 0 にクリアされます。

```{.python .input}
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

```{.python .input}
#@tab pytorch
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
```

```{.python .input}
#@tab tensorflow
#@save
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
        None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)
    
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)
    
X = tf.constant([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, tf.constant([1, 2]))
```

(**最後の数軸にまたがるすべてのエントリをマスクすることもできます**) 必要であれば、そのようなエントリをゼロ以外の値で置き換えるよう指定することもできます。

```{.python .input}
X = d2l.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

```{.python .input}
#@tab pytorch
X = d2l.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2,3,4))
sequence_mask(X, tf.constant([1, 2]), value=-1)
```

これで、[**ソフトマックスのクロスエントロピー損失を拡張して、無関係な予測のマスキングを可能にします**] 最初は、予測されるすべてのトークンのマスクが 1 に設定されます。有効な長さが指定されると、パディングトークンに対応するマスクはゼロにクリアされます。最終的には、すべてのトークンの損失にマスクを掛けて、損失に含まれるパディングトークンの無関係な予測を除外します。

```{.python .input}
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        # `weights` shape: (`batch_size`, `num_steps`, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

```{.python .input}
#@tab pytorch
#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
```

```{.python .input}
#@tab tensorflow
#@save
class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """The softmax cross-entropy loss with masks."""
    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len
    
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def call(self, label, pred):
        weights = tf.ones_like(label, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss
```

[**健全性チェック**] では、3 つの同一のシーケンスを作成できます。次に、これらのシーケンスの有効な長さがそれぞれ 4、2、0 であることを指定できます。その結果、最初のシーケンスの損失は 2 番目のシーケンスの損失の 2 倍になり、3 番目のシーケンスの損失はゼロになります。

```{.python .input}
loss = MaskedSoftmaxCELoss()
loss(d2l.ones((3, 4, 10)), d2l.ones((3, 4)), np.array([4, 2, 0]))
```

```{.python .input}
#@tab pytorch
loss = MaskedSoftmaxCELoss()
loss(d2l.ones(3, 4, 10), d2l.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
```

```{.python .input}
#@tab tensorflow
loss = MaskedSoftmaxCELoss(tf.constant([4, 2, 0]))
loss(tf.ones((3,4), dtype = tf.int32), tf.ones((3, 4, 10))).numpy()
```

## [**トレーニング**]
:label:`sec_seq2seq_training`

次のトレーニングループでは、:numref:`fig_seq2seq` に示すように、特別なシーケンスの開始トークンと、最後のトークンを除いた元の出力シーケンスをデコーダーへの入力として連結します。元の出力シーケンス (トークンラベル) がデコーダに供給されるため、これを*teacher forcing* と呼びます。あるいは、前のタイムステップの*predicted* トークンを現在の入力としてデコーダにフィードすることもできます。

```{.python .input}
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    net.initialize(init.Xavier(), force_reinit=True, ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [
                x.as_in_ctx(device) for x in batch]
            bos = np.array(
                [tgt_vocab['<bos>']] * Y.shape[0], ctx=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with autograd.record():
                Y_hat, _ = net(X, dec_input, X_valid_len)
                l = loss(Y_hat, Y, Y_valid_len)
            l.backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

```{.python .input}
#@tab tensorflow
#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss",
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]),
                             shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training=True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = d2l.grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')
```

これで、機械翻訳データセットでシーケンス間学習を行うために [**RNNエンコーダ/デコーダモデルを作成してトレーニング**] できるようになりました。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

## [**予測**]

トークンごとに出力シーケンストークンを予測するために、各デコーダタイムステップで、前のタイムステップから予測されたトークンが入力としてデコーダに供給されます。トレーニングと同様に、初期タイムステップで、シーケンスの開始 (」<bos>「) トークンがデコーダに供給されます。この予測プロセスは :numref:`fig_seq2seq_predict` で説明されています。シーケンスの終わり (」<eos>「) トークンが予測されると、出力シーケンスの予測は完了です。 

![Predicting the output sequence token by token using an RNN encoder-decoder.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

:numref:`sec_beam-search` では、配列生成のためのさまざまな戦略を紹介します。

```{.python .input}
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = np.array([len(src_tokens)], ctx=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = np.expand_dims(np.array(src_tokens, ctx=device), axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=device), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(axis=2)
        pred = dec_X.squeeze(axis=0).astype('int32').item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab pytorch
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

```{.python .input}
#@tab tensorflow
#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    save_attention_weights=False):
    """Predict for sequence to sequence."""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = tf.expand_dims(src_tokens, axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state, training=False)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        # Save attention weights
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    return ' '.join(tgt_vocab.to_tokens(tf.reshape(output_seq, shape = -1).numpy().tolist())), attention_weight_seq
```

## 予測された系列の評価

予測されたシーケンスをラベルシーケンス (グラウンドトゥルース) と比較することで評価できます。BLEU (バイリンガル評価アンダースタディ) は、もともと機械翻訳の結果を評価するために提案されましたが :cite:`Papineni.Roukos.Ward.ea.2002`、さまざまなアプリケーションの出力シーケンスの品質測定に広く使用されています。原則として、予測されたシーケンスの $n$ グラムについて、BLEU はこの $n$ グラムがラベルシーケンスに含まれるかどうかを評価します。 

$n$ グラムの精度を $n$ グラムで表します。これは、予測シーケンスとラベルシーケンスで一致した $n$ グラムの数と、予測されたシーケンスの $n$ グラムの数との比率です。説明すると、ラベルシーケンス $A$、$B$、$C$、$D$、$E$、$F$、および予測されたシーケンス $A$、$B$、$B$、$C$、$D$、$p_1 = 4/5$、$p_2 = 3/4$、$p_2 = 3/4$、$p_2 = 3/4$、$D$ 3617、$p_4 = 0$ です。また、$\mathrm{len}_{\text{label}}$ と $\mathrm{len}_{\text{pred}}$ をそれぞれラベルシーケンスと予測シーケンスのトークンの数とします。その場合、BLEU は次のように定義されます。 

$$ \exp\left(\min\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

$k$ はマッチングで最長の $n$ グラムです。 

:eqref:`eq_bleu` の BLEU の定義に基づき、予測されるシーケンスがラベルシーケンスと同じであれば、BLEU は 1 になります。さらに、$n$ グラムが長いほどマッチングが難しいため、BLEU は長い $n$ グラムの精度により大きな重みを割り当てます。具体的には、$p_n$ を固定すると $n$ が大きくなるにつれて $p_n^{1/2^n}$ が増加します (元の用紙では $p_n^{1/n}$ が使用されています)。さらに、短いシーケンスを予測すると $p_n$ の値が高くなる傾向があるため、:eqref:`eq_bleu` の乗算項の前の係数は、予測される短いシーケンスにペナルティを課します。たとえば、$k=2$ の場合、ラベルシーケンス $A$、$B$、$C$、$D$、$E$、$F$、および予測されるシーケンス $A$、$B$ を指定すると、$p_1 = p_2 = 1$ ではペナルティ係数 $\exp(1-6/2) \approx 0.14$ によって BLEU が低下します。 

以下のように [**BLEU対策を実施**] します。

```{.python .input}
#@tab all
def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

最後に、学習済みの RNN エンコーダー/デコーダーを使用して [**英文をフランス語に翻訳**] し、結果の BLEU を計算します。

```{.python .input}
#@tab mxnet, pytorch
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

```{.python .input}
#@tab tensorflow
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
```

## [概要

* エンコーダ/デコーダアーキテクチャの設計に従い、2 つの RNN を使用して、シーケンス間学習のモデルを設計できます。
* エンコーダとデコーダを実装する場合、マルチレイヤRNNを使用できます。
* マスクを使用して、損失を計算するときなど、無関係な計算を除外できます。
* 符号化器/復号化器の学習では、教師強制アプローチは (予測とは対照的に) 元の出力シーケンスを復号器に供給します。
* BLEU は、予測されたシーケンスとラベルシーケンスの間で $n$ グラムを照合することにより、出力シーケンスを評価するための一般的な尺度です。

## 演習

1. ハイパーパラメータを調整してトランスレーション結果を改善できますか。
1. 損失計算でマスクを使用せずに実験を再実行します。どのような結果が見られますか？なぜ？
1. エンコーダとデコーダのレイヤ数や隠れユニットの数が異なる場合、デコーダの隠れ状態を初期化するにはどうすればよいでしょうか？
1. 学習では、教師の強制を、前のタイムステップでの予測をデコーダーに入力することで置き換えます。これはパフォーマンスにどのような影響を与えますか？
1. GRU を LSTM に置き換えて実験を再実行します。
1. デコーダの出力層を設計する他の方法はありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1062)
:end_tab:

# ゲーテッドリカレントユニット (GRU)
:label:`sec_gru`

:numref:`sec_bptt` では、RNN における勾配の計算方法について説明しました。特に、行列の長い積は勾配の消失や爆発を引き起こす可能性があることを発見しました。このような勾配異常が実際に何を意味するのかを簡単に考えてみましょう。 

* 将来のすべての観測を予測する上で、早期観測が非常に重要な状況に遭遇する可能性があります。最初の観測値にチェックサムが含まれていて、シーケンスの最後にチェックサムが正しいかどうかを見分けることが目的である、やや不自然なケースを考えてみましょう。この場合、最初のトークンの影響は非常に重要です。重要な初期情報を*メモリセル*に保存する仕組みをいくつか用意したいと考えています。このようなメカニズムがなければ、後続のすべての観測に影響するため、この観測値に非常に大きな勾配を割り当てる必要があります。
* 一部のトークンが適切な観測値を持たない状況に遭遇する可能性があります。たとえば、Web ページを解析するときに、ページ上で伝えられる感情を評価する目的には無関係な補助 HTML コードが存在する場合があります。そのようなトークンを潜在状態表現で*スキップ*するためのメカニズムが欲しいです。
* シーケンスの各部分の間に論理的な区切りがある状況に遭遇することがあります。たとえば、本の章間の遷移や、証券の弱気相場と強気相場の間の移行があるかもしれません。この場合、内部の状態表現を「リセット」する手段があるといいでしょう。

これに対処するために、いくつかの方法が提案されている。最も早いものの一つは長期短期記憶 :cite:`Hochreiter.Schmidhuber.1997` です。これについては :numref:`sec_lstm` で論じます。ゲートリカレントユニット (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` は、やや合理化されたバリアントで、多くの場合、同等のパフォーマンスを提供し、:cite:`Chung.Gulcehre.Cho.ea.2014` の計算速度が大幅に向上します。そのシンプルさから、GRU から始めましょう。 

## ゲート付き隠れ状態

バニラRNとGRUの主な違いは、後者が隠れ状態のゲーティングをサポートしていることです。これは、隠れた状態がいつ*更新*され、いつ*リセット*されるべきかについて、専用のメカニズムがあることを意味します。これらのメカニズムは学習され、上記の懸念に対処します。例えば、最初のトークンが非常に重要な場合、最初の観測の後に隠れ状態を更新しないよう学習します。同様に、無関係な一時的な観察をスキップする方法を学びます。最後に、必要なときに潜伏状態をリセットする方法を学習します。これについては、以下で詳しく説明します。 

### ゲートのリセットとゲートの更新

最初に導入する必要があるのは、*リセットゲート* と*アップデートゲート* です。凸の組み合わせを実行できるように $(0, 1)$ のエントリをもつベクトルになるように設計します。例えば、リセットゲートを使うと、以前の状態をどれだけ記憶しておきたいかをコントロールできます。同様に、更新ゲートによって、新しい状態のどれだけが古い状態のコピーになるかを制御できます。 

まず、これらのゲートを設計します。:numref:`fig_gru_1` は、現在のタイムステップの入力と前のタイムステップの隠れ状態を考慮して、GRU のリセットゲートと更新ゲートの両方の入力を示しています。2 つのゲートの出力は、シグモイド活性化関数をもつ 2 つの全結合層によって与えられます。 

![Computing the reset gate and the update gate in a GRU model.](../img/gru-1.svg)
:label:`fig_gru_1`

数学的には、所定のタイムステップ $t$ について、入力がミニバッチ $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (例の数:$n$、入力数:$d$) で、前のタイムステップの隠れ状態が $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (隠れユニット数:$h$) であるとします。次に、リセットゲート $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ とアップデートゲート $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ は次のように計算されます。 

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

$\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ と $\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ は重みパラメータで、$\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ はバイアスです。ブロードキャスト (:numref:`subsec_broadcasting` を参照) は合計中にトリガーされることに注意してください。:numref:`sec_mlp` で導入されたシグモイド関数を使用して、入力値を $(0, 1)$ の間隔に変換します。 

### 候補の隠れ状態

次に、リセットゲート $\mathbf{R}_t$ を :eqref:`rnn_h_with_state` の通常の潜在状態更新メカニズムと統合します。それは次のことにつながる
*候補隠れ状態*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$、タイムステップ $t$ で次のようになります。 

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ と $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ は重みパラメータ、$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ はバイアス、シンボル $\odot$ はアダマール (要素単位) 積演算子です。ここでは tanh の形式で非線形性を使用して、候補の隠れ状態の値が区間 $(-1, 1)$ に留まるようにします。 

更新ゲートのアクションを組み込む必要があるため、結果は*候補*になります。:eqref:`rnn_h_with_state` と比較すると、:eqref:`gru_tilde_H` の $\mathbf{R}_t$ と $\mathbf{H}_{t-1}$ の要素ごとの乗算によって、以前の状態の影響を小さくすることができます。リセットゲート $\mathbf{R}_t$ のエントリが 1 に近づくと、:eqref:`rnn_h_with_state` のようにバニラ RNN が回復します。リセットゲート $\mathbf{R}_t$ のすべてのエントリで 0 に近い場合、隠れ状態の候補は、$\mathbf{X}_t$ を入力とする MLP の結果です。したがって、既存の隠れ状態はデフォルトに*リセット* されます。 

:numref:`fig_gru_2` は、リセットゲートを適用した後の計算フローを示しています。 

![Computing the candidate hidden state in a GRU model.](../img/gru-2.svg)
:label:`fig_gru_2`

### 隠れ状態

最後に、アップデートゲート $\mathbf{Z}_t$ の効果を組み込む必要があります。これにより、新しい非表示ステート $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ が古いステート $\mathbf{H}_{t-1}$ に過ぎず、新しい候補ステート $\tilde{\mathbf{H}}_t$ がどれだけ使用されるかが決まります。更新ゲート $\mathbf{Z}_t$ は、$\mathbf{H}_{t-1}$ と $\tilde{\mathbf{H}}_t$ の両方を要素ごとに凸状に組み合わせるだけで、この目的に使用できます。これにより、GRU の最終的な更新方程式が導かれます。 

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

アップデートゲート $\mathbf{Z}_t$ が 1 に近づくと、古い状態が保持されます。この場合、$\mathbf{X}_t$ からの情報は本質的に無視され、依存関係チェーン内のタイムステップ $t$ は事実上スキップされます。対照的に、$\mathbf{Z}_t$ が 0 に近づくと、新しい潜在状態 $\mathbf{H}_t$ は候補の潜在状態 $\tilde{\mathbf{H}}_t$ に近づきます。これらの設計は、RNN の消失勾配問題に対処し、時間ステップ距離が大きいシーケンスの依存関係をより適切に捕捉するのに役立ちます。たとえば、サブシーケンス全体のすべてのタイムステップで更新ゲートが 1 に近い場合、サブシーケンスの長さに関係なく、最初のタイムステップの古い隠れ状態は簡単に保持され、最後に渡されます。 

:numref:`fig_gru_3` は、更新ゲートが動作した後の計算フローを示しています。 

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`

要約すると、GRU には次の 2 つの特徴があります。 

* リセットゲートは、短期間の依存関係を順番に捉えるのに役立ちます。
* 更新ゲートは、シーケンス内の長期的な依存関係を把握するのに役立ちます。

## ゼロからの実装

GRU モデルの理解を深めるために、GRU モデルをゼロから実装してみましょう。まず、:numref:`sec_rnn_scratch` で使用したタイムマシンデータセットを読み取ります。データセットを読み取るためのコードを以下に示す。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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

### (**モデルパラメーターの初期化**)

次のステップでは、モデルパラメーターを初期化します。標準偏差が 0.01 のガウス分布から重みを引き出し、バイアスを 0 に設定します。ハイパーパラメータ `num_hiddens` は、非表示ユニットの数を定義します。更新ゲート、リセットゲート、候補隠れ状態、および出力層に関連するすべての重みとバイアスをインスタンス化します。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input}
#@tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    def three():
        return (tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32),
                tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32),
                tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

### モデルを定義する

ここで [**隠れ状態初期化関数**] `init_gru_state` を定義します。:numref:`sec_rnn_scratch` で定義されている関数 `init_rnn_state` と同様に、この関数は値がすべてゼロのシェイプ (バッチサイズ、隠れユニットの数) を持つテンソルを返します。

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_gru_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

これで [**GRU モデルを定義する**] の準備が整いました。その構造は、更新方程式がより複雑であることを除いて、基本的な RNN セルの構造と同じです。

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)
```

### トレーニングと予測

[**トレーニング**] と予測は :numref:`sec_rnn_scratch` とまったく同じように機能します。学習後、学習セットのパープレキシティと、指定された接頭辞「time traveller」と「traveller」にそれぞれ続く予測シーケンスを出力します。

```{.python .input}
#@tab mxnet, pytorch
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
vocab_size, num_hiddens, device_name = len(vocab), 256, d2l.try_gpu()._device_name
# defining tensorflow training strategy
strategy = tf.distribute.OneDeviceStrategy(device_name)
num_epochs, lr = 500, 1
with strategy.scope():
    model = d2l.RNNModelScratch(len(vocab), num_hiddens, init_gru_state, gru, get_params)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## [**簡潔な実装**]

高レベル API では、GPU モデルを直接インスタンス化できます。これにより、上記で明示した設定の詳細がすべてカプセル化されます。このコードは、前に説明した多くの詳細に Python よりもコンパイルされた演算子を使用するため、非常に高速です。

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab tensorflow
gru_cell = tf.keras.layers.GRUCell(num_hiddens,
    kernel_initializer='glorot_uniform')
gru_layer = tf.keras.layers.RNN(gru_cell, time_major=True,
    return_sequences=True, return_state=True)

device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)
with strategy.scope():
    model = d2l.RNNModel(gru_layer, vocab_size=len(vocab))

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, strategy)
```

## [概要

* ゲート RNN は、時間ステップの距離が大きいシーケンスの依存関係をより適切に捕捉できます。
* リセットゲートは、短期間の依存関係を順番に捉えるのに役立ちます。
* 更新ゲートは、シーケンス内の長期的な依存関係を把握するのに役立ちます。
* GRU には、リセットゲートがスイッチオンされるたびに極端なケースとして、基本的な RNN が含まれています。また、更新ゲートをオンにすることで、サブシーケンスをスキップすることもできます。

## 演習

1. タイムステップ $t > t'$ での出力を予測するために、タイムステップ $t'$ の入力のみを使用すると仮定します。各タイムステップのリセットゲートと更新ゲートの最適な値は何ですか。
1. ハイパーパラメータを調整し、実行時間、パープレキシティ、および出力シーケンスに及ぼす影響を解析します。
1. `rnn.RNN` と `rnn.GRU` の実装のランタイム、パープレキシティ、および出力文字列を相互に比較します。
1. リセットゲートのみ、更新ゲートだけなど、GRU の一部のみを実装するとどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:

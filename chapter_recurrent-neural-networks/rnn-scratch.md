# リカレントニューラルネットワークのゼロからの実装
:label:`sec_rnn_scratch`

このセクションでは、:numref:`sec_rnn` の説明に従って、文字レベルの言語モデル用に RNN をゼロから実装します。そのようなモデルはH.G.Wellsの*The Time Machine*でトレーニングされます。先ほどと同様に、:numref:`sec_language_model` で導入されたデータセットを最初に読み込むことから始めます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab tensorflow
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(
    batch_size, num_steps, use_random_iter=True)
```

## [**ワンホットエンコーディング**]

`train_iter` では、各トークンは数値インデックスで表されることを思い出してください。これらのインデックスをニューラルネットワークに直接供給すると、学習が難しくなる可能性があります。多くの場合、各トークンをより表現力豊かな特徴ベクトルとして表現します。最も簡単な表現は*ワンホットエンコーディング* と呼ばれ、:numref:`subsec_classification-problem` で導入されました。 

簡単に言うと、各インデックスを異なる単位ベクトルにマッピングします。ボキャブラリ内の異なるトークンの数が $N$ (`len(vocab)`) で、トークンインデックスの範囲が $0$ から $N-1$ であると仮定します。トークンのインデックスが整数 $i$ の場合、長さが $N$ のすべて 0 のベクトルを作成し、$i$ の位置にある要素を 1 に設定します。このベクトルは、元のトークンのワンホットベクトルです。インデックスが 0 と 2 のワンホットベクトルを以下に示します。

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

毎回サンプリングする (**ミニバッチの形状**) (**は (バッチサイズ、タイムステップ数) です。`one_hot` 関数は、このようなミニバッチを語彙サイズ (`len(vocab)`) と等しい 3 次元テンソルに変換します。**) 入力を転置して、形状の出力 (タイムステップ数、バッチサイズ、ボキャブラリサイズ) を得ることがよくあります。これにより、ミニバッチの隠れ状態をタイムステップごとに更新するために、最も外側の次元をより便利にループできます。

```{.python .input}
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

## モデルパラメーターの初期化

次に、[**RNN モデルのモデルパラメーターを初期化します**]。隠れユニットの数 `num_hiddens` は調整可能なハイパーパラメーターです。言語モデルをトレーニングする場合、インプットとアウトプットは同じボキャブラリから取得されます。したがって、それらは同じ次元を持ち、語彙の大きさに等しいです。

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
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

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

## RNN モデル

RNN モデルを定義するには、まず [**初期化時に隠れ状態を返す `init_rnn_state` 関数**] が必要です。この関数は、0 で埋められ、(バッチサイズ、隠れ単位数) の形をしたテンソルを返します。タプルを使うと、隠れ状態に複数の変数が含まれる状況に対処しやすくなります。これについては後のセクションで説明します。

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

[**次の `rnn` 関数は、隠れ状態の計算方法とタイムステップでの出力を定義しています。**] RNN モデルは `inputs` の最外次元をループし、ミニバッチの隠れ状態 `H` をタイムステップごとに更新することに注意してください。また、ここでのアクティベーション関数では $\tanh$ 関数を使用しています。:numref:`sec_mlp` で説明したように、要素が実数にわたって一様に分布している場合、$\tanh$ 関数の平均値は 0 になります。

```{.python .input}
def rnn(inputs, state, params):
    # Shape of `inputs`: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input}
#@tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

必要な関数をすべて定義したら、次に [**これらの関数をラップしてパラメーターを格納するクラスを作成します**]。RNN モデルをゼロから実装します。

```{.python .input}
class RNNModelScratch:  #@save
    """An RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)
```

たとえば、隠れ状態の次元が変わらないことを確認するために、[**出力の形状が正しいかどうかを確認する**] とします。

```{.python .input}
#@tab mxnet
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab tensorflow
# defining tensorflow training strategy
device_name = d2l.try_gpu()._device_name
strategy = tf.distribute.OneDeviceStrategy(device_name)

num_hiddens = 512
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
state = net.begin_state(X.shape[0])
Y, new_state = net(X, state)
Y.shape, len(new_state), new_state[0].shape
```

出力形状は (タイムステップ数 $\times$ バッチサイズ、語彙サイズ) ですが、隠れ状態の形状は同じ (バッチサイズ、隠れ単位数) のままであることがわかります。 

## 予測

[**最初に、ユーザーが提供した `prefix`**] に続いて新しい文字を生成する予測関数を定義してみましょう。これは複数の文字を含む文字列です。`prefix` でこれらの開始文字をループすると、出力を生成せずに隠れ状態を次のタイムステップに渡し続けます。これは*ウォームアップ* 期間と呼ばれ、モデル自体が更新される (隠れ状態の更新など) が、予測は行わない期間です。ウォームアップ期間が過ぎると、隠れ状態は一般に開始時の初期化値よりも良くなります。そこで、予測された文字を生成して放出します。

```{.python .input}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

これで `predict_ch8` 関数をテストできます。プレフィックスを `time traveller ` と指定し、10 個の追加文字を生成させます。ネットワークに学習をさせていないと、無意味な予測が生成されます。

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
predict_ch8('time traveller ', 10, net, vocab)
```

## [**グラデーションクリッピング**]

長さ $T$ のシーケンスの場合、この $T$ タイムステップの勾配を反復計算で計算します。その結果、逆伝播時に長さ $\mathcal{O}(T)$ の行列積のチェーンが生成されます。:numref:`sec_numerical_stability` で述べたように、$T$ が大きいと勾配が爆発したり消失したりするなど、数値が不安定になる可能性があります。したがって、RNN モデルはトレーニングを安定させるために追加の支援を必要とすることがよくあります。 

一般に、最適化問題を解く場合、モデルパラメーター (ベクトル形式 $\mathbf{x}$ など) に対して、ミニバッチ上の負の勾配 $\mathbf{g}$ の方向に更新ステップを実行します。たとえば、$\eta > 0$ を学習率として、1 回の反復で $\mathbf{x}$ を $\mathbf{x} - \eta \mathbf{g}$ として更新します。さらに、目的関数 $f$ が定数 $L$ をもつ*リップシッツ連続* のように適切に動作すると仮定します。つまり、どの$\mathbf{x}$と$\mathbf{y}$についても、 

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

この場合、パラメータベクトルを $\eta \mathbf{g}$ で更新すると、 

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

つまり、$L \eta \|\mathbf{g}\|$ を超える変化は観測されないということです。これは呪いであり祝福でもある。呪いの面では、進歩のスピードを制限します。一方、祝福の側では、私たちが間違った方向に進んだ場合に物事がうまくいかない程度を制限します。 

勾配が非常に大きくなり、最適化アルゴリズムが収束しないことがあります。この問題は、学習率 $\eta$ を下げることで対処できます。しかし、大きなグラデーションが*まれにしか得られない場合はどうなるでしょうか？この場合、そのようなアプローチは完全に不当に見えるかもしれません。一般的な代替方法の 1 つとして、グラデーション $\mathbf{g}$ を指定の半径のボール (たとえば $\theta$) に投影してクリップする方法があります。 

(** $\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$ドル**) 

これにより、勾配ノルムが $\theta$ を超えることはなく、更新された勾配は元の方向 $\mathbf{g}$ に完全に揃っていることがわかります。また、任意のミニバッチ (およびその中の任意のサンプル) がパラメーターベクトルに及ぼす影響を制限するという望ましい副作用もあります。これにより、モデルにある程度のロバスト性が与えられます。グラデーションクリッピングは、グラデーションの爆発を素早く修正します。問題を完全に解決するわけではありませんが、問題を軽減するための多くの手法の1つです。 

以下では、ゼロから実装されたモデル、または高レベル API で構築されたモデルの勾配をクリップする関数を定義します。また、すべてのモデルパラメーターに対して勾配ノルムを計算することにも注意してください。

```{.python .input}
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, gluon.Block):
        params = [p.data() for p in net.collect_params().values()]
    else:
        params = net.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab tensorflow
def grad_clipping(grads, theta):  #@save
    """Clip the gradient."""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad
```

## 訓練

モデルを学習させる前に、[**モデルを一エポックで学習させる関数を定義する**] とします。:numref:`sec_softmax_scratch` のモデルを 3 か所でトレーニングする方法とは異なります。 

1. シーケンシャルデータのサンプリング方法 (ランダムサンプリングとシーケンシャルパーティショニング) が異なると、隠れ状態の初期化に違いが生じます。
1. モデルパラメーターを更新する前に、グラデーションをクリップします。これにより、学習プロセス中のある時点で勾配が膨らんだ場合でも、モデルが発散しないことが保証されます。
1. パープレキシティを使用してモデルを評価します。:numref:`subsec_perplexity` で説明したように、これによって長さの異なるシーケンスが比較可能になります。

具体的には、シーケンシャル・パーティショニングを使用する場合は、各エポックの開始時にのみ隠れ状態を初期化します。次のミニバッチの $i^\mathrm{th}$ サブシーケンスの例は、現在の $i^\mathrm{th}$ サブシーケンスの例に隣接しているため、現在のミニバッチの最後の隠れ状態を使用して、次のミニバッチの開始時に隠れ状態が初期化されます。このようにして、隠れた状態で格納されたシーケンスの履歴情報が、エポック内の隣接するサブシーケンスに渡って流れる可能性があります。ただし、隠れ状態の計算は、同じエポック内の以前のすべてのミニバッチに依存するため、勾配計算が複雑になります。計算コストを削減するために、隠れ状態の勾配計算が常に 1 つのミニバッチのタイムステップに制限されるように、ミニバッチを処理する前に勾配をデタッチします。  

ランダムサンプリングを使用する場合、各サンプルはランダムな位置でサンプリングされるため、反復ごとに隠れ状態を再初期化する必要があります。:numref:`sec_softmax_scratch` の `train_epoch_ch3` 関数と同様に、`updater` はモデルパラメーターを更新する汎用関数です。これは、最初から実装された `d2l.sgd` 関数か、ディープラーニングフレームワークに組み込まれた最適化関数のいずれかです。

```{.python .input}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater, use_random_iter):
    """Train a model within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))

        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**トレーニング関数は、ゼロから、または高レベル API を使用して実装された RNN モデルをサポートします。**]

```{.python .input}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

[**これでRNNモデルをトレーニングできるようになりました**] データセットでは10000個のトークンしか使用しないため、より適切に収束するためにはモデルにより多くのエポックが必要です。

```{.python .input}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

[**最後に、ランダムサンプリング法を使用した結果を確認しましょう**]

```{.python .input}
#@tab mxnet,pytorch
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
with strategy.scope():
    net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                          get_params)
train_ch8(net, train_iter, vocab_random_iter, lr, num_epochs, strategy,
          use_random_iter=True)
```

上記の RNN モデルをゼロから実装することは有益ですが、便利ではありません。次のセクションでは、実装を容易にし、実行を高速化する方法など、RNN モデルを改善する方法について説明します。 

## [概要

* RNN ベースの文字レベル言語モデルをトレーニングして、ユーザーが指定したテキストプレフィックスに続くテキストを生成できます。
* 単純な RNN 言語モデルは、入力エンコーディング、RNN モデリング、および出力生成から構成されます。
* RNN モデルは学習のために状態の初期化が必要ですが、ランダムサンプリングとシーケンシャルパーティショニングでは異なる方法を使用します。
* シーケンシャル・パーティショニングを使用する場合、計算コストを削減するために勾配をデタッチする必要があります。
* ウォームアップ期間により、予測を行う前にモデルが自身を更新する (たとえば、初期化された値よりも優れた隠れ状態を取得する) ことができます。
* グラデーションクリッピングはグラデーションの爆発を防ぎますが、消えていくグラデーションは修正できません。

## 演習

1. ワンホットエンコーディングは、オブジェクトごとに異なる埋め込みを選択することに相当することを示します。
1. ハイパーパラメーター (エポック数、隠れユニットの数、ミニバッチのタイムステップ数、学習率など) を調整して、パープレキシティを改善します。
    * どれくらい低く行けますか？
    * ワンホットエンコーディングを学習可能な埋め込みに置き換えます。これはパフォーマンスの向上につながりますか？
    * H.G. ウェルズの他の本、例えば [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36) ではどの程度うまく機能しますか？
1. 予測関数を変更して、最も可能性の高い次の文字を選択するのではなく、サンプリングを使用するようにします。
    * 何が起きる？
    * $\alpha > 1$ の $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ からサンプリングするなどして、より可能性の高い出力に向けてモデルをバイアスします。
1. グラデーションをクリッピングせずに、このセクションのコードを実行します。何が起きる？
1. 隠れ状態を計算グラフから分離しないように、逐次分割を変更します。ランニングタイムは変わりますか？困惑はどう？
1. このセクションで使用されているアクティベーション関数を ReLU に置き換え、このセクションの実験を繰り返します。グラデーションクリッピングはまだ必要ですか？なぜ？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:

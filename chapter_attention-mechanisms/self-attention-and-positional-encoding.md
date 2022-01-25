# 自己注意と位置エンコーディング
:label:`sec_self-attention-and-positional-encoding`

ディープラーニングでは、CNN または RNN を使用してシーケンスをエンコードすることがよくあります。ここで、アテンション・メカニズムを使用して、一連のトークンをアテンション・プーリングに送り、同じトークンのセットがクエリ、キー、および値として機能するようにしたとします。具体的には、各クエリはすべてのキーと値のペアを処理し、アテンション出力を 1 つ生成します。クエリ、キー、および値は同じ場所から取得されるため、
*セルフアテンション* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`。これは*イントラアテンション* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`とも呼ばれます。
このセクションでは、シーケンスの順序に関する追加情報の使用など、セルフアテンションを使用したシーケンスエンコーディングについて説明します。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

## [**自己注意**]

$\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$) のセルフアテンションが同じ長さ $\mathbf{y}_1, \ldots, \mathbf{y}_n$ のシーケンスを出力する一連の入力トークン $\mathbf{x}_1, \ldots, \mathbf{x}_n$ が与えられた場合、 

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

:eqref:`eq_attn-pooling`のアテンション・プーリング$f$の定義によります次のコードスニペットは、マルチヘッドアテンションを使用して、形状 (バッチサイズ、タイムステップ数、またはトークン単位のシーケンス長、$d$) を持つテンソルの自己注意を計算します。出力テンソルの形状は同じです。

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
#@tab mxnet, pytorch
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

```{.python .input}
#@tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens, training=False).shape
```

## CNN、RNN、およびセルフアテンションを比較する
:label:`subsec_cnn-rnn-self-attention`

$n$ トークンのシーケンスを、長さが等しい別のシーケンスにマッピングするアーキテクチャを比較してみましょう。この場合、各入力または出力トークンは $d$ 次元のベクトルで表されます。具体的には、CNN、RNN、自己注意について検討します。計算の複雑さ、逐次演算、および最大経路長を比較します。シーケンシャル演算は並列計算を妨げますが、シーケンス位置の任意の組み合わせ間のパスが短くなると、シーケンス :cite:`Hochreiter.Bengio.Frasconi.ea.2001` 内の長距離依存関係を学習しやすくなります。 

![Comparing CNN (padding tokens are omitted), RNN, and self-attention architectures.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

カーネルサイズが $k$ の畳み込み層を考えてみましょう。CNN を使用したシーケンス処理の詳細については、後の章で説明します。ここでは、シーケンスの長さが $n$ で、入力チャネル数と出力チャネル数がどちらも $d$、畳み込み層の計算複雑度は $\mathcal{O}(knd^2)$ であることを知っておく必要があります。:numref:`fig_cnn-rnn-self-attention` が示すように、CNN は階層型であるため、$\mathcal{O}(1)$ のシーケンシャルオペレーションがあり、最大パス長は $\mathcal{O}(n/k)$ です。たとえば $\mathbf{x}_1$ と $\mathbf{x}_5$ は、:numref:`fig_cnn-rnn-self-attention` ではカーネルサイズが 3 の 2 レイヤ CNN の受容フィールド内にあります。 

RNN の隠れ状態を更新する場合、重み行列 $d \times d$ と $d$ 次元の隠れ状態の乗算は $\mathcal{O}(d^2)$ の計算複雑さを持ちます。シーケンスの長さは $n$ なので、リカレント層の計算複雑度は $\mathcal{O}(nd^2)$ です。:numref:`fig_cnn-rnn-self-attention` によると、並列化できない $\mathcal{O}(n)$ 個のシーケンシャルオペレーションがあり、最大パス長も $\mathcal{O}(n)$ です。 

自己注意では、クエリ、キー、および値はすべて $n \times d$ 行列です。:eqref:`eq_softmax_QK_V` のスケーリングされたドット積アテンションについて考えてみます。$n \times d$ 行列は $d \times n$ 行列で乗算され、出力 $n \times n$ 行列は $n \times d$ 行列で乗算されます。その結果、自己注意は$\mathcal{O}(n^2d)$の計算複雑さを持ちます。:numref:`fig_cnn-rnn-self-attention` でわかるように、各トークンは自己注意によって他のトークンと直接接続されています。したがって、$\mathcal{O}(1)$ の逐次演算で計算を並列に行うことができ、最大パス長も $\mathcal{O}(1)$ になります。 

全体として、CNNと自己注意の両方が並列計算を享受し、自己注意は最短の最大経路長を持ちます。ただし、シーケンスの長さに関する二次計算の複雑さにより、非常に長いシーケンスでは自己注意が非常に遅くなります。 

## [**位置符号化**]
:label:`subsec_positional-encoding`

シーケンスのトークンを一つずつ繰り返し処理するRNNとは異なり、自己注意は逐次演算を捨てて並列計算を優先します。シーケンスの順序情報を使用するには、入力表現に*positional encoding* を追加して、絶対位置情報または相対位置情報を注入できます。位置エンコーディングは、学習することも、固定することもできます。以下では、正弦関数と余弦関数 :cite:`Vaswani.Shazeer.Parmar.ea.2017` に基づく固定位置符号化について説明します。 

入力表現 $\mathbf{X} \in \mathbb{R}^{n \times d}$ に、シーケンスの $n$ トークンに対する $d$ 次元の埋め込みが含まれているとします。位置エンコーディングは、$i^\mathrm{th}$ 行と $(2j)^\mathrm{th}$ または $(2j + 1)^\mathrm{th}$ 列の要素を持つ、同じ形状の位置埋め込み行列 $\mathbf{P} \in \mathbb{R}^{n \times d}$ を使用して $\mathbf{X} + \mathbf{P}$ を出力します。 

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

一見すると、この三角関数のデザインは奇妙に見えます。この設計を説明する前に、まず次の `PositionalEncoding` クラスに実装しておきましょう。

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
#@tab tensorflow
#@save
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough `P`
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

位置埋め込み行列 $\mathbf{P}$ では、[**行はシーケンス内の位置に対応し、列は異なる位置エンコーディングの次元を表します**]。以下の例では、位置埋め込み行列の $6^{\mathrm{th}}$ 列と $7^{\mathrm{th}}$ 列の頻度が $8^{\mathrm{th}}$ 列と $9^{\mathrm{th}}$ 列よりも高いことがわかります。$6^{\mathrm{th}}$ 列と $7^{\mathrm{th}}$ 列の間のオフセット ($8^{\mathrm{th}}$ と $9^{\mathrm{th}}$ で同じ) は、正弦関数と余弦関数が交互になったためです。

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

### 絶対位置情報

符号化次元に沿って単調に減少した周波数が絶対位置情報とどのように関連しているかを調べるために、$0, 1, \ldots, 7$ の [**バイナリ表現**] を出力してみましょう。ご覧のとおり、最下位ビット、2番目に低いビット、3番目に低いビットが、それぞれすべての数字、2つの数字、4つの数字ごとに交互になります。

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

バイナリ表現では、高いビットは低いビットよりも周波数が低くなります。同様に、以下のヒートマップに示すように、三角関数を使用して [**位置符号化は符号化次元に沿って周波数を減少させる**]。出力は浮動小数点数なので、このような連続表現は 2 進表現よりもスペース効率が良いです。

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 相対的な位置情報

上記の位置エンコーディングは、絶対的な位置情報を取り込むことに加え、モデルが相対的な位置によって出席することを容易に学習することを可能にする。これは、固定位置オフセット $\delta$ の場合、$i + \delta$ の位置エンコードは $i$ の位置の線形投影で表すことができるためです。 

この図法は数学的に説明できます。$\omega_j = 1/10000^{2j/d}$ を表すと、:eqref:`eq_positional-encoding-def` に含まれる $(p_{i, 2j}, p_{i, 2j+1})$ の任意のペアは $\delta$ の任意の固定オフセットに対して $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$ に直線的に投影できます。 

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

$2\times 2$ 射影行列はどの位置インデックス $i$ にも依存しません。 

## [概要

* 自己注意では、クエリ、キー、値はすべて同じ場所から取得されます。
* CNNとセルフアテンションはどちらも並列計算を享受し、セルフアテンションは最短の最大パス長を持ちます。ただし、シーケンスの長さに関する二次計算の複雑さにより、非常に長いシーケンスでは自己注意が非常に遅くなります。
* シーケンスの順序情報を使用するには、入力表現に位置エンコーディングを追加することで、絶対位置情報または相対位置情報を注入できます。

## 演習

1. 位置エンコーディングでセルフアテンション層を積み重ねることにより、シーケンスを表現するディープアーキテクチャを設計するとします。何が問題になりますか？
1. 学習可能な位置符号化法を設計できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:

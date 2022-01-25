# イメージの畳み込み
:label:`sec_conv_layer`

畳み込み層が理論的にどのように機能するかを理解できたので、畳み込み層が実際にどのように機能するかを確認する準備が整いました。画像データの構造を探索するための効率的なアーキテクチャとして畳み込みニューラルネットワークをモチベーションとして、実行例として画像を使用します。 

## 相互相関演算

厳密に言えば、畳み込み層が表現する演算は相互相関としてより正確に記述されるため、畳み込み層は誤った名称であることを思い出してください。:numref:`sec_why-conv` の畳み込み層の記述に基づいて、このような層では、入力テンソルとカーネルテンソルが結合され、(**相互相関演算**) によって出力テンソルが生成されます。 

ここではチャネルを無視して、これが2次元データと隠れ表現でどのように機能するのか見てみましょう。:numref:`fig_correlation` では、入力は高さ 3、幅 3 の 2 次元テンソルです。テンソルの形状を $3 \times 3$ または ($3$, $3$) とマークします。カーネルの高さと幅はどちらも 2 です。*カーネルウィンドウ* (または*畳み込みウィンドウ*) の形状は、カーネルの高さと幅 (ここでは $2 \times 2$) で与えられる。 

![Two-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

2 次元の相互相関演算では、入力テンソルの左上隅に畳み込みウィンドウを配置し、入力テンソルを左から右、上から下にスライドさせます。畳み込みウィンドウが特定の位置にスライドすると、そのウィンドウに含まれる入力サブテンソルとカーネルテンソルが要素単位で乗算され、結果として得られるテンソルが合計されて 1 つのスカラー値が生成されます。この結果から、対応する位置における出力テンソルの値が得られます。ここで、出力テンソルの高さは 2、幅は 2 で、4 つの要素は 2 次元の相互相関演算から導き出されます。 

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

各軸に沿って、出力サイズは入力サイズよりわずかに小さくなることに注意してください。カーネルの幅と高さは 1 より大きいため、カーネルがイメージ内に完全に収まる位置についてのみ適切に相互相関を計算できます。出力サイズは、入力サイズ $n_h \times n_w$ から畳み込みカーネル $k_h \times k_w$ のサイズを引いた値で与えられます。 

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

これは、畳み込みカーネルをイメージ全体で「シフト」するのに十分なスペースが必要なためです。後で、カーネルをシフトするのに十分なスペースを確保するために、イメージの境界の周囲にゼロを埋め込むことによって、サイズを変更しないでおく方法を見ていきます。次に、このプロセスを関数 `corr2d` に実装します。この関数は、入力テンソル `X` とカーネルテンソル `K` を受け入れ、出力テンソル `Y` を返します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

2次元相互相関演算の [**上記の実装の出力を検証**] するために、:numref:`fig_correlation` から入力テンソル `X` とカーネルテンソル `K` を構築できます。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## 畳み込み層

畳み込み層は入力とカーネルを相互相関させ、スカラーバイアスを加算して出力を生成します。畳み込み層の 2 つのパラメーターは、カーネルとスカラーバイアスです。畳み込み層に基づいてモデルを学習させる場合、通常、完全結合層の場合と同様に、カーネルをランダムに初期化します。 

これで、上で定義した関数 `corr2d` に基づいて [**二次元畳み込み層を実装する**] の準備が整いました。`__init__` コンストラクター関数では、`weight` と `bias` を 2 つのモデルパラメーターとして宣言します。順伝播関数は `corr2d` 関数を呼び出し、バイアスを加算します。

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

$h \times w$ 畳み込みカーネルまたは $h \times w$ 畳み込みカーネルでは、畳み込みカーネルの高さと幅はそれぞれ $h$ と $w$ です。また、$h \times w$ 畳み込みカーネルをもつ畳み込み層を単に $h \times w$ 畳み込み層と呼んでいます。 

## イメージ内の物体エッジ検出

ここで [**畳み込み層の簡単な応用:画像内の物体のエッジを検出する**] をピクセル変化の位置を見つけることで解析してみましょう。まず、$6\times 8$ ピクセルの「イメージ」を作成します。真ん中の 4 列は黒 (0) で、残りは白 (1) です。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

次に、高さ 1、幅 2 のカーネル `K` を構築します。入力に対して相互相関演算を実行すると、水平に隣接する要素が同じであれば、出力は 0 になります。それ以外の場合、出力は 0 以外になります。

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

これで、引数 `X` (入力) と `K` (カーネル) を指定して、相互相関演算を実行する準備ができました。ご覧のとおり、[**白から黒のエッジに1を、黒から白のエッジに-1を検出します**] その他の出力はすべて0になります。

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

これで、転置されたイメージにカーネルを適用できます。さすがに消えてしまう。[**カーネル `K` は垂直エッジのみを検出します**]

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## カーネルの学習

有限差分 `[1, -1]` によるエッジ検出器の設計は、これがまさに私たちが求めているものであることがわかっていれば素晴らしいことです。しかし、より大きなカーネルを見て、連続する畳み込み層を考慮すると、各フィルターが手動で行うべきことを正確に指定することは不可能かもしれません。 

ここで、入出力ペアのみを見て [**`X` から `Y` を生成したカーネルを知る**] ことができるかどうかを見てみましょう。まず畳み込み層を構築し、そのカーネルをランダムテンソルとして初期化します。次に、各反復で、二乗誤差を使用して `Y` と畳み込み層の出力を比較します。その後、勾配を計算してカーネルを更新できます。わかりやすくするために、以下では 2 次元の畳み込み層に組み込みクラスを使用し、バイアスを無視します。

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2  # Learning rate

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # Learning rate

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

10 回の反復の後、エラーは小さい値に低下したことに注意してください。ここで [**学習したカーネルテンソルを見てみましょう**]

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

実際、学習したカーネルテンソルは、先に定義したカーネルテンソル `K` に非常に近いです。 

## 相互相関と畳み込み

:numref:`sec_why-conv` からの相互相関演算と畳み込み演算の対応に関する観測を思い出してください。ここでは、引き続き二次元畳み込み層について考えてみましょう。そのような層が :eqref:`eq_2d-conv-discrete` で定義されているように、相互相関ではなく厳密な畳み込み演算を実行するとどうなるでしょうか。厳密な*畳み込み* 演算の出力を得るには、2 次元のカーネルテンソルを水平方向と垂直方向に反転し、入力テンソルで*相互相関* 演算を実行するだけで済みます。 

カーネルはディープラーニングのデータから学習されるため、そのような層が厳密な畳み込み演算または相互相関演算を実行しても、畳み込み層の出力は影響を受けないことに注意してください。 

これを説明するために、畳み込み層が*相互相関* を実行し、:numref:`fig_correlation` のカーネルを学習すると仮定します。このカーネルは行列 $\mathbf{K}$ で表されます。他の条件が変わらないと仮定すると、この層が代わりに厳密な*畳み込み* を実行すると、$\mathbf{K}'$ が水平方向と垂直方向の両方で反転された後、学習されたカーネル $\mathbf{K}'$ は $\mathbf{K}$ と同じになります。つまり、:numref:`fig_correlation` と $\mathbf{K}'$ の入力に対して畳み込み層が厳密な*畳み込み* を実行すると、:numref:`fig_correlation` で同じ出力 (入力と $\mathbf{K}$ の相互相関) が得られます。 

ディープラーニングに関する文献の標準的な用語に従い、厳密には少し異なりますが、相互相関演算を畳み込みと呼び続けます。また、*element* という用語は、層表現または畳み込みカーネルを表すテンソルのエントリ (またはコンポーネント) を指すために使用します。 

## 特徴マップと受容野

:numref:`subsec_why-conv-channels` で説明したように、:numref:`fig_correlation` の畳み込み層の出力は、後続の層の空間次元 (幅や高さなど) で学習された表現 (特徴) と見なすことができるため、*特徴マップ* と呼ばれることもあります。CNN では、ある層の要素 $x$ について、その*受容場* は、順伝播中に $x$ の計算に影響する可能性のある (以前のすべての層の) すべての要素を指します。受容野は入力の実際のサイズより大きい場合があることに注意してください。 

:numref:`fig_correlation`を引き続き使って受容野を説明しよう。$2 \times 2$ 畳み込みカーネルでは、シェーディングされた出力要素 (値 $19$) の受容場は、入力の陰影部分の 4 つの要素になります。ここで、$2 \times 2$ の出力を $\mathbf{Y}$ と表現し、$2 \times 2$ の畳み込み層を追加して $\mathbf{Y}$ を入力とし、単一要素 $z$ を出力する、より深い CNN を考えてみましょう。この場合、$\mathbf{Y}$ の $z$ の受容野には $\mathbf{Y}$ の 4 つの要素すべてが含まれ、入力の受容野には 9 つの入力要素がすべて含まれます。したがって、フィーチャマップ内のエレメントが、より広いエリアにわたって入力フィーチャを検出するためにより大きな受容場を必要とする場合、より深いネットワークを構築できます。 

## [概要

* 2 次元畳み込み層の中核となる計算は、2 次元の相互相関演算です。最も単純な形式では、2 次元の入力データとカーネルに対して相互相関演算を実行し、バイアスを加えます。
* 画像のエッジを検出するカーネルを設計できます。
* カーネルのパラメータはデータから学ぶことができます。
* データから学習したカーネルでは、畳み込み層の出力は、そのような層の演算 (厳密な畳み込みまたは相互相関) にかかわらず影響を受けません。
* 特徴マップのいずれかのエレメントが、入力上のより広い特徴を検出するためにより大きな受容場を必要とする場合、より深いネットワークが考えられます。

## 演習

1. 対角エッジをもつイメージ `X` を作成します。
    1. このセクションのカーネル `K` を適用するとどうなりますか？
    1. `X` を転置するとどうなりますか？
    1. `K` を転置するとどうなりますか？
1. 作成した `Conv2D` クラスのグラデーションを自動的に見つけようとすると、どのようなエラーメッセージが表示されますか？
1. 入力テンソルとカーネルテンソルを変更して、相互相関演算を行列乗算としてどのように表現しますか？
1. いくつかのカーネルを手作業で設計する。
    1. 二次導関数のカーネルの形式は何ですか？
    1. 積分のカーネルは何ですか？
    1. 次数$d$の微分を得るためのカーネルの最小サイズはどれくらいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:

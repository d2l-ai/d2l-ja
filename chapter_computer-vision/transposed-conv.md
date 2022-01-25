# 転置畳み込み
:label:`sec_transposed_conv`

畳み込み層 (:numref:`sec_conv_layer`) やプーリング層 (:numref:`sec_pooling`) などの CNN 層は、通常、入力の空間次元 (高さと幅) を減らす (ダウンサンプリング) するか、変更しないままにします。ピクセルレベルで分類するセマンティックセグメンテーションでは、入力と出力の空間次元が同じであれば便利です。たとえば、1 つの出力ピクセルのチャネル次元は、同じ空間位置にある入力ピクセルの分類結果を保持できます。 

これを実現するために、特に CNN レイヤーによって空間次元が削減された後に、中間フィーチャマップの空間次元を増加 (アップサンプリング) できる別のタイプの CNN レイヤーを使用できます。このセクションでは、 
*転置畳み込み*。これは*分数ストライド畳み込み* :cite:`Dumoulin.Visin.2016`とも呼ばれ、 
畳み込みによるダウンサンプリングの逆転に使用します。

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## 基本操作

とりあえずチャンネルを無視して、ストライドが 1 でパディングなしの基本的な転置畳み込み演算から始めましょう。$n_h \times n_w$ の入力テンソルと $k_h \times k_w$ カーネルが与えられているとします。各行で $n_w$ 回、各列で $n_h$ 回、ストライドを 1 にしてカーネルウィンドウをスライドすると、合計 $n_h n_w$ の中間結果が得られます。各中間結果は $(n_h + k_h - 1) \times (n_w + k_w - 1)$ テンソルで、ゼロとして初期化されます。各中間テンソルを計算するために、入力テンソルの各要素にカーネルが乗算され、結果として得られる $k_h \times k_w$ テンソルが各中間テンソルの一部を置き換えます。各中間テンソルにおける置換部分の位置は、計算に使用される入力テンソル内の要素の位置に対応することに注意してください。最後に、すべての中間結果が合計され、出力が生成されます。 

例として、:numref:`fig_trans_conv` は $2\times 2$ カーネルの転置畳み込みが $2\times 2$ 入力テンソルに対してどのように計算されるかを示しています。 

![Transposed convolution with a $2\times 2$ kernel. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv.svg)
:label:`fig_trans_conv`

入力行列 `X` とカーネル行列 `K` に対して `trans_conv` (**この基本的な転置畳み込み演算を実装**) できます。

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

カーネルを介して入力要素を*削減* する通常の畳み込み (:numref:`sec_conv_layer`) とは対照的に、転置畳み込みは
*ブロードキャスト* 入力要素 
カーネル経由で、入力よりも大きな出力を生成します。基本的な二次元転置畳み込み演算の [**上記の実装の出力を検証**] するために、:numref:`fig_trans_conv` から入力テンソル `X` とカーネルテンソル `K` を構築できます。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```

あるいは、入力 `X` とカーネル `K` がどちらも 4 次元テンソルの場合、[**高レベル API を使用して同じ結果を得る**] ことができます。

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## [**パディング、ストライド、マルチチャンネル**]

入力にパディングが適用される通常の畳み込みとは異なり、転置畳み込みでは出力に適用されます。たとえば、高さと幅の両側のパディング数を 1 に指定すると、転置された畳み込み出力から最初と最後の行と列が削除されます。

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

転置畳み込みでは、ストライドは入力ではなく中間結果 (つまり出力) に対して指定されます。:numref:`fig_trans_conv` の同じ入力テンソルとカーネルテンソルを使用して、ストライドを 1 から 2 に変更すると、中間テンソルの高さと重みの両方が大きくなるため、出力テンソルは :numref:`fig_trans_conv_stride2` になります。 

![Transposed convolution with a $2\times 2$ kernel with stride of 2. The shaded portions are a portion of an intermediate tensor as well as the input and kernel tensor elements used for the  computation.](../img/trans_conv_stride2.svg)
:label:`fig_trans_conv_stride2`

次のコードスニペットは、:numref:`fig_trans_conv_stride2` の stride 2 の転置畳み込み出力を検証できます。

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

入力チャンネルと出力チャンネルが複数ある場合、転置された畳み込みは通常の畳み込みと同じように機能します。入力に $c_i$ チャネルがあり、転置畳み込みによって各入力チャネルに $k_h\times k_w$ カーネルテンソルが割り当てられるとします。複数の出力チャンネルを指定すると、出力チャンネルごとに $c_i\times k_h\times k_w$ カーネルが作成されます。 

すべての場合と同様に、$\mathsf{X}$ を畳み込み層 $f$ に供給して $\mathsf{Y}=f(\mathsf{X})$ を出力し、$f$ と同じハイパーパラメーターを持つ転置畳み込み層 $g$ を作成すると、$\mathsf{X}$ のチャンネル数である出力チャンネル数を除き、$g(Y)$ は$\mathsf{X}$。これを次の例に示します。

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## [**行列転置への接続**]
:label:`subsec-connection-to-mat-transposition`

転置畳み込みは、行列の転置にちなんで名付けられました。説明するために、まず行列乗算を使用して畳み込みを実装する方法を見てみましょう。以下の例では、$3\times 3$ の入力 `X` と $2\times 2$ の畳み込みカーネル `K` を定義し、`corr2d` 関数を使用して畳み込み出力 `Y` を計算しています。

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
Y
```

次に、畳み込みカーネル `K` を、ゼロを多く含むスパース重み行列 `W` として書き直します。重み行列の形状は ($4$, $9$) で、ゼロ以外の要素は畳み込みカーネル `K` から得られます。

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

入力 `X` を 1 行ずつ連結して、長さ 9 のベクトルを取得します。その後、`W` の行列乗算とベクトル化された `X` の行列乗算により、長さ 4 のベクトルが得られます。形状を変更すると、上記の元の畳み込み演算から同じ結果 `Y` が得られます。行列の乗算を使用して畳み込みを実装しただけです。

```{.python .input}
#@tab all
Y == d2l.matmul(W, d2l.reshape(X, -1)).reshape(2, 2)
```

同様に、行列乗算を使用して転置畳み込みを実装できます。次の例では、上記の通常の畳み込みからの $2 \times 2$ 出力 `Y` を転置畳み込みへの入力として取ります。行列を乗算してこの演算を実装するには、重み行列 `W` を新しいシェイプ $(9, 4)$ で転置するだけで済みます。

```{.python .input}
#@tab all
Z = trans_conv(Y, K)
Z == d2l.matmul(W.T, d2l.reshape(Y, -1)).reshape(3, 3)
```

行列を乗算して畳み込みを実装することを検討してください。入力ベクトル $\mathbf{x}$ と重み行列 $\mathbf{W}$ を指定すると、その入力に重み行列を乗算してベクトル $\mathbf{y}=\mathbf{W}\mathbf{x}$ を出力することで、畳み込みの順伝播関数を実装できます。バックプロパゲーションはチェーンルールと $\nabla_{\mathbf{x}}\mathbf{y}=\mathbf{W}^\top$ に従うため、畳み込みのバックプロパゲーション関数は、その入力に転置された重み行列 $\mathbf{W}^\top$ を乗算することで実装できます。したがって、転置された畳み込み層は、畳み込み層の順伝播関数と逆伝播関数を交換するだけで済みます。その順伝播関数と逆伝播関数は、入力ベクトルにそれぞれ $\mathbf{W}^\top$ と $\mathbf{W}$ を乗算します。 

## [概要

* カーネルを介して入力要素を削減する通常の畳み込みとは異なり、転置畳み込みは入力要素をカーネル経由でブロードキャストするため、入力より大きい出力が生成されます。
* $\mathsf{X}$ を畳み込み層 $f$ に供給して $\mathsf{Y}=f(\mathsf{X})$ を出力し、$\mathsf{X}$ のチャンネル数である出力チャンネル数を除いて $f$ と同じハイパーパラメーターをもつ転置畳み込み層 $g$ を作成すると、$g(Y)$ は $\mathsf{X}$ と同じ形状になります。
* 畳み込みは行列乗算を使って実装できます。転置された畳み込み層は、畳み込み層の順伝播関数と逆伝播関数を交換するだけです。

## 演習

1. :numref:`subsec-connection-to-mat-transposition` では、畳み込み入力 `X` と転置畳み込み出力 `Z` は同じ形状になっています。それらは同じ価値を持っていますか？なぜ？
1. 畳み込みを実装するのに行列の乗算を使うのは効率的ですか？なぜ？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:

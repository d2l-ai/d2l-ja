# 複数の入力チャンネルと複数の出力チャンネル
:label:`sec_channels`

:numref:`subsec_why-conv-channels` では、各イメージを構成する複数のチャネル (たとえば、カラーイメージには赤、緑、青の量を示す標準の RGB チャネルがある) と複数チャネルの畳み込み層について説明してきましたが、これまで、すべての数値例を単純化して 1 つの入力と単一の出力チャンネル。これにより、入力、畳み込みカーネル、および出力をそれぞれ二次元テンソルと考えることができました。 

ミックスにチャンネルを追加すると、入力と非表示表現の両方が 3 次元のテンソルになります。たとえば、各 RGB 入力イメージの形状は $3\times h\times w$ です。サイズが 3 のこの軸を*チャネル* 次元と呼びます。このセクションでは、複数の入力チャンネルと複数の出力チャンネルをもつ畳み込みカーネルについて詳しく見ていきます。 

## 複数入力チャンネル

入力データに複数のチャネルが含まれる場合、入力データと相互相関を実行できるように、入力データと同じ数の入力チャネルを持つ畳み込みカーネルを構築する必要があります。入力データのチャネル数が $c_i$ であると仮定すると、畳み込みカーネルの入力チャネル数も $c_i$ である必要があります。畳み込みカーネルのウィンドウ形状が $k_h\times k_w$ の場合、$c_i=1$ の場合、畳み込みカーネルは $k_h\times k_w$ という形の 2 次元テンソルと考えることができます。 

ただし、$c_i>1$ の場合、*すべての* 入力チャンネルに対して形状 $k_h\times k_w$ のテンソルを含むカーネルが必要です。これらの $c_i$ テンソルを連結すると、$c_i\times k_h\times k_w$ という形状の畳み込みカーネルが生成されます。入力カーネルと畳み込みカーネルはそれぞれ $c_i$ チャネルを持つため、入力の 2 次元テンソルと各チャネルの畳み込みカーネルの 2 次元テンソルに対して相互相関演算を実行し、$c_i$ の結果を加算して (チャネルを合計して) 2次元テンソル。これは、マルチチャネル入力とマルチ入力チャネル畳み込みカーネル間の 2 次元相互相関の結果です。 

:numref:`fig_conv_multi_in` では、2 つの入力チャネルを持つ 2 次元の相互相関の例を示します。影付きの部分は、最初の出力要素であり、出力の計算に使用される入力テンソル要素とカーネルテンソル要素 $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$ です。 

![Cross-correlation computation with 2 input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

ここで何が起こっているのかを本当に理解するために、（**複数の入力チャンネルで相互相関演算を実装する**）ことができます。ここで行っているのは、チャネルごとに 1 つの相互相関演算を実行して、その結果を合計することだけです。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

:numref:`fig_conv_multi_in` の値に対応する入力テンソル `X` とカーネルテンソル `K` を構築して、相互相関演算の (**出力を検証**) することができます。

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## 複数出力チャンネル
:label:`subsec_multi-output-channels`

入力チャンネルの数にかかわらず、これまでのところ、常に1つの出力チャンネルになってしまいました。しかし、:numref:`subsec_why-conv-channels` で説明したように、各レイヤーに複数のチャンネルを配置することが不可欠であることがわかりました。最も一般的なニューラルネットワークアーキテクチャでは、ニューラルネットワークの上位に上がるにつれてチャネルの次元が大きくなります。通常は、空間分解能と*チャネル深度*を大きくするためにダウンサンプリングします。直感的に、各チャネルは異なる機能セットに応答していると考えることができます。現実は、この直感の最も素朴な解釈よりも少し複雑です。なぜなら、表現は独立して学習されるのではなく、共同で役立つように最適化されているからです。したがって、1 つのチャネルがエッジ検出器を学習するのではなく、チャネル空間のある方向がエッジの検出に対応している場合があります。 

$c_i$ と $c_o$ で入出力チャンネルの数をそれぞれ示し、$k_h$ と $k_w$ をカーネルの高さと幅とします。複数のチャンネルをもつ出力を得るために、*すべての* 出力チャンネルに対して $c_i\times k_h\times k_w$ という形状のカーネルテンソルを作成できます。畳み込みカーネルの形状が $c_o\times c_i\times k_h\times k_w$ になるように、これらを出力チャンネルの次元で連結します。相互相関演算では、各出力チャネルの結果は、その出力チャネルに対応する畳み込みカーネルから計算され、入力テンソルのすべてのチャネルから入力を受け取ります。 

[**複数チャンネルの出力を計算**] する相互相関関数を以下のように実装します。

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

カーネルテンソル `K` を `K+1` (`K` では要素ごとに 1 つ足す) と `K+2` と連結して、3 つの出力チャンネルをもつ畳み込みカーネルを構築します。

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

以下では、カーネルテンソル `K` を使用して、入力テンソル `X` に対して相互相関演算を行います。これで、出力には3つのチャンネルが含まれています。1 番目のチャネルの結果は、前の入力テンソル `X` および多入力チャネル、単一出力チャネルカーネルの結果と一致します。

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ 畳み込みレイヤー

最初は、[**$1 \times 1$ 畳み込み**]、つまり $k_h = k_w = 1$ はあまり意味をなさないようです。結局のところ、畳み込みは隣接するピクセルを相関させます。$1 \times 1$ 畳み込みは明らかにそうではありません。それでも、これらは一般的な操作であり、複雑なディープネットワークの設計に含まれることもあります。それが実際に何をしているのかを詳しく見てみましょう。 

最小ウィンドウが使用されるため、$1\times 1$ 畳み込みでは、より大きい畳み込み層では、高さと幅の次元で隣接する要素間の相互作用で構成されるパターンを認識できなくなります。$1\times 1$ 畳み込みの計算はチャネル次元でのみ行われます。 

:numref:`fig_conv_1x1` は、3 つの入力チャネルと 2 つの出力チャネルをもつ $1\times 1$ 畳み込みカーネルを使用した相互相関計算を示しています。入力と出力の高さと幅は同じであることに注意してください。出力の各要素は、入力イメージ内の*同じ位置* にある要素の線形結合から導き出されます。$1\times 1$ 畳み込み層は、$c_i$ の対応する入力値を $c_o$ の出力値に変換するために、すべてのピクセル位置に適用される完全結合層を構成していると考えることができます。これはまだ畳み込み層なので、重みはピクセル位置全体で結び付けられます。したがって、$1\times 1$ 畳み込み層には $c_o\times c_i$ の重み (バイアスを加えた値) が必要です。 

![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

これが実際に機能するかどうかを確認してみましょう。完全結合層を使用して $1 \times 1$ 畳み込みを実装します。唯一のことは、行列乗算の前後にデータ形状を調整する必要があるということです。

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully-connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

$1\times 1$ 畳み込みを実行すると、上記の関数は以前に実装された相互相関関数 `corr2d_multi_in_out` と等価になります。これをいくつかのサンプルデータで確認してみましょう。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## [概要

* 畳み込み層のモデルパラメーターを拡張するために、複数のチャネルを使用できます。
* $1\times 1$ 畳み込みレイヤーは、ピクセル単位で適用すると、完全結合レイヤーに相当します。
* $1\times 1$ 畳み込み層は、通常、ネットワーク層間のチャネル数を調整し、モデルの複雑さを制御するために使用されます。

## 演習

1. サイズが $k_1$ と $k_2$ の 2 つの畳み込みカーネルがあると仮定します (間に非線形性はありません)。
    1. 演算の結果は、単一の畳み込みで表すことができることを証明する。
    1. 等価単一畳み込みの次元はどれくらいですか？
    1. その逆は本当ですか？
1. 形状 $c_i\times h\times w$ の入力と、形状 $c_o\times c_i\times k_h\times k_w$、パディングが $(p_h, p_w)$、ストライドが $(s_h, s_w)$ の畳み込みカーネルを想定します。
    1. 順伝播の計算コスト (乗算と加算) はどれくらいですか?
    1. メモリフットプリントはどれくらいですか？
    1. 逆方向計算のメモリフットプリントはどれくらいですか？
    1. バックプロパゲーションの計算コストはどれくらいですか？
1. 入力チャンネル数$c_i$と出力チャンネル数$c_o$を2倍にすると、計算回数はどの要因で増えますか？パディングを2倍にするとどうなりますか？
1. 畳み込みカーネルの高さと幅が $k_h=k_w=1$ の場合、順伝播の計算の複雑さはどれくらいですか？
1. このセクションの最後の例の変数 `Y1` と `Y2` はまったく同じですか。なぜ？
1. 畳み込みウィンドウが $1\times 1$ でない場合、行列乗算を使用して畳み込みをどのように実装しますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:

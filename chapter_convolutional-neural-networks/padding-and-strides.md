# パディングとストライド
:label:`sec_padding`

前の :numref:`fig_correlation` の例では、入力の高さと幅の両方が 3 で、畳み込みカーネルの高さと幅が両方とも 2 で、次元 $2\times2$ の出力表現が生成されます。:numref:`sec_conv_layer` で一般化したように、入力形状が $n_h\times n_w$、畳み込みカーネル形状が $k_h\times k_w$ であると仮定すると、出力形状は $(n_h-k_h+1) \times (n_w-k_w+1)$ になります。したがって、畳み込み層の出力形状は、入力の形状と畳み込みカーネルの形状によって決まります。 

いくつかのケースでは、パディングやストライド畳み込みなど、出力のサイズに影響する手法を取り入れています。動機として、カーネルの幅と高さは一般に $1$ より大きいため、連続して畳み込みを何度も適用すると、出力が入力よりもかなり小さくなる傾向があることに注意してください。$240 \times 240$ ピクセルのイメージから始めると、$5 \times 5$ の畳み込みの $10$ レイヤーはイメージを $200 \times 200$ ピクセルに減らし、30\ %$ ドルのイメージを切り取って、元のイメージの境界に関する興味深い情報をすべて消去します。
*パディング* は、この問題を処理する最も一般的なツールです。

また、元の入力解像度が扱いにくい場合など、次元を大幅に削減したい場合もあります。
*ストライド畳み込み* は、このような場合に役立つ一般的な手法です。

## パディング

前述のとおり、畳み込みレイヤーを適用する際に注意が必要な問題の 1 つは、画像の周囲のピクセルが失われがちであることです。通常、小さなカーネルを使用するため、任意の畳み込みでは数ピクセルしか失われないかもしれませんが、連続して多くの畳み込み層を適用すると、この数が増える可能性があります。この問題の直接的な解決策の 1 つは、入力イメージの境界の周囲にフィラーのピクセルを追加して、イメージの実効サイズを大きくすることです。通常は、余分のピクセルの値を 0 に設定します。:numref:`img_conv_pad` では $3 \times 3$ の入力をパディングし、そのサイズを $5 \times 5$ に増やしました。対応する出力は $4 \times 4$ 行列に増加します。影付きの部分は、最初の出力要素であり、出力の計算に使用される入力テンソル要素とカーネルテンソル要素 $0\times0+0\times1+0\times2+0\times3=0$ です。 

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

一般に、合計 $p_h$ 行のパディング (上半分は下)、合計で $p_w$ 列のパディング (左半分は右) を追加すると、出力シェイプは次のようになります。 

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$

これは、出力の高さと幅がそれぞれ $p_h$ と $p_w$ ずつ増加することを意味します。 

多くの場合、$p_h=k_h-1$ と $p_w=k_w-1$ を設定して、入力と出力の高さと幅を同じにします。これにより、ネットワークの構築時に各層の出力形状を予測しやすくなります。ここで $k_h$ が奇数であると仮定して、高さの両側に $p_h/2$ 行をパディングします。$k_h$ が偶数の場合は、$\lceil p_h/2\rceil$ 行を入力の先頭に、$\lfloor p_h/2\rfloor$ 行を一番下に埋め込む方法があります。幅の両側を同じように埋めます。 

CNN では通常、1、3、5、7 など、高さと幅の値が奇数の畳み込みカーネルを使用します。奇数のカーネルサイズを選択すると、上下の行数が同じで、左右に同じ数の列でパディングしながら、空間次元を維持できるという利点があります。 

さらに、次元を正確に保持するために奇数カーネルとパディングを使用するこの習慣は、事務的な利点をもたらします。どの 2 次元テンソル `X` でも、カーネルのサイズが奇数で、すべての辺のパディングの行と列の数が同じで、入力と同じ高さと幅で出力が生成される場合、出力 `Y[i, j]` は入力カーネルと畳み込みカーネルの相互相関によって計算されることがわかります。窓は`X[i, j]`を中心にしてあります。 

次の例では、高さと幅が 3 の 2 次元の畳み込み層を作成し、(**すべての辺に 1 ピクセルのパディングを適用する**)。高さと幅が 8 の入力があると、出力の高さと幅も 8 であることがわかります。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

畳み込みカーネルの高さと幅が異なる場合、[**高さと幅に異なるパディング数を設定**] することで、出力と入力の高さと幅を同じにすることができます。

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

## ストライド

相互相関を計算するときは、入力テンソルの左上隅にある畳み込みウィンドウから開始し、それをすべての位置で下と右の両方にスライドさせます。前の例では、デフォルトでは一度に 1 つのエレメントをスライドさせていました。ただし、計算効率の向上またはダウンサンプリングのために、ウィンドウを一度に複数の要素に移動して、中間位置をスキップすることがあります。 

スライドごとにトラバースされる行と列の数を*stride* と呼びます。ここまでは、高さと幅の両方に 1 のストライドを使用しました。場合によっては、より大きいストライドを使用したい場合があります。:numref:`img_conv_stride` は、ストライドが垂直方向に 3、水平方向に 2 の 2 次元の相互相関演算を示しています。影付きの部分は、出力要素、および出力計算に使用される入力テンソル要素、カーネルテンソル要素 $0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$ です。最初の列の 2 番目の要素が出力されると、畳み込みウィンドウが 3 行下にスライドすることがわかります。最初の行の 2 番目の要素が出力されると、畳み込みウィンドウは 2 列右にスライドします。畳み込みウィンドウが入力上で 2 列右にスライドし続けると、入力要素はウィンドウを埋めることができないため (パディングの列をもう 1 つ追加しない限り)、出力はありません。 

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

一般に、高さのストライドが $s_h$ で、幅のストライドが $s_w$ の場合、出力シェイプは次のようになります。 

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

$p_h=k_h-1$ と $p_w=k_w-1$ を設定すると、出力シェイプは $\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$ に簡略化されます。さらに進んで、入力の高さと幅が、高さと幅のストライドで割り切れる場合、出力シェイプは $(n_h/s_h) \times (n_w/s_w)$ になります。 

以下では、[**高さと幅の両方のストライドを2に設定**] して、入力の高さと幅を半分にします。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

次に (**もう少し複雑な例**) を見ていきましょう。

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

簡潔にするために、入力された高さと幅の両側のパディング数がそれぞれ $p_h$ と $p_w$ の場合、パディングを $(p_h, p_w)$ と呼びます。具体的には、$p_h = p_w = p$ の場合、パディングは $p$ になります。高さと幅のストライドがそれぞれ $s_h$ と $s_w$ の場合、ストライドを $(s_h, s_w)$ と呼びます。具体的には、$s_h = s_w = s$ の場合、ストライドは $s$ になります。デフォルトでは、パディングは 0、ストライドは 1 です。実際には、不均一なストライドやパディングを使用することはほとんどありません。つまり、通常 $p_h = p_w$ と $s_h = s_w$ があります。 

## [概要

* パディングによって、出力の高さと幅が大きくなることがあります。これは、出力に入力と同じ高さと幅を与えるためによく使用されます。
* ストライドによって出力の解像度を下げることができます。たとえば、出力の高さと幅を入力の高さと幅の $1/n$ ($n$ は $1$ より大きい整数) に減らすことができます。
* パディングとストライドは、データの次元を効果的に調整するために使用できます。

## 演習

1. このセクションの最後の例では、数学を使用して出力形状を計算し、実験結果と整合性があるかどうかを確認します。
1. このセクションの実験では、他のパディングとストライドの組み合わせを試してみてください。
1. オーディオ信号の場合、ストライド 2 は何に対応しますか？
1. ストライドが1より大きい場合の計算上の利点は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:

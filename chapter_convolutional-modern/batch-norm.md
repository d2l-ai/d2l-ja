# バッチ正規化
:label:`sec_batch_norm`

ディープニューラルネットワークの学習は困難です。そして、それらを妥当な時間内に収束させるのは難しい場合があります。このセクションでは、ディープネットワーク :cite:`Ioffe.Szegedy.2015` の収束を一貫して高速化する、一般的で効果的な手法である*バッチ正規化*について説明します。:numref:`sec_resnet` で後述する残差ブロックとともに、バッチ正規化により、開業医は 100 層を超えるネットワークを日常的にトレーニングできるようになりました。 

## ディープネットワークのトレーニング

バッチ正規化を促進するために、特に機械学習モデルやニューラルネットワークをトレーニングする際に生じる実際的な課題をいくつか見てみましょう。 

まず、データの前処理に関する選択によって、最終結果に大きな違いが生じることがよくあります。住宅価格の予測へのMLPの適用を思い出してください (:numref:`sec_kaggle_house`)。実データを操作する際の最初のステップは、入力フィーチャの平均がゼロ、分散が 1 になるように入力フィーチャを標準化することでした。直感的に、この標準化はオプティマイザとうまく機能します。パラメータを*先験的に*同様のスケールで配置するからです。 

第2に、典型的なMLPまたはCNNの場合、中間層の変数（MLPでのアフィン変換出力など）は、入力から出力までの層に沿って、同じ層内の単位にまたがって、またモデルの更新による時間の経過とともに、大きく変化する大きさの値をとることがあります。パラメーター。バッチ正規化の発明者たちは、このような変数の分布におけるこのドリフトがネットワークの収束を妨げる可能性があると非公式に仮定しました。直観的には、ある層に別の層の 100 倍の変数値がある場合、学習率の代償的な調整が必要になる可能性があると推測できます。 

第三に、より深いネットワークは複雑で、簡単に過剰適合する可能性があります。これは、正則化がより重要になることを意味します。 

バッチ正規化は個々のレイヤー (オプションですべてのレイヤー) に適用され、次のように機能します。トレーニングの反復ごとに、まず入力の平均を引いて標準偏差で割ることによって (バッチ正規化の) 入力を正規化します。現在のミニバッチ。次に、スケール係数とスケールオフセットを適用します。*バッチ正規化* の名前が由来するのは、まさに*バッチ* 統計に基づくこの*正規化* によるものです。 

サイズが 1 のミニバッチでバッチ正規化を適用しようとすると、何も学習できないことに注意してください。これは、平均を引いた後、隠れた各単位の値が0になるからです。ご想像のとおり、このセクション全体をバッチの正規化に充てており、十分な大きさのミニバッチを使用しているので、このアプローチは効果的で安定していることが証明されています。ここで重要なのは、バッチ正規化を適用する場合、バッチの正規化を使用しない場合よりもバッチサイズの選択がさらに重要になる可能性があることです。 

正式には、$\mathbf{x} \in \mathcal{B}$ はミニバッチ $\mathcal{B}$ からのバッチ正規化 ($\mathrm{BN}$) への入力を表し、バッチ正規化は次の式に従って $\mathbf{x}$ を変換します。 

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm` では、$\hat{\boldsymbol{\mu}}_\mathcal{B}$ がサンプル平均、$\hat{\boldsymbol{\sigma}}_\mathcal{B}$ がミニバッチ $\mathcal{B}$ のサンプル標準偏差です。標準化を適用すると、結果のミニバッチはゼロの平均と単位分散になります。単位分散 (他のマジックナンバーとの比較) の選択は任意であるため、通常は elementwise を含めます。
*スケールパラメータ* $\boldsymbol{\gamma}$ と*シフトパラメータ* $\boldsymbol{\beta}$
$\mathbf{x}$ と同じ形をしています$\boldsymbol{\gamma}$ と $\boldsymbol{\beta}$ は、他のモデルパラメーターと一緒に学習する必要があるパラメーターであることに注意してください。 

したがって、中間層の変数の大きさは学習中に発散できません。これは、バッチ正規化によって中間層がアクティブにセンタリングされ、所定の平均とサイズ ($\hat{\boldsymbol{\mu}}_\mathcal{B}$ および ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$) に再スケーリングされるためです。実践者の直感または知恵のひとつは、バッチ正規化はより積極的な学習率を可能にするように思われるということです。 

正式には、:eqref:`eq_batchnorm` の $\hat{\boldsymbol{\mu}}_\mathcal{B}$ と ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ を次のように計算します。 

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

経験的分散推定値が消失する場合でも、ゼロ除算を試みないように、分散推定に小さな定数 $\epsilon > 0$ を追加することに注意してください。推定値 $\hat{\boldsymbol{\mu}}_\mathcal{B}$ と ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ は、ノイズの多い平均と分散の推定値を使用してスケーリングの問題に対処します。このうるささが問題になるはずだと思うかもしれません。結局のところ、これは実際に有益です。 

これは、ディープラーニングでは繰り返されるテーマであることが判明しました。理論的にはまだ十分に特徴付けられていない理由から、最適化におけるさまざまなノイズ源が、学習の高速化と過適合の減少につながることがよくあります。この変動は正則化の一形態として機能しているように見えます。一部の予備調査では、:cite:`Teye.Azizpour.Smith.2018` と :cite:`Luo.Wang.Shao.ea.2018` がバッチ正規化のプロパティをベイズ事前分布とペナルティにそれぞれ関連付けています。特に、バッチ正規化が $50 \sim 100$ の範囲の中程度のミニバッチサイズに最適である理由の謎が明らかになります。 

学習済みモデルを修正すると、平均と分散を推定するにはデータセット全体を使用したほうがよいと思うかもしれません。トレーニングが完了したら、同じ画像を、その画像が存在するバッチに応じて異なる分類をしたいのはなぜですか？トレーニング中は、モデルを更新するたびにすべてのデータ例の中間変数が変化するため、このような正確な計算は実行できません。ただし、モデルがトレーニングされると、データセット全体に基づいて各層の変数の平均と分散を計算できます。実際、これはバッチ正規化を使用するモデルでは標準的な方法であるため、バッチ正規化層の機能は*トレーニングモード* (ミニバッチ統計による正規化) と*予測モード* (データセット統計による正規化) では異なります。 

これで、バッチ正規化が実際にどのように機能するのかを見ていきましょう。 

## バッチ正規化レイヤ

完全結合層と畳み込み層のバッチ正規化の実装は若干異なります。両方のケースについては以下で説明します。バッチ正規化と他のレイヤーとの主な違いの 1 つは、バッチ正規化は一度に完全なミニバッチで実行されるため、他のレイヤーを導入したときのようにバッチディメンションを無視できないことです。 

### 完全接続レイヤー

バッチ正規化を全結合層に適用すると、元の用紙はアフィン変換後、非線形活性化関数の前にバッチ正規化を挿入します (後のアプリケーションでは、活性化関数の直後にバッチ正規化が挿入される可能性があります) :cite:`Ioffe.Szegedy.2015`。全結合層への入力を $\mathbf{x}$、アフィン変換を $\mathbf{W}\mathbf{x} + \mathbf{b}$ (重みパラメーター $\mathbf{W}$ とバイアスパラメーター $\mathbf{b}$ をもつ)、活性化関数を $\phi$ で表すと、バッチ正規化対応の完全結合層出力の計算を表すことができます。$\mathbf{h}$ は次のようになります。 

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

平均と分散は、変換が適用される*同じ*ミニバッチで計算されることを思い出してください。 

### 畳み込み層

同様に、畳み込み層では、畳み込み後、非線形活性化関数の前にバッチ正規化を適用できます。畳み込みに複数の出力チャンネルがある場合、これらのチャンネルの*各*出力に対してバッチ正規化を実行する必要があり、各チャンネルには独自のスケールパラメーターとシフトパラメーターがあり、どちらもスカラーです。ミニバッチに $m$ の例が含まれ、各チャンネルの畳み込みの出力の高さが $p$、幅が $q$ であると仮定します。畳み込み層では、各バッチ正規化を出力チャネルあたり $m \cdot p \cdot q$ 要素に対して同時に実行します。したがって、平均と分散を計算するときにすべての空間位置の値を収集し、その結果同じ平均と分散を所定のチャネルに適用して、各空間位置の値を正規化します。 

### 予測中のバッチ正規化

前述のとおり、バッチ正規化は通常、トレーニングモードと予測モードでは動作が異なります。まず、モデルに学習をさせると、サンプル平均のノイズとミニバッチでのそれぞれの推定から生じるサンプル分散は望ましくなくなります。第二に、バッチごとの正規化統計を計算する余裕がないかもしれません。たとえば、モデルを適用して、一度に 1 つの予測を行う必要がある場合があります。 

通常、学習後、データセット全体を使用して変数統計の安定した推定値を計算し、予測時に固定します。したがって、バッチの正規化は、学習時とテスト時に異なる動作をします。ドロップアウトもこの特性を示すことを思い出してください。 

## (**ゼロからの実装**)

以下では、テンソルを用いたバッチ正規化層をゼロから実装します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

これで [**適切な `BatchNorm` レイヤーを作成**]。レイヤーはスケール `gamma` とシフト `beta` の適切なパラメーターを維持し、どちらもトレーニング中に更新されます。さらに、レイヤーは平均と分散の移動平均を維持し、後からモデル予測に使用できるようにします。 

アルゴリズムの詳細は別として、レイヤーの実装の基礎となるデザインパターンに注目してください。通常、数学は独立した関数 (`batch_norm` など) で定義します。次に、この機能をカスタムレイヤーに統合します。カスタムレイヤーのコードは、適切なデバイスコンテキストへのデータの移動、必要な変数の割り当てと初期化、移動平均の追跡 (ここでは平均と分散) の追跡など、簿記の問題を主に処理します。このパターンにより、数学をボイラープレートコードから明確に分離できます。また、便宜上、ここでは入力形状を自動的に推論する必要はないため、全体を通してフィーチャの数を指定する必要があることにも注意してください。心配しないでください。ディープラーニングフレームワークの高レベルバッチ正規化 API がこれを処理してくれるので、後でデモンストレーションします。

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## [**LeNetでバッチ正規化を適用する**]

`BatchNorm` をコンテキストで適用する方法を確認するために、以下では従来の LeNet モデル (:numref:`sec_lenet`) に適用します。バッチ正規化は、畳み込み層または全結合層の後で、対応する活性化関数の前に適用されることを思い出してください。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

前回同様、[**Fashion-MNIST データセットでネットワークをトレーニング**] します。このコードは、LeNet (:numref:`sec_lenet`) を最初にトレーニングしたときのものと実質的に同じです。主な違いは、学習率が大きいことです。

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

最初のバッチ正規化層から学習した [**スケールパラメーター `gamma` とシフトパラメーター `beta`**] を見てみましょう。

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## [**簡潔な実装**]

先ほど定義した `BatchNorm` クラスと比較すると、ディープラーニングフレームワークの高レベル API で定義された `BatchNorm` クラスを直接使用できます。このコードは、上記の実装とほぼ同じように見えます。

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

以下では、[**モデルのトレーニングに同じハイパーパラメータを使用する**] 通常どおり、高レベル API バリアントはコードが C++ または CUDA にコンパイルされているのに対し、カスタム実装は Python によって解釈される必要があるため、実行速度がはるかに速くなることに注意してください。

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## 論争

直感的に、バッチ正規化は最適化ランドスケープをよりスムーズにすると考えられています。しかし、ディープモデルをトレーニングするときに観察する現象については、投機的直感と真の説明を区別するように注意する必要があります。そもそもなぜ単純なディープニューラルネットワーク（MLPや従来のCNN）がうまく一般化されるのか分からないことを思い出してください。ドロップアウトや体重減少があっても、目に見えないデータに一般化する能力は、従来の学習論的汎化保証では説明できないほど柔軟性があります。 

著者らは、バッチ正規化を提案した元の論文で、強力で有用なツールを紹介したことに加え、*内部共変量シフト*を減らすことによって、なぜそれが機能するのかについて説明した。おそらく*内部共変量シフト*によって、著者は前述の直感のようなもの、つまり変数値の分布がトレーニングの過程で変化するという概念を意味していました。ただし、この説明には2つの問題がありました。i）このドリフトは*共変量シフト*とは大きく異なり、名前を誤った名称にします。ii）説明は十分に明記されていない直感を提供しますが、*なぜこの手法が正確に機能するのかという疑問は、厳密な説明を求める未解決の問題です。。この本を通して、私たちは実践者がディープニューラルネットワークの開発を導くために使用する直感を伝えることを目指しています。しかし、これらの誘導的直観を、確立された科学的事実から切り離すことが重要であると私たちは考えています。最終的には、この資料を習得して独自の研究論文を書き始めると、技術的な主張と勘の区別を明確にしたいと思うでしょう。 

バッチ正規化が成功した後、*内部共変量シフト*によるバッチ正規化の説明は、技術文献での議論や、機械学習研究の提示方法に関する広範な議論で繰り返し浮上しています。2017年のNeurIPSカンファレンスでTest of Time Awardを受賞した際に行われた記憶に残るスピーチで、アリ・ラヒミは現代のディープラーニングの実践を錬金術に例える議論の中で、*内部共変量シフト*を焦点として使用しました。その後、機械学習 :cite:`Lipton.Steinhardt.2018` の厄介な傾向を概説したポジションペーパーで、この例を詳細に再検討しました。他の著者らは、バッチ正規化の成功について別の説明を提案しており、一部の著者らは、元の論文 :cite:`Santurkar.Tsipras.Ilyas.ea.2018` で主張されているのとは逆の動作を示しているにもかかわらず、バッチ正規化が成功すると主張している。 

私たちは、*内部共変量シフト*は、技術的な機械学習の文献で毎年行われている何千もの同様に曖昧な主張のどれよりも批判に値するものではないことに留意します。おそらく、これらの議論の焦点としての共鳴は、ターゲットオーディエンスに対する幅広い認識のおかげです。バッチ正規化は、展開されているほぼすべての画像分類器に適用され、数万件の引用数に及ぶこの手法を紹介した論文で得られた、不可欠な方法であることが証明されています。 

## [概要

* モデルトレーニング中、バッチ正規化はミニバッチの平均と標準偏差を利用してニューラルネットワークの中間出力を連続的に調整し、ニューラルネットワーク全体の各層の中間出力の値がより安定するようにします。
* 全結合層と畳み込み層のバッチ正規化方法は少し異なります。
* ドロップアウト層と同様に、バッチ正規化層は学習モードと予測モードで計算結果が異なります。
* バッチ正規化には、主に正則化による多くの有益な副作用があります。一方、内部共変量シフトを減らすという本来の動機は有効な説明ではないようだ。

## 演習

1. バッチ正規化の前に、全結合層または畳み込み層からバイアスパラメーターを削除できますか？なぜ？
1. バッチ正規化を使用する場合と使用しない場合の LeNet の学習率を比較します。
    1. 学習とテストの精度の増加をプロットします。
    1. 学習率はどれくらい大きくできますか？
1. すべての層でバッチ正規化が必要ですか？それを試して？
1. ドロップアウトをバッチ正規化に置き換えることはできますか？行動はどのように変化しますか？
1. パラメーター `beta` と `gamma` を固定し、結果を観察して分析します。
1. 高レベル API の `BatchNorm` のオンラインドキュメントを参照して、バッチ正規化のための他のアプリケーションを確認してください。
1. リサーチのアイデア:適用できる他の正規化変換について考えてみよう。確率積分変換を適用できますか？フルランクの共分散推定はどうですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:

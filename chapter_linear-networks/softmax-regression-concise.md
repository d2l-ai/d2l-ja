# ソフトマックス回帰の簡潔な実装
:label:`sec_softmax_concise`

:numref:`sec_linear_concise` のディープラーニングフレームワーク (**線形回帰の実装がはるかに容易になった**) の (**同様に高レベル API **) (~~here~~) (またはそれ以上) は、分類モデルの実装に便利です。:numref:`sec_softmax_scratch` のように、Fashion-MNIST データセットに固執し、バッチサイズを 256 に保ちましょう。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
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
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## モデルパラメーターの初期化

:numref:`sec_softmax` で述べたように、[**softmax 回帰の出力層は完全結合層です。**] したがって、このモデルを実装するには、`Sequential` に 10 個の出力をもつ完全結合層を 1 つ追加するだけで済みます。繰り返しますが、`Sequential` は実際には必要ありませんが、ディープモデルを実装するときはどこにでもあるので、習慣を形成したほうがよいでしょう。繰り返しますが、重みをゼロ平均と標準偏差 0.01 でランダムに初期化します。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define the flatten
# layer to reshape the inputs before the linear layer in our network
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Softmax 実装の再検討
:label:`subsec_softmax-implementation-revisited`

前の :numref:`sec_softmax_scratch` の例では、モデルの出力を計算し、この出力をクロスエントロピー損失まで実行しました。数学的には、それは完全に合理的なことです。しかし、計算の観点からすると、べき乗は数値の安定性の問題の原因となる可能性があります。 

ソフトマックス関数は $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ を計算することを思い出してください。$\hat y_j$ は予測確率分布 $\hat{\mathbf{y}}$ の $j^\mathrm{th}$ 要素、$o_j$ はロジット $\mathbf{o}$ の $j^\mathrm{th}$ 要素です。$o_k$ の一部が非常に大きい (つまり、非常に正の) 場合、$\exp(o_k)$ は、特定のデータ型で取得できる最大数 (*オーバーフロー*) よりも大きい可能性があります。これにより、分母 (および/または分子) が `inf` (無限大) になり、$\hat y_j$ の場合は 0、`inf`、または `nan` (数値ではない) のいずれかに遭遇することになります。このような状況では、クロスエントロピーに対する明確な戻り値は得られません。 

これを回避する 1 つのトリックは、ソフトマックスの計算を続行する前に、すべての $o_k$ から $\max(o_k)$ を引くことです。この $o_k$ を定数係数でシフトしても softmax の戻り値は変わらないことがわかります。 

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$

減算と正規化のステップの後、$o_j - \max(o_k)$ の負の値が大きくなり、対応する $\exp(o_j - \max(o_k))$ が 0 に近い値になることがあります。これらは有限精度 (*アンダーフロー*) のためにゼロに丸められ、$\hat y_j$ がゼロになり、$\log(\hat y_j)$ は `-inf` になります。バックプロパゲーションの道を少し進むと、恐ろしい`nan`の結果のスクリーン一杯に直面するかもしれません。 

幸いなことに、指数関数を計算しているにもかかわらず、最終的にはその対数を取るつもりです (クロスエントロピー損失を計算するとき)。これら 2 つの演算子 softmax と crossentropy を組み合わせることで、逆伝播中に悩まされる可能性のある数値安定性の問題を回避できます。次の式に示すように、$\exp(o_j - \max(o_k))$ の計算は避け、$\log(\exp(\cdot))$ でキャンセルされるため、代わりに $o_j - \max(o_k)$ を直接使用できます。 

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

モデルで出力確率を評価したい場合に備えて、従来のソフトマックス関数を手元に置いておきます。しかし、ソフトマックスの確率を新しい損失関数に渡す代わりに、["LogsumExp trick"](https://en.wikipedia.org/wiki/LogSumExp) のようなスマートな処理を行う [**クロスエントロピー損失関数内でロジットを渡し、ソフトマックスとその対数を一度に計算する**] だけにします。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## 最適化アルゴリズム

ここでは、最適化アルゴリズムとして学習率 0.1 で (**ミニバッチ確率的勾配降下法**)。これは線形回帰の例で適用したものと同じで、オプティマイザの一般的な適用性を示しています。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## 訓練

次に :numref:`sec_softmax_scratch` で [**定義されたトレーニング関数を呼び出します**](~~以前~~)、モデルをトレーニングします。

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

以前と同様、このアルゴリズムは、今回は以前よりも少ないコード行数ではありますが、適切な精度を実現する解に収束します。 

## [概要

* 高レベル API を使用すると、softmax 回帰をより簡潔に実装できます。
* 計算の観点から見ると、ソフトマックス回帰の実装には複雑さがあります。多くの場合、ディープラーニングフレームワークでは、数値の安定性を確保するために、これらのよく知られたトリック以外にも追加の予防措置が講じられていることに注意してください。これにより、実際にすべてのモデルをゼロからコーディングしようとした場合に遭遇する落とし穴がさらに増えるのを防ぐことができます。

## 演習

1. バッチサイズ、エポック数、学習率などのハイパーパラメーターを調整して、結果を確認します。
1. 学習のエポック数を増やします。しばらくするとテストの精度が低下するのはなぜですか？どうやってこれを直せる？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:

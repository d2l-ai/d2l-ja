# アテンションプーリング:Nadaraya-Watson カーネル回帰
:label:`sec_nadaraya-watson`

これで、:numref:`fig_qkv` のフレームワークにおけるアテンションメカニズムの主要コンポーネントがわかりました。要約すると、クエリ (意志的キュー) とキー (非意志的キュー) の間の相互作用は、*アテンションプーリング*という結果になります。アテンションプーリングは、値 (感覚入力) を選択的に集約して出力を生成します。このセクションでは、アテンション・プーリングについて詳しく説明し、アテンション・メカニズムが実際にどのように機能するのかを高レベルで把握できるようにします。具体的には、1964 年に提案された Nadaraya-Watson カーネル回帰モデルは、アテンションメカニズムを備えた機械学習を実証するための、シンプルでありながら完全な例です。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
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
tf.random.set_seed(seed=1322)
```

## [**データセットの生成**]

簡単にするために、次の回帰問題を考えてみましょう。入出力ペアのデータセット $\{(x_1, y_1), \ldots, (x_n, y_n)\}$、新しい入力 $x$ の出力 $\hat{y} = f(x)$ を予測するための $f$ の学習方法を考えてみましょう。 

ここでは、ノイズ項 $\epsilon$ をもつ次の非線形関数に従って、人工データセットを生成します。 

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

$\epsilon$ は、平均がゼロで標準偏差が 0.5 の正規分布に従います。50 個のトレーニング例と 50 個のテスト例の両方が生成されます。後で注意のパターンをよりよく視覚化するために、トレーニング入力がソートされます。

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab tensorflow
n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
```

```{.python .input}
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab pytorch
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

```{.python .input}
#@tab tensorflow
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal((n_train,), 0.0, 0.5)  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

次の関数は、すべての学習例 (円で表される)、ノイズ項を含まないグラウンドトゥルースデータ生成関数 `f` (「Truth」のラベル付き)、学習済みの予測関数 (「Pred」のラベル) をプロットします。

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## 平均プーリング

まず、この回帰問題に対する世界で最も馬鹿げた推定量から始めます。平均プーリングを使用して、すべてのトレーニング出力を平均化します。 

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

これは下にプロットされています。ご覧のとおり、この推定器はそれほどスマートではありません。

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```

## [**ノンパラメトリック注意プーリング**]

明らかに、平均プーリングでは入力 $x_i$ が省略されます。Nadaraya :cite:`Nadaraya.1964` と Watson :cite:`Watson.1964` は、入力位置に応じて出力 $y_i$ を計量する、より優れたアイディアを提案しました。 

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-watson`

$K$ は*カーネル* です。:eqref:`eq_nadaraya-watson` の推定器は*Nadaraya-Watson カーネル回帰* と呼ばれています。ここでは、カーネルの詳細については説明しません。:numref:`fig_qkv`の注意メカニズムの枠組みを思い出してください。アテンションという観点からは、:eqref:`eq_nadaraya-watson` をより一般化された*アテンション・プーリング*の形式で書き換えることができます。 

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`

$x$ はクエリ、$(x_i, y_i)$ はキーと値のペアです。:eqref:`eq_attn-pooling` と :eqref:`eq_avg-pooling` を比較すると、ここでのアテンションプーリングは値 $y_i$ の加重平均です。:eqref:`eq_attn-pooling` の*アテンションウェイト* $\alpha(x, x_i)$ は、クエリ $x$ と $\alpha$ によってモデル化されたキー $x_i$ の間の相互作用に基づいて、対応する値 $y_i$ に割り当てられます。どのクエリでも、すべてのキーと値のペアに対するアテンションの重みは有効な確率分布です。これらは負ではなく、合計で 1 になります。 

アテンションプーリングの直感を得るには、次のように定義されている*Gaussian kernel* を考えてみてください。 

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$

ガウスカーネルを :eqref:`eq_attn-pooling` と :eqref:`eq_nadaraya-watson` に接続すると、 

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian`

:eqref:`eq_nadaraya-watson-gaussian` では、指定されたクエリ $x$ に近いキー $x_i$ が取得されます。
*キーの対応する値 $y_i$ にアテンションウェイトを大きくすることで、より多くのアテンション*

特に、Nadaraya-Watson カーネル回帰はノンパラメトリックモデルであるため、:eqref:`eq_nadaraya-watson-gaussian` は*ノンパラメトリックアテンションプーリング* の一例です。以下では、このノンパラメトリック注意モデルに基づいて予測をプロットします。予測されたラインは、平均プーリングによって生成されるラインよりも滑らかで、グラウンドトゥルースに近いです。

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the 
# same testing inputs (i.e., same queries)
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# Each element of `y_hat` is weighted average of values, where weights are attention weights
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))
plot_kernel_reg(y_hat)
```

それでは、[**アテンションウェイト**] を見てみましょう。ここで、テスト入力はクエリであり、トレーニング入力はキーです。両方の入力がソートされているので、クエリとキーのペアが近いほど、アテンションプーリングのアテンションウェイトが高くなることがわかります。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## **パラメトリックアテンションプーリング**

ノンパラメトリック Nadaraya-Watson カーネル回帰には、*一貫性* という利点があります。十分なデータがあれば、このモデルは最適解に収束します。それでも、学習可能なパラメーターをアテンションプーリングに簡単に統合できます。 

たとえば、:eqref:`eq_nadaraya-watson-gaussian` とは少し異なります。次の例では、クエリ $x$ とキー $x_i$ の間の距離に、学習可能なパラメーター $w$ が乗算されます。 

$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-watson-gaussian-para`

このセクションの残りの部分では、:eqref:`eq_nadaraya-watson-gaussian-para` のアテンションプーリングのパラメーターを学習して、このモデルをトレーニングします。 

### バッチマトリックス乗算
:label:`subsec_batch_dot`

ミニバッチの注意をより効率的に計算するために、ディープラーニングフレームワークが提供するバッチ行列乗算ユーティリティを活用できます。 

最初のミニバッチには形状 $a\times b$ の行列 $\mathbf{X}_1, \ldots, \mathbf{X}_n$ が含まれており、2 番目のミニバッチには形状 $b\times c$ の行列 $n$ が含まれているとします。バッチマトリックス乗算の結果、$n$ の行列 $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ が $a\times c$ という形状になります。したがって、[**形状 ($n$、$a$、$b$) と ($n$、$b$、$b$、$c$) の 2 つのテンソルが与えられた場合、バッチ行列乗算出力の形状は ($n$、$a$、$c$) になります。**]

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

```{.python .input}
#@tab tensorflow
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape
```

注意メカニズムの文脈では、[**ミニバッチ行列乗算を使用して、ミニバッチ内の値の加重平均を計算できます。**]

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
#@tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

### モデルを定義する

ミニバッチ行列乗算を使用して、:eqref:`eq_nadaraya-watson-gaussian-para` の [**パラメトリックアテンションプーリング**] に基づいて、Nadaraya-Watson カーネル回帰のパラメトリックバージョンを定義します。

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

```{.python .input}
#@tab tensorflow
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))
        
    def call(self, queries, keys, values, **kwargs):
        # For training queries are `x_train`. Keys are distance of taining data for each point. Values are `y_train`.
        # Shape of the output `queries` and `attention_weights`: (no. of queries, no. of key-value pairs)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```

### 訓練

以下では、[**トレーニングデータセットをキーと値に変換**] して、アテンションモデルをトレーニングします。パラメトリックアテンションプーリングでは、トレーニング入力は、自身を除くすべてのトレーニング例からキーと値のペアを取り、出力を予測します。

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

```{.python .input}
#@tab tensorflow
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
```

二乗損失と確率的勾配降下法を用いて、[**パラメトリック注意モデルに学習をさせる**]。

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab tensorflow
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```

パラメトリックアテンションモデルに学習をさせた後、[**その予測をプロット**] できます。学習データセットをノイズで近似しようとすると、予測された線は、先にプロットされたノンパラメトリック線よりも滑らかではありません。

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab tensorflow
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# Shape of `value`: (`n_test`, `n_train`)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

ノンパラメトリック注意プーリングと比較すると、学習可能でパラメトリックな設定で [**注意の重みの大きい領域がシャープになる**]。

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab tensorflow
d2l.show_heatmaps(tf.expand_dims(tf.expand_dims(net.attention_weights, axis=0), axis=0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## [概要

* Nadaraya-Watson カーネル回帰は、注意メカニズムを備えた機械学習の一例です。
* Nadaraya-Watson カーネル回帰のアテンションプーリングは、学習出力の加重平均です。アテンションという観点から、アテンションウェイトは、クエリの関数と、その値とペアになったキーに基づいて、値に割り当てられます。
* アテンションプーリングは、ノンパラメトリックでもパラメトリックでもかまいません。

## 演習

1. トレーニング例の数を増やします。ノンパラメトリック Nadaraya-Watson カーネル回帰をもっとよく学べますか？
1. パラメトリックアテンションプーリング実験で学習した $w$ の価値は何ですか？アテンションウェイトを視覚化すると、ウェイト付けされた領域がシャープになるのはなぜですか？
1. ノンパラメトリック Nadaraya-Watson カーネル回帰にハイパーパラメーターを追加して、より良い予測を行うにはどうすればよいでしょうか。
1. このセクションのカーネル回帰に対して、別のパラメトリックアテンションプーリングを設計します。この新しいモデルに学習をさせ、注意の重みを可視化します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:

# ソフトマックス回帰のゼロからの実装
:label:`sec_softmax_scratch`

(**線形回帰をゼロから実装したように、**) ソフトマックス回帰も同様に基本的であり、(**あなたは**の残酷な詳細を知っておくべきです) (~~softmax regression~~) そしてそれを自分でどのように実装するか。:numref:`sec_fashion_mnist` で導入されたばかりの Fashion-MNIST データセットを使用して、バッチサイズ 256 のデータイテレータを設定します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## モデルパラメーターの初期化

線形回帰の例と同様に、ここでの各例は固定長ベクトルで表されます。生データセットの各例は $28 \times 28$ イメージです。このセクションでは [**各画像を平坦化し、長さ 784 のベクトルとして扱う**] 今後は、画像の空間構造を利用するためのより洗練された戦略について説明する予定ですが、ここでは各ピクセル位置を単なる別の特徴として扱います。 

softmax 回帰では、クラス数と同じ数の出力があることを思い出してください。(**データセットには 10 個のクラスがあるため、ネットワークの出力次元は 10.**) したがって、重みは $784 \times 10$ 行列を構成し、バイアスは $1 \times 10$ 行ベクトルを構成します。線形回帰と同様に、重み `W` をガウスノイズとバイアスで初期化し、初期値 0 を取ります。

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Softmax オペレーションの定義

ソフトマックス回帰モデルを実装する前に、:numref:`subseq_lin-alg-reduction` と :numref:`subseq_lin-alg-non-reduction` で説明したように、和演算子がテンソルの特定の次元に沿ってどのように機能するかを簡単に確認しておきましょう。[**行列 `X` を指定すると、すべての要素 (デフォルト) または同じ軸の要素のみを合計できます。**] つまり、同じ列 (軸 0) または同じ行 (軸 1) です。`X` が形状 (2, 3) のテンソルで、列を合計した場合、結果は形状 (3,) をもつベクトルになることに注意してください。sum 演算子を呼び出すときに、合計した次元を折りたたむのではなく、元のテンソルの軸数を維持するように指定できます。これにより、形状 (1, 3) の 2 次元テンソルになります。

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

これで準備が整いました (**softmax 操作を実装する**)。softmax は 3 つのステップで構成されていることを思い出してください:(i) 各項を累乗する (`exp` を使用)、(ii) 各行を合計して (バッチには例ごとに 1 つの行がある)、(iii) 各行を正規化定数で除算し、結果の合計が 1 になるようにします。コードを見る前に、これがどのように方程式で表されるかを思い出してください。 

(** $\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$ドル)
**)

分母 (正規化定数) は、*分割関数* と呼ばれることもあります (その対数は対数分割関数と呼ばれます)。その名前の由来は [統計物理学](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics) にあります。関連する方程式は、粒子のアンサンブル上の分布をモデル化しています。

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

ご覧のとおり、任意のランダム入力に対して、[**各要素を非負の数に変換します。さらに、各行の合計は、確率の要求に応じて最大 1, **] になります。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

これは数学的には正しいように見えますが、行列の要素が大きいか非常に小さいため、数値のオーバーフローやアンダーフローに対する予防策を講じられなかったため、実装が少しずさんでした。 

## モデルを定義する

softmax 演算を定義したので、[**softmax 回帰モデルを実装する**] 次のコードは、ネットワークを介して入力を出力にどのようにマッピングするかを定義します。モデルにデータを渡す前に、関数 `reshape` を使用して、バッチ内の各元のイメージをベクトルに平坦化することに注意してください。

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## 損失関数の定義

次に、:numref:`sec_softmax` で紹介されたクロスエントロピー損失関数を実装する必要があります。現時点では、分類問題は回帰問題よりはるかに多いため、これはすべての深層学習で最も一般的な損失関数です。 

クロスエントロピーは、真のラベルに割り当てられた予測確率の負の対数尤度をとることを思い出してください。Python の for ループ (非効率になりがちです) で予測を反復処理するのではなく、1 つの演算子ですべての要素を選択することができます。以下では、[**3 つのクラスの予測確率の 2 つの例とそれに対応するラベル `y` を含む標本データ `y_hat` を作成します] `y` では、最初の例では最初のクラスが正しい予測であり、2 番目の例では 3 番目のクラスがグラウンドトゥルースであることがわかっています。[**`y_hat` の確率の指標として `y` を使用, **] 最初の例では最初のクラスの確率を、2 番目の例では第 3 クラスの確率を選びます。

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

これで、たった一行のコードで効率的に (**クロスエントロピー損失関数を実装する**) ことができます。

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## 分類精度

予測確率分布 `y_hat` を考えると、ハード予測を出力する必要がある場合は常に、予測確率が最も高いクラスを選択します。実際、多くのアプリケーションでは選択が必要です。Gmail では、メールを [プライマリ]、[ソーシャル]、[更新]、[フォーラム] に分類する必要があります。内部で確率を推定するかもしれませんが、一日の終わりにはクラスの中から一つを選ばなければなりません。 

予測がラベルクラス `y` と一致する場合、予測は正しいです。分類精度は、正しいすべての予測の比率です。精度を直接最適化することは難しい (微分できない) 場合がありますが、私たちが最も重視するのはパフォーマンス指標であることが多く、分類器の学習時にはほぼ必ず報告します。 

精度を計算するために、次の操作を行います。まず、`y_hat` が行列の場合、2 番目の次元には各クラスの予測スコアが格納されていると仮定します。`argmax` を使用して、各行で最も大きいエントリのインデックスによって予測されるクラスを取得します。次に、[**予測されたクラスとグラウンドトゥルースの `y` を要素ごとに比較します。**] 等価演算子 `==` はデータ型に敏感であるため、`y_hat` のデータ型を `y` のデータ型に一致するように変換します。結果は 0 (偽) と 1 (真) のエントリを含むテンソルになります。合計を取ると、正しい予測の数が算出されます。

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

予測確率分布とラベルとして、前に定義した変数 `y_hat` と `y` を引き続き使用します。最初の例の予測クラスは 2 (行の最大要素はインデックス 2 の 0.6) で、実際のラベル 0 と矛盾していることがわかります。2 番目の例の予測クラスは 2 (行の最大要素はインデックス 2 で 0.5) で、これは実際のラベル 2 と一致します。したがって、これら 2 つの例の分類精度率は 0.5 です。

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

[**同様に、データセット上の任意のモデル `net` の精度を評価できます**] データイテレータ `data_iter` を介してアクセスします。

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

ここで `Accumulator` は、複数の変数にわたって合計を累積するユーティリティクラスです。上記の `evaluate_accuracy` 関数では、正しい予測数と予測数の両方をそれぞれ格納するために、`Accumulator` インスタンスに 2 つの変数を作成します。両方とも、データセットを反復処理するにつれて、時間の経過とともに累積されます。

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

[**`net` モデルをランダムな重みで初期化したため、このモデルの精度はランダム推測、**] に近いはずです。つまり、10 クラスで 0.1 になります。

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## 訓練

:numref:`sec_linear_scratch` の線形回帰の実装を読めば、softmax 回帰の [**トレーニングループ**] は驚くほど馴染みのあるものになるはずです。ここでは、再利用できるように実装をリファクタリングします。まず、1 エポックで学習させる関数を定義します。`updater` はモデルパラメーターを更新する一般的な関数で、バッチサイズを引数として受け取ります。`d2l.sgd` 関数のラッパーか、フレームワークに組み込まれている最適化関数のいずれかになります。

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.sum().backward()
            updater.step()
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

トレーニング関数の実装を示す前に、[**データをアニメーションでプロットするユーティリティクラス**] を定義します。このクラスは、本書の残りの部分でコードを単純化することを目的としています。

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

[~~The training function ~~] 次のトレーニング関数は、`num_epochs` で指定された複数のエポックに対して `train_iter` を介してアクセスされるトレーニングデータセットでモデル `net` をトレーニングします。各エポックの終わりに、`test_iter` 経由でアクセスされるテスト用データセットでモデルが評価されます。`Animator` クラスを活用して、トレーニングの進捗状況を可視化します。

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

ゼロからの実装として、:numref:`sec_linear_scratch` で定義された [**ミニバッチ確率的勾配降下法を使用する**]、学習率 0.1 でモデルの損失関数を最適化します。

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

ここで [**10 エポックでモデルをトレーニングします。**] エポック数 (`num_epochs`) と学習率 (`lr`) はどちらも調整可能なハイパーパラメーターであることに注意してください。これらの値を変更することで、モデルの分類精度を高めることができる場合があります。

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## 予測

トレーニングが完了したので、モデルは [**いくつかの画像を分類**] する準備が整いました。一連の画像がある場合、実際のラベル (テキスト出力の 1 行目) とモデルからの予測 (テキスト出力の 2 行目) を比較します。

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## [概要

* softmax 回帰を使用すると、マルチクラス分類用のモデルを学習させることができます。
* ソフトマックス回帰の学習ループは線形回帰の学習ループとよく似ています。データの取得と読み取り、モデルと損失関数の定義、最適化アルゴリズムを使用したモデルの学習です。すぐにわかるように、ほとんどの一般的なディープラーニングモデルには同様のトレーニング手順があります。

## 演習

1. このセクションでは、softmax 演算の数学的定義に基づいて softmax 関数を直接実装しました。これはどのような問題を引き起こす可能性がありますか？ヒント:$\exp(50)$ のサイズを計算してみてください。
1. このセクションの関数 `cross_entropy` は、クロスエントロピー損失関数の定義に従って実装されています。この実装では何が問題になりますか？ヒント:対数の領域を考えてみましょう。
1. 上記の2つの問題を解決するには、どのような解決策が考えられますか？
1. 最も可能性の高いラベルを返すのは常に良い考えですか？例えば、医療診断のためにこれを行いますか？
1. ソフトマックス回帰を使用して、いくつかの特徴に基づいて次の単語を予測すると仮定します。大きな語彙から生じる可能性のある問題は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:

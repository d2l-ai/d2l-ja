# 多層パーセプトロンのゼロからの実装
:label:`sec_mlp_scratch`

多層パーセプトロン (MLP) を数学的に特徴付けたので、自分で実装してみましょう。ソフトマックス回帰 (:numref:`sec_softmax_scratch`) で達成した以前の結果と比較するために、Fashion-MNIST 画像分類データセット (:numref:`sec_fashion_mnist`) を引き続き使用します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
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

Fashion-MNist には 10 個のクラスが含まれており、各イメージはグレースケールピクセル値の $28 \times 28 = 784$ グリッドで構成されていることを思い出してください。ここでも、ピクセル間の空間構造は無視するので、784 個の入力フィーチャと 10 個のクラスを含む単純な分類データセットと考えることができます。はじめに、[**隠れ層が 1 つ、隠れ単位が 256 個の MLP を実装します。**] これら両方の量をハイパーパラメーターと見なすことができます。通常、レイヤーの幅は 2 の累乗で選択しますが、ハードウェアでのメモリの割り当て方法とアドレス指定方法により、計算効率が高くなる傾向があります。 

ここでも、パラメータをいくつかのテンソルで表します。*すべての層* について、1 つの重み行列と 1 つのバイアスベクトルを追跡する必要があることに注意してください。いつものように、これらのパラメータに関して損失の勾配にメモリを割り当てます。

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## アクティベーション機能

すべてがどのように機能するかを確実に知るために、組み込みの `relu` 関数を直接呼び出すのではなく、max 関数を使って [**ReLU アクティベーションを実装**] します。

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## モデル

空間構造を無視しているので、各 2 次元イメージを `reshape` の長さの `num_inputs` の平面ベクトルにします。最後に、わずか数行のコードで (**モデルを実装**) します。

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## 損失関数

数値の安定性を確保するため、また softmax 関数をゼロから実装しているため (:numref:`sec_softmax_scratch`)、ソフトマックス損失とクロスエントロピー損失の計算には、高レベル API からの積分関数を活用しています。:numref:`subsec_softmax-implementation-revisited` のこれらの複雑さについての以前の議論を思い出してください。興味のある読者には、損失関数のソースコードを調べて、実装の詳細についての知識を深めることをお勧めします。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## 訓練

幸いなことに、[**MLP の学習ループはソフトマックス回帰の場合とまったく同じです。**] `d2l` パッケージをもう一度活用して `train_ch3` 関数 (:numref:`sec_softmax_scratch` を参照) を呼び出し、エポック数を 10、学習率を 0.1 に設定します。

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

学習したモデルを評価するために、[**テストデータに適用する**]。

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## [概要

* 単純な MLP の実装は、手動で行う場合でも簡単であることがわかりました。
* しかし、レイヤーの数が多いと、MLP をゼロから実装するのは面倒です (たとえば、モデルのパラメーターの命名や追跡など)。

## 演習

1. ハイパーパラメータ `num_hiddens` の値を変更して、このハイパーパラメータが結果にどのように影響するかを確認します。このハイパーパラメータの最適値を決定し、他の値をすべて一定に保ちます。
1. 非表示レイヤーを追加して、結果にどのような影響があるかを確認します。
1. 学習率を変更すると、結果にどのような影響がありますか？モデルアーキテクチャとその他のハイパーパラメータ (エポック数を含む) を修正した場合、どの学習率で最良の結果が得られますか?
1. すべてのハイパーパラメータ（学習率、エポック数、隠れ層の数、層あたりの隠れユニット数）を合わせて最適化すると、どのような結果が得られますか？
1. 複数のハイパーパラメータを扱うのがはるかに難しい理由を説明する。
1. 複数のハイパーパラメータに対する検索を構造化するために考えられる最も賢い戦略は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:

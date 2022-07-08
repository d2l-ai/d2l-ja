# ナイーブ・ベイズ
:label:`sec_naive_bayes`

これまでのセクションを通して、確率論と確率変数について学びました。この理論を機能させるために、*naive Bayes* 分類器を導入しましょう。これは、数字の分類を実行するために確率論的ファンダメンタルズだけを使います。 

学習とは、すべての仮定を作ることです。これまでに見たことのない新しいデータ例を分類する場合、どのデータ例が互いに類似しているかについていくつかの仮定をしなければなりません。単純ベイズ分類器は、一般的で非常に明確なアルゴリズムであり、計算を簡略化するためにすべての特徴が互いに独立していると仮定しています。このセクションでは、このモデルを適用して画像内の文字を認識します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 光学式文字認識

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` は、広く使用されているデータセットの 1 つです。これには、学習用に 60,000 個のイメージ、検証用に 10,000 個のイメージが含まれています。各画像には、0 から 9 までの手書きの数字が含まれています。このタスクは、各画像を対応する数字に分類することです。 

Gluon は `data.vision` モジュールに `MNIST` クラスを提供し、インターネットからデータセットを自動的に取得します。その後、Gluon はダウンロード済みのローカルコピーを使用します。パラメーター `train` の値を `True` または `False` に設定して、トレーニングセットとテストセットのどちらをリクエストするかを指定します。各イメージは、幅と高さの両方が $28$ で、シェイプ ($28$,$28$,$1$) のグレースケールイメージです。カスタマイズした変換を使用して、最後のチャネルディメンションを削除します。また、データセットは符号なし $8$ ビット整数で各ピクセルを表します。問題を単純化するために、二項特徴量に量子化します。

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

画像と対応するラベルを含む特定の例にアクセスできます。

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

この例では `image` という変数に格納され、高さと幅が $28$ ピクセルのイメージに対応しています。

```{.python .input}
#@tab all
image.shape, image.dtype
```

このコードでは、各イメージのラベルがスカラーとして格納されます。その型は $32$ ビット整数です。

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

また、複数の例に同時にアクセスすることもできます。

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

これらの例を視覚化してみましょう。

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 分類のための確率論的モデル

分類タスクでは、例をカテゴリにマッピングします。この例はグレースケール $28\times 28$ イメージで、カテゴリは数字です。(詳細な説明については :numref:`sec_softmax` を参照してください。)分類タスクを表現する自然な方法の 1 つは、確率論的な質問です。フィーチャ (つまり、イメージピクセル) から最も可能性の高いラベルは何ですか?例のフィーチャを $\mathbf x\in\mathbb R^d$ で表し、ラベルを $y\in\mathbb R$ で表します。ここでの特徴はイメージピクセルで、$2$ 次元イメージを $d=28^2=784$ とラベルが数字になるようにベクトルに再形成できます。フィーチャが与えられたラベルの確率は $p(y  \mid  \mathbf{x})$ です。これらの確率 (この例では $y=0, \ldots,9$ の $p(y  \mid  \mathbf{x})$) を計算できる場合、分類器は次の式で与えられる予測 $\hat{y}$ を出力します。 

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

残念ながら、このためには $\mathbf{x} = x_1, ..., x_d$ の値ごとに $p(y  \mid  \mathbf{x})$ を見積もる必要があります。各フィーチャが $2$ の値の 1 つを取ることができると想像してください。たとえば、$x_1 = 1$ という機能は apple という単語が特定の文書に出現することを示し、$x_1 = 0$ はそうではないことを意味する場合があります。もし $30$ のバイナリ特徴があったら、$2^{30}$ (10億以上!) を分類する準備が必要であることを意味しています。入力ベクトル $\mathbf{x}$ の取り得る値です。 

また、学習はどこにありますか？対応するラベルを予測するために考えられるすべての例を見る必要がある場合、実際にはパターンを学習するのではなく、データセットを記憶しているだけです。 

## ナイーブ・ベイズ分類器

幸いなことに、条件付き独立性についていくつかの仮定をすることで、帰納的バイアスを導入し、比較的控えめなトレーニング例から一般化できるモデルを構築することができます。はじめに、ベイズの定理を使って分類器を次のように表現してみましょう。 

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

分母は正規化項 $p(\mathbf{x})$ であり、ラベル $y$ の値に依存しないことに注意してください。そのため、$y$ の異なる値で分子を比較することだけについて心配する必要があります。分母を計算するのが難しいと判明したとしても、分子を評価できれば、それを無視して逃げることができます。幸いなことに、正規化定数を回復したい場合でも可能です。$\sum_y p(y  \mid  \mathbf{x}) = 1$ 以降、正規化項はいつでも回復できます。 

それでは、$p( \mathbf{x}  \mid  y)$ に注目しましょう。確率の連鎖則を使うと、$p( \mathbf{x}  \mid  y)$という項を次のように表すことができます。 

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

それだけでは、この表現はそれ以上私たちを得ることはありません。それでも、おおよそ $2^d$ パラメータを推定する必要があります。しかし、*特徴量が条件付きで互いに独立していると仮定すると、この用語が$\prod_i p(x_i  \mid  y)$に簡略化され、予測変数が得られるため、突然、より良い形になります。 

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

$i$ と $y$ ごとに $p(x_i=1  \mid  y)$ を推定し、その値を $P_{xy}[i, y]$ に保存すると、$P_{xy}$ は $d\times n$ 行列で $n$ はクラスの数で $y\in\{1, \ldots, n\}$ になります。この値を使って $p(x_i = 0 \mid y)$ を推定することもできます。 

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{for } t_i = 0 .
\end{cases}
$$

また、$y$ ごとに $p(y)$ を推定し、$P_y[y]$ に保存します。$P_y$ は長さが $n$ のベクトルになります。それから、新しい例$\mathbf t = (t_1, t_2, \ldots, t_d)$については、 

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

どんな$y$にも合います。したがって、条件付き独立性の仮定により、モデルの複雑さは、フィーチャ数 $\mathcal{O}(2^dn)$ に対する指数関数的依存から $\mathcal{O}(dn)$ の線形依存性へと移行しました。 

## 訓練

今問題なのは、$P_{xy}$ と $P_y$ がわからないことです。したがって、最初にいくつかのトレーニングデータからそれらの値を推定する必要があります。これはモデルの*トレーニング*です。$P_y$ の見積もりはそれほど難しくありません。ここでは $10$ クラスのみを扱っているので、各桁の出現回数 $n_y$ を数え、それをデータの総量 $n$ で割ります。たとえば、数字の 8 が $n_8 = 5,800$ 回出現し、合計で $n = 60,000$ 枚の画像がある場合、確率推定値は $p(y=8) = 0.0967$ になります。

```{.python .input}
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

さあ、もう少し難しいことに移りましょう $P_{xy}$。白黒イメージを選択したので、$p(x_i  \mid  y)$ はクラス $y$ でピクセル $i$ がオンになる確率を表します。前と同じように、イベントが発生する回数$n_{iy}$を数えて、$y$の総発生回数、つまり$n_y$で割ることができます。しかし、少し厄介なことがあります。特定のピクセルが決して黒くならないかもしれません（例えば、よく切り取られた画像の場合、角のピクセルは常に白になるかもしれません）。統計学者がこの問題に対処する便利な方法は、すべての発生に疑似カウントを追加することです。したがって、$n_{iy}$ ではなく $n_{iy}+1$ を使用し、$n_y$ の代わりに $n_{y} + 1$ を使用します。これは*ラプラススムージング*とも呼ばれます。それはアドホックに見えるかもしれませんが、ベイズの観点からは十分に動機付けられているかもしれません。

```{.python .input}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 1), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

これらの $10\times 28\times 28$ の確率 (各クラスのピクセルごと) を可視化することで、平均に見える数字を得ることができます。 

これで :eqref:`eq_naive_bayes_estimation` を使用して新しいイメージを予測できます。$\mathbf x$ を指定すると、次の関数は $y$ ごとに $p(\mathbf x \mid y)p(y)$ を計算します。

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

これはひどく間違っていた！その理由を調べるために、ピクセルごとの確率を見てみましょう。通常は $0.001$ から $1$ までの数字です。私たちはそれらの$784$を掛けています。この時点で、これらの数値をコンピューターで計算しているため、指数の範囲が固定されていることに言及する価値があります。何が起こるかというと、*数値アンダーフロー*が発生します。つまり、小さい数値をすべて乗算すると、ゼロに切り捨てられるまでさらに小さい値になります。:numref:`sec_maximum_likelihood`ではこれを理論的な問題として議論しましたが、実際にはこの現象がはっきりとわかります。 

この節で説明したように、$\log a b = \log a + \log b$、つまり加算対数に切り替えるという事実を利用してこれを修正します。$a$ と $b$ の両方が小さい数値であっても、対数の値は適切な範囲内になければなりません。

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

対数は増加する関数なので、:eqref:`eq_naive_bayes_estimation` を次のように書き換えることができます。 

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

以下の安定版を実装できます。

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

ここで、予測が正しいかどうかを確認できます。

```{.python .input}
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

ここでいくつかの検証例を予測すると、ベイズ分類器がかなりうまく機能することがわかります。

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

最後に、分類器の全体的な精度を計算してみましょう。

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

最新のディープネットワークのエラー率は $0.01$ 未満です。パフォーマンスが比較的低いのは、モデルで行った統計的仮定が正しくないためです。ラベルのみに応じて、すべてのピクセルが「独立に」生成されると仮定しました。これは明らかに人間が数字を書く方法ではなく、この誤った仮定が私たちの過度にナイーブ（ベイズ）分類器の崩壊につながりました。 

## 概要 * ベイズの法則を使うと、観測されるすべての特徴が独立していると仮定して分類器を作成できます。* この分類器は、ラベルとピクセル値の組み合わせの出現回数を数えることで、データセットでトレーニングできます。* この分類器は、スパムなどのタスクで何十年もの間ゴールドスタンダードでした検出。 

## 演習

1. つの要素 $[0,1,1,0]$ の XOR によって与えられるラベルをもつデータセット $[[0,0], [0,1], [1,0], [1,1]]$ について考えてみます。このデータセットに基づいて構築されたナイーブベイズ分類器の確率はどれくらいですか。それは私たちのポイントをうまく分類していますか？そうでない場合、どのような仮定に違反しますか
1. 確率を推定する際にラプラス平滑化を使用せず、学習では観測されなかった値を含むデータ例が検定時に到達したとします。モデルは何を出力しますか
1. 単純ベイズ分類器は、確率変数の依存性がグラフ構造でエンコードされるベイジアンネットワークの具体例です。完全な理論はこの節の範囲外ですが (詳細については :cite:`Koller.Friedman.2009` を参照)、排他的論理和 (XOR) モデルで 2 つの入力変数間の明示的な依存を許可することで、分類器を正常に作成できる理由を説明してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:

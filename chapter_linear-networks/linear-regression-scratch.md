# 線形回帰のゼロからの実装
:label:`sec_linear_scratch`

線形回帰の背後にある重要な概念を理解できたので、コードでの実践的な実装に取り掛かることができます。このセクションでは (**データパイプライン、モデル、損失関数、ミニバッチ確率的勾配降下オプティマイザーなど、メソッド全体をゼロから実装します。**) 最新のディープラーニングフレームワークではこの作業のほとんどすべてを自動化できますが、ゼロから実装することが唯一の方法です自分が何をしているのか本当にわかっていることを確認するためです。さらに、モデルをカスタマイズしたり、独自のレイヤーや損失関数を定義したりするときは、内部で物事がどのように機能するかを理解すると便利です。このセクションでは、テンソルと自動微分にのみ依存します。その後、ディープラーニングフレームワークの特徴を生かして、より簡潔な実装を紹介します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## データセットの生成

単純化するために、[**加法性ノイズを含む線形モデルに従って人工データセットを構築する**] 私たちの仕事は、データセットに含まれる有限な例集合を使用して、このモデルのパラメーターを回復することです。データを低次元に保ち、簡単に視覚化できるようにします。次のコードスニペットでは、1000 個の例を含むデータセットを生成します。各サンプルは、標準正規分布からサンプリングされた 2 つの特徴量から構成されます。したがって、合成データセットは行列 $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$ になります。 

(**データセットを生成する真のパラメーターは $\mathbf{w} = [2, -3.4]^\top$ と $b = 4.2$、**) 合成ラベルは、ノイズ項 $\epsilon$ をもつ次の線形モデルに従って割り当てられます。 

(** $\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$ドル**) 

$\epsilon$ は、フィーチャとラベルの潜在的な計測誤差をキャプチャするものと考えることができます。標準的な仮定が成り立ち、$\epsilon$ は平均 0 の正規分布に従うと仮定します。問題を簡単にするために、標準偏差を 0.01 に設定します。次のコードは、合成データセットを生成します。

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

[**`features` の各行は 2 次元のデータ例で構成され、`labels` の各行は 1 次元のラベル値 (スカラー) で構成されていることに注意してください。**]

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

2 番目のフィーチャ `features[:, 1]` と `labels` を使用して散布図を生成すると、この 2 つのフィーチャ間の線形相関を明確に観察できます。

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## データセットの読み取り

モデルのトレーニングは、データセットに対して複数のパスを作成し、サンプルのミニバッチを一度に 1 つずつ取得し、それらを使用してモデルを更新することで構成されることを思い出してください。このプロセスは機械学習アルゴリズムのトレーニングにとって非常に重要なので、データセットをシャッフルしてミニバッチでアクセスするユーティリティ関数を定義する価値があります。 

以下のコードでは、[**`data_iter` 関数を定義**](~~that~~) して、この機能の 1 つの可能な実装を示します。関数 (**バッチサイズ、特徴の行列、ラベルのベクトルをとり、サイズ `batch_size` のミニバッチを生成する**) 各ミニバッチは、特徴量とラベルのタプルで構成されます。

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

一般的には、並列化操作に優れた GPU ハードウェアを活用するために、適度なサイズのミニバッチを使用することに注意してください。各例はモデルを通じて並列に供給でき、各例の損失関数の勾配も並列で取得できるため、GPU を使用すると、1 つの例を処理するよりも短時間で数百もの例を処理できます。 

直感を深めるために、データ例の最初の小さなバッチを読んで印刷してみましょう。各ミニバッチ内のフィーチャの形状から、ミニバッチのサイズと入力フィーチャの数の両方がわかります。同様に、ラベルのミニバッチは `batch_size` で指定された形状になります。

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

反復を実行すると、データセット全体が使い果たされるまで、個別のミニバッチが連続して取得されます (これを試してください)。上記で実装したイテレーションは教訓的な目的には適していますが、実際の問題でトラブルに巻き込まれるような点では非効率的です。たとえば、すべてのデータをメモリにロードし、大量のランダムメモリアクセスを実行する必要があります。ディープラーニングフレームワークに実装されたビルトインイテレーターは非常に効率的で、ファイルに格納されたデータとデータストリームを介して供給されるデータの両方を処理できます。 

## モデルパラメーターの初期化

[**モデルのパラメーターの最適化を始める前に**] ミニバッチ確率的勾配降下法 (**最初にいくつかのパラメーターが必要です**) 次のコードでは、平均 0、標準偏差 0.01 の正規分布から乱数をサンプリングして重みを初期化します。バイアスを 0 に設定します。

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

パラメーターを初期化したら、次のタスクは、データに十分適合するまでパラメーターを更新することです。更新のたびに、パラメーターに関する損失関数の勾配を取る必要があります。この勾配が与えられると、損失が減少する方向に各パラメータを更新できます。 

誰も勾配を明示的に計算したくないので (これは面倒でエラーが発生しやすい)、:numref:`sec_autograd` で導入された自動微分を使用して勾配を計算します。 

## モデルを定義する

次に、[**モデルを定義し、入力とパラメーターを出力に関連付ける**] 必要があります。線形モデルの出力を計算するには、入力フィーチャ $\mathbf{X}$ とモデルの重み $\mathbf{w}$ の行列-ベクトルドット積を取り、オフセット $b$ を各例に追加するだけです。$\mathbf{Xw}$ 以下はベクトルで、$b$ はスカラーであることに注意してください。:numref:`subsec_broadcasting` で説明されているブロードキャストメカニズムを思い出してください。ベクトルとスカラーを追加すると、ベクトルの各コンポーネントにスカラーが追加されます。

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## 損失関数の定義

[**モデルを更新するには損失関数の勾配を取る必要がある**] ので、(**損失関数を先に定義する**) 必要があります。ここでは :numref:`sec_linear_regression` で説明されている二乗損失関数を使用します。実装では、真の値 `y` を予測値のシェイプ `y_hat` に変換する必要があります。次の関数が返す結果も `y_hat` と同じ形になります。

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## 最適化アルゴリズムの定義

:numref:`sec_linear_regression` で説明したように、線形回帰には閉形式の解があります。しかし、これは線形回帰に関する本ではなく、ディープラーニングに関する本です。本書で紹介する他のモデルは解析的に解くことができないため、この機会にミニバッチ確率的勾配降下法の最初の実例を紹介します。[~~線形回帰には閉形式の解がありますが、本書の他のモデルにはありません。ここではミニバッチ確率的勾配降下法について紹介します。~~] 

各ステップで、データセットからランダムに抽出された1つのミニバッチを使用して、パラメーターに対する損失の勾配を推定します。次に、損失を減らす可能性のある方向にパラメータを更新します。次のコードは、一連のパラメーター、学習率、およびバッチサイズを指定して、ミニバッチの確率的勾配降下法の更新を適用します。更新ステップのサイズは、学習率 `lr` によって決まります。損失は例のミニバッチの合計として計算されるため、標準的なステップサイズの大きさがバッチサイズの選択に大きく依存しないように、ステップサイズをバッチサイズ (`batch_size`) で正規化します。

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## 訓練

これですべてのパーツが揃ったので、[**メインのトレーニングループを実装する**] 準備ができました。ディープラーニングのキャリアを通じて、ほぼ同じトレーニングループが何度も繰り返し見られるため、このコードを理解することが重要です。 

各反復で、トレーニング例のミニバッチを取得し、モデルに渡して一連の予測を取得します。損失を計算した後、ネットワークの逆方向パスを開始し、各パラメータに関する勾配を保存します。最後に、最適化アルゴリズム `sgd` を呼び出してモデルパラメーターを更新します。 

要約すると、次のループを実行します。 

* パラメーターを初期化する $(\mathbf{w}, b)$
* 完了するまで繰り返す
    * グラディエントを計算する $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新パラメータ $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

*epoch* ごとに、トレーニングデータセットのすべての例を通過した後、データセット全体 (`data_iter` 関数を使用) を反復処理します (例の数がバッチサイズで割り切れると仮定)。エポック数 `num_epochs` と学習率 `lr` はどちらもハイパーパラメーターで、ここではそれぞれ 3 と 0.03 に設定します。残念ながら、ハイパーパラメータの設定は難しく、試行錯誤による調整が必要です。ここではこれらの詳細は省略していますが、:numref:`chap_optimization` の後半で修正します。

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

この場合、データセットを自分で合成したため、真のパラメータが何であるかを正確に把握できます。したがって、トレーニングループを通じて [**真のパラメータと学習したパラメータを比較して、トレーニングの成功を評価する**] ことができます。実際、彼らはお互いに非常に近いことが分かります。

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

パラメータを完全に回復できるのは当然のことではないことに注意してください。しかし、機械学習では通常、真の基礎となるパラメーターの回復にはあまり関心がなく、高精度の予測につながるパラメーターへの関心が高まります。幸いなことに、困難な最適化問題であっても、確率的勾配降下法は非常に優れた解を見出すことがよくあります。これは、ディープネットワークでは、非常に正確な予測につながるパラメーターの構成が多数存在するためです。 

## [概要

* レイヤーの定義や高度なオプティマイザーを必要とせずに、テンソルと自動微分のみを使用して、ディープネットワークをゼロから実装して最適化する方法を確認しました。
* このセクションでは、可能なことの表面のみをスクラッチします。次のセクションでは、今紹介した概念に基づいた追加モデルについて説明し、より簡潔に実装する方法を学習します。

## 演習

1. 重みをゼロに初期化するとどうなるでしょうか。アルゴリズムはまだ機能しますか？
1. [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) が電圧と電流の間のモデルを考え出そうとしているとします。自動微分を使用してモデルのパラメーターを学習できるか
1. [プランクの法則](https://en.wikipedia.org/wiki/Planck%27s_law) を使って、スペクトルエネルギー密度を使って物体の温度を決定できますか？
1. 二次微分を計算する場合に遭遇する可能性のある問題は何ですか？どうやって直すの？
1.  `squared_loss` 関数に `reshape` 関数が必要なのはなぜですか？
1. さまざまな学習率を試して、損失関数の値がどれだけ速く低下するかを調べます。
1. 例の数をバッチサイズで割れない場合、`data_iter` 関数の動作はどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

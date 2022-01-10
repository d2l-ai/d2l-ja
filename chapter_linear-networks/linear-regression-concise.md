# 線形回帰の簡潔な実装
:label:`sec_linear_concise`

過去数年間、ディープラーニングに幅広く関心が高まっているため、企業、学者、愛好家は、勾配ベースの学習アルゴリズムを実装する反復作業を自動化するためのさまざまな成熟したオープンソースフレームワークを開発するようになりました。:numref:`sec_linear_scratch` では、(i) データストレージと線形代数にはテンソル、(ii) 勾配の計算には自動微分のみに依存していました。実際には、データイテレータ、損失関数、オプティマイザ、ニューラルネットワーク層が非常に一般的であるため、現代のライブラリでもこれらのコンポーネントが実装されています。 

このセクションでは、ディープラーニングフレームワークの :numref:`sec_linear_scratch` (**高レベル API を使用して簡潔に**) の (**線形回帰モデルの実装方法**) を紹介します。 

## データセットの生成

まず、:numref:`sec_linear_scratch` と同じデータセットを生成します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## データセットの読み取り

独自のイテレータをロールするのではなく、[**データを読み込むためにフレームワーク内の既存の API を呼び出す**] `features` と `labels` を引数として渡し、データイテレータオブジェクトをインスタンス化するときに `batch_size` を指定します。また、ブール値 `is_train` は、データイテレータオブジェクトが各エポックでデータをシャッフルする (データセットを通過する) かどうかを示します。

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

:numref:`sec_linear_scratch` で `data_iter` 関数を呼び出したのとほぼ同じ方法で `data_iter` を使うことができます。それが機能していることを確認するために、サンプルの最初のミニバッチを読んで印刷します。:numref:`sec_linear_scratch` と比較すると、ここでは `iter` を使用して Python イテレータを構築し、`next` を使用してイテレータから最初の項目を取得します。

```{.python .input}
#@tab all
next(iter(data_iter))
```

## モデルを定義する

:numref:`sec_linear_scratch` で線形回帰をゼロから実装したとき、モデルパラメーターを明示的に定義し、基本的な線形代数演算を使用して出力を生成するように計算をコーディングしました。あなたはこれを行う方法を知っているべきです*。しかし、モデルがより複雑になり、これをほぼ毎日行う必要があると、喜んで支援を受けることができます。この状況は、自分のブログをゼロからコーディングするのと似ています。それを1回か2回行うことはやりがいがあり、有益ですが、ブログを必要とするたびに車輪の再発明に1か月を費やしたら、お粗末なWeb開発者になるでしょう。 

標準的な操作では、[**フレームワークの定義済みレイヤーを使用**] できます。これにより、実装に集中するのではなく、特にモデルの構築に使用されるレイヤーに集中できます。最初に `Sequential` クラスのインスタンスを参照するモデル変数 `net` を定義します。`Sequential` クラスは、連鎖される複数のレイヤーのコンテナーを定義します。入力データが与えられると、`Sequential` インスタンスはそのデータを第 1 レイヤーに渡し、出力を 2 番目のレイヤーの入力として渡します。次の例では、モデルは 1 つのレイヤーのみで構成されているため、`Sequential` は実際には必要ありません。しかし、今後のモデルのほとんどすべてに複数のレイヤーが含まれるため、最も標準的なワークフローに慣れるためだけに使用します。 

:numref:`fig_single_neuron` に示された単層ネットワークのアーキテクチャを思い出してください。各入力が行列ベクトル乗算によって各出力に接続されているため、この層は*完全接続* であると言われます。

:begin_tab:`mxnet`
グルーオンでは、全結合層は `Dense` クラスで定義されています。1 つのスカラー出力のみを生成したいので、その数を 1 に設定します。 

便宜上、Gluonでは各レイヤーの入力形状を指定する必要がないことに注意してください。したがって、ここでは、この線形層に入る入力の数をグルーオンに伝える必要はありません。最初にモデルにデータを渡そうとしたとき、例えば `net(X)` を後で実行すると、Gluon は各レイヤーへの入力数を自動的に推測します。この仕組みについては、後ほど詳しく説明します。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、完全結合層は `Linear` クラスで定義されています。`nn.Linear` に 2 つの引数を渡したことに注意してください。1 つ目は入力フィーチャの次元 (2) を指定し、2 つ目は出力フィーチャの次元 (単一のスカラー、つまり 1) を指定します。
:end_tab:

:begin_tab:`tensorflow`
Keras では、完全結合層は `Dense` クラスで定義されています。1 つのスカラー出力のみを生成したいので、その数を 1 に設定します。 

便宜上、Kerasでは各レイヤーの入力形状を指定する必要がないことに注意してください。したがって、ここでは、この線形層に入る入力の数をKerasに伝える必要はありません。最初にモデルにデータを渡そうとしたとき、例えば `net(X)` を後で実行すると、Keras は各レイヤーへの入力数を自動的に推測します。この仕組みについては、後ほど詳しく説明します。
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## モデルパラメーターの初期化

`net` を使用する前に、線形回帰モデルの重みや偏りなど (**モデルパラメーターの初期化**) を行う必要があります。ディープラーニングフレームワークには、パラメーターの初期化方法が事前に定義されていることがよくあります。ここでは、平均が 0、標準偏差が 0.01 の正規分布から各重みパラメータをランダムにサンプリングするように指定します。bias パラメータは 0 に初期化されます。

:begin_tab:`mxnet`
`initializer` モジュールを MXNet からインポートします。このモジュールは、モデルパラメーターを初期化するためのさまざまなメソッドを提供します。Gluon は `init` を `initializer` パッケージにアクセスするためのショートカット (略称) として使用できるようにしています。重みの初期化方法を指定するのは `init.Normal(sigma=0.01)` を呼び出すことだけです。バイアスパラメータはデフォルトで 0 に初期化されます。
:end_tab:

:begin_tab:`pytorch`
`nn.Linear` を構築する際に入力次元と出力次元を指定したので、パラメータに直接アクセスして初期値を指定できるようになりました。まず、ネットワーク内の最初の層である `net[0]` によって層を特定し、`weight.data` および `bias.data` メソッドを使用してパラメーターにアクセスします。次に、置換メソッド `normal_` と `fill_` を使用してパラメーター値を上書きします。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow の `initializers` モジュールは、モデルパラメーターの初期化にさまざまな方法を提供します。Keras で初期化方法を指定する最も簡単な方法は、`kernel_initializer` を指定してレイヤーを作成するときです。ここで `net` をもう一度作り直します。
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
上記のコードは単純に見えるかもしれませんが、ここで何か奇妙なことが起きていることに注意してください。Gluon は入力の次元数をまだ把握していませんが、ネットワークのパラメーターを初期化しています。この例のように2になるか、2000になるかもしれません。Gluonは、舞台裏で初期化が実際には*延期*されるため、これを回避できます。実際の初期化は、初めてネットワークを介してデータを渡そうとしたときにのみ行われます。パラメータはまだ初期化されていないため、パラメータにアクセスしたり操作したりすることはできないことに注意してください。
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
上記のコードは単純に見えるかもしれませんが、ここで何か奇妙なことが起きていることに注意してください。Keras は入力の次元数をまだ把握していませんが、ネットワークのパラメーターを初期化しています。この例のように2になるか、2000になるかもしれません。Kerasはこれを回避することができます。なぜなら、舞台裏では初期化が実際には*延期*されるからです。実際の初期化は、初めてネットワークを介してデータを渡そうとしたときにのみ行われます。パラメータはまだ初期化されていないため、パラメータにアクセスしたり操作したりすることはできないことに注意してください。
:end_tab:

## 損失関数の定義

:begin_tab:`mxnet`
Gluon では、`loss` モジュールがさまざまな損失関数を定義しています。この例では、二乗損失の Gluon 実装 (`L2Loss`) を使用します。
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` クラスは平均二乗誤差を計算します (:eqref:`eq_mse` の係数 $1/2$ を除く)。**] デフォルトでは、例に対する平均損失が返されます。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` クラスは平均二乗誤差を計算します (:eqref:`eq_mse` では $1/2$ 係数を使用しない)。デフォルトでは、例に対する平均損失が返されます。
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## 最適化アルゴリズムの定義

:begin_tab:`mxnet`
ミニバッチ確率的勾配降下法はニューラルネットワークを最適化するための標準ツールであり、Gluon は `Trainer` クラスを通じてこのアルゴリズムのさまざまなバリエーションと共にこれをサポートしています。`Trainer` をインスタンス化するときは、最適化するパラメーター (`net.collect_params()` 経由でモデル `net` から取得可能)、使用する最適化アルゴリズム (`sgd`)、および最適化アルゴリズムに必要なハイパーパラメーターのディクショナリを指定します。ミニバッチ確率的勾配降下法では、値 `learning_rate` を設定するだけで、ここでは 0.03 に設定されます。
:end_tab:

:begin_tab:`pytorch`
ミニバッチ確率的勾配降下法はニューラルネットワークを最適化するための標準ツールであり、PyTorch は `optim` モジュールのこのアルゴリズムのさまざまなバリエーションと共にこれをサポートしています。(**`SGD` インスタンスをインスタンス化**) する際には、最適化アルゴリズムに必要なハイパーパラメーターのディクショナリを使用して、最適化するパラメーター (`net.parameters()` 経由でネットから取得可能) を指定します。ミニバッチ確率的勾配降下法では、値 `lr` を設定するだけで、ここでは 0.03 に設定されます。
:end_tab:

:begin_tab:`tensorflow`
ミニバッチ確率的勾配降下法はニューラルネットワークを最適化するための標準ツールであり、Keras は `optimizers` モジュールのこのアルゴリズムのさまざまなバリエーションと共にこれをサポートしています。ミニバッチ確率的勾配降下法では、値 `learning_rate` を設定するだけで、ここでは 0.03 に設定されます。
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## 訓練

ディープラーニングフレームワークの高レベル API を使用してモデルを表現するには、比較的少ないコード行しか必要としないことに気づいたかもしれません。パラメーターを個別に割り当てたり、損失関数を定義したり、ミニバッチ確率的勾配降下法を実装したりする必要はありませんでした。いったん複雑なモデルを扱うようになれば、高レベル API の利点はかなり大きくなるでしょう。しかし、いったん基本的な要素がすべて揃ったら、[**トレーニングループ自体は、すべてをゼロから実装したときと非常に似ています。**] 

メモリをリフレッシュするには:いくつかのエポックで、データセット (`train_data`) を完全に渡し、入力のミニバッチと対応するグラウンドトゥルースラベルを繰り返し取得します。ミニバッチごとに、次の儀式を行います。 

* `net(X)` を呼び出して予測を生成し、損失 `l` (順伝播) を計算します。
* バックプロパゲーションを実行して勾配を計算します。
* オプティマイザーを呼び出してモデルパラメーターを更新します。

良い尺度として、各エポック後に損失を計算し、それを出力して進行状況を監視します。

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

以下では、データセットを生成した [**有限データで学習したモデルパラメータと実パラメータを比較**] します。パラメーターにアクセスするには、まず `net` から必要な層にアクセスし、その層の重みとバイアスにアクセスします。ゼロからの実装と同様に、推定されたパラメーターは対応するグラウンドトゥルースに近いことに注意してください。

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## [概要

:begin_tab:`mxnet`
* Gluon を使えば、モデルをより簡潔に実装できます。
* Gluon では、`data` モジュールはデータ処理用のツールを提供し、`nn` モジュールは多数のニューラルネットワーク層を定義し、`loss` モジュールは多くの一般的な損失関数を定義します。
* MXNet のモジュール `initializer` は、モデルパラメーターの初期化にさまざまなメソッドを提供します。
* 次元と記憶域は自動的に推定されますが、初期化される前にパラメータにアクセスしようとしないよう注意してください。
:end_tab:

:begin_tab:`pytorch`
* PyTorch の高レベル API を使えば、モデルをより簡潔に実装できます。
* PyTorch では `data` モジュールはデータ処理用のツールを提供し、`nn` モジュールは多数のニューラルネットワーク層と共通の損失関数を定義します。
* パラメータの値を `_` で終わるメソッドに置き換えることで、パラメータを初期化できます。
:end_tab:

:begin_tab:`tensorflow`
* TensorFlow の高レベル API を使用することで、モデルをより簡潔に実装できます。
* TensorFlow では、`data` モジュールはデータ処理用のツールを提供し、`keras` モジュールは多数のニューラルネットワーク層と一般的な損失関数を定義します。
* TensorFlow のモジュール `initializers` は、モデルパラメーターの初期化のためのさまざまなメソッドを提供します。
* 次元と記憶域は自動的に推論されます (ただし、初期化される前にパラメーターにアクセスしようとしないよう注意してください)。
:end_tab:

## 演習

:begin_tab:`mxnet`
1. `l = loss(output, y)` を `l = loss(output, y).mean()` に置き換える場合、コードが同じように動作するように `trainer.step(batch_size)` を `trainer.step(1)` に変更する必要があります。なぜ？
1. モジュール `gluon.loss` および `init` で提供されている損失関数と初期化方法については、MXNet のドキュメントを参照してください。損失をフーバーの損失で置き換えます。
1. `dense.weight` のグラデーションにはどうやってアクセスしますか？

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. `nn.MSELoss(reduction='sum')` を `nn.MSELoss()` に置き換えた場合、コードの学習率を同じように変更するにはどうすればよいでしょうか。なぜ？
1. PyTorch のドキュメントを参照して、提供されている損失関数と初期化メソッドを確認してください。損失をフーバーの損失で置き換えます。
1. `net[0].weight` のグラデーションにはどうやってアクセスしますか？

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. TensorFlow のドキュメントを参照して、どのような損失関数と初期化方法が提供されているかを確認してください。損失をフーバーの損失で置き換えます。

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

# ドロップアウト
:label:`sec_dropout`

:numref:`sec_weight_decay` では、重みの $L_2$ ノルムにペナルティを課すことで、統計モデルを正則化する古典的なアプローチを導入しました。確率論的に言えば、重みは平均がゼロのガウス分布から値を取るという事前の信念を前提としていたと主張することで、この手法を正当化できます。より直感的に言えば、少数の疑似的な関連性に過度に依存しすぎないように、モデルがその重みを多数の特徴量に分散させるよう奨励したと主張するかもしれません。 

## オーバーフィットの再検討

例よりも多くの特徴量に直面すると、線形モデルは過適合になりがちです。しかし、特徴量よりも多くの例を挙げれば、一般に線形モデルは過適合にならないと期待できます。残念ながら、線形モデルが一般化する信頼性にはコストがかかります。単純に適用すると、線形モデルは特徴間の相互作用を考慮しません。線形モデルは、すべての特徴量について、コンテキストを無視して正または負の重みを割り当てなければなりません。 

従来のテキストでは、一般化可能性と柔軟性の間のこの根本的な緊張は、*バイアスと分散のトレードオフ*として説明されています。線形モデルはバイアスが高く、小さなクラスの関数しか表現できません。ただし、これらのモデルは分散が小さく、データのランダムサンプルが異なっても同様の結果が得られます。 

ディープニューラルネットワークは、バイアス分散スペクトルの反対側に存在します。ニューラルネットワークは、線形モデルとは異なり、各特徴を個別に調べることに限定されません。フィーチャグループ間の相互作用を学習できます。たとえば、電子メールに「ナイジェリア」と「ウエスタンユニオン」が一緒に表示されているのはスパムを示しているが、個別にはスパムではないと推測する場合があります。 

特徴量よりもはるかに多くの例がある場合でも、ディープニューラルネットワークは過適合する可能性があります。2017年、研究者のグループは、ランダムにラベル付けされた画像でディープネットをトレーニングすることで、ニューラルネットワークの非常に高い柔軟性を実証しました。入力を出力にリンクする真のパターンがないにもかかわらず、確率的勾配降下法によって最適化されたニューラルネットワークは、学習セット内のすべてのイメージに完全にラベルを付けることができることを発見しました。これが何を意味するのか考えてみてください。ラベルがランダムに一様に割り当てられ、クラスが 10 個ある場合、ホールドアウトデータの精度が 10% を超える分類器は存在しません。ここでの汎化ギャップはなんと 90% です。私たちのモデルが表現力豊かで、これがひどく過度にフィットする可能性がある場合、いつオーバーフィットしないと予想すべきですか？ 

ディープネットワークの不可解な汎化特性の数学的基礎は未解決の研究課題であり、理論志向の読者はこのトピックをより深く掘り下げることを奨励します。ここでは、ディープネットの一般化を実証的に改善する傾向がある実用的なツールの調査に移ります。 

## 摂動によるロバスト性

優れた予測モデルに期待されることについて簡単に考えてみましょう。私たちは、目に見えないデータでもうまく機能することを望んでいます。古典的な一般化理論は、訓練とテストの性能のギャップを埋めるためには、単純なモデルを目指すべきだと示唆しています。シンプルさは、少数の次元の形でもたらされます。:numref:`sec_model_selection` では、線形モデルの単項基底関数について論じるときに、このことを検討しました。さらに、:numref:`sec_weight_decay` で重みの減衰 ($L_2$ 正則化) について説明したときにわかったように、パラメーターの (逆) ノルムも簡略化の有効な尺度を表します。単純さのもう 1 つの有用な概念は、滑らかさです。つまり、関数は入力に対する小さな変化に敏感であってはならないということです。たとえば、画像を分類する場合、ピクセルにランダムノイズを追加してもほとんど無害であると予想されます。 

1995年、クリストファー・ビショップは、入力ノイズによるトレーニングがTikhonov正則化:cite:`Bishop.1995`と同等であることを証明したときに、この考えを形式化しました。この研究により、関数が滑らかである (したがって単純である) という要件と、入力の摂動に対して弾力性があるという要件との間に明確な数学的なつながりが描かれました。 

そして2014年、Srivastava et al. :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` は、ビショップのアイデアをネットワークの内部層にも適用する方法について巧妙なアイデアを開発しました。つまり、学習中に次の層を計算する前に、ネットワークの各層にノイズを注入することを提案しました。彼らは、多くの層を持つ深層ネットワークに学習させる場合、ノイズを注入すると入出力マッピングだけで滑らかさが強制されることに気付きました。 

*dropout* と呼ばれる彼らのアイデアは、順伝播中に各内部層を計算しながらノイズを注入することを含み、ニューラルネットワークを訓練するための標準的な手法となっています。この方法は*dropout*と呼ばれていますので、文字通り
*トレーニング中に一部のニューロンを脱落させる。
学習中、各反復で、標準ドロップアウトは、次の層を計算する前に、各層のノードの一部をゼロにすることで構成されます。 

明確にするために、私たちはビショップへのリンクで私たち自身の物語を押し付けています。ドロップアウトに関するオリジナルの論文は、有性生殖との驚くべき類推を通して直感を提供します。著者らは、ニューラルネットワークの過適合は、各層が前の層の特定の活性化パターンに依存し、この条件を「共適応」と呼んでいる状態によって特徴付けられると主張している。ドロップアウト, 彼らが主張する, 有性生殖が共適応遺伝子を破壊すると主張されているのと同じように、共適応を崩壊させる. 

ここで重要な課題は、このノイズをどのように注入するかです。1 つのアイディアは、各レイヤーの期待値が (他のレイヤーは固定しながら) ノイズがないと想定される値と等しくなるように、ノイズを*バイアスなし*の方法で注入することです。 

Bishopの研究では、線形モデルへの入力にガウスノイズを加えました。学習の反復ごとに、平均 0 $\epsilon \sim \mathcal{N}(0,\sigma^2)$ の分布からサンプリングされたノイズを入力の $\mathbf{x}$ に追加し、摂動点 $\mathbf{x}' = \mathbf{x} + \epsilon$ を生成しました。予想通り、$E[\mathbf{x}'] = \mathbf{x}$。 

標準のドロップアウト正則化では、保持された (ドロップアウトされていない) ノードの割合で正規化することで、各層のバイアスを除去します。つまり、*ドロップアウト確率* $p$ では、各中間活性化 $h$ は次のように確率変数 $h'$ に置き換えられます。 

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

設計上、期待値は変わりません、つまり $E[h'] = h$ です。 

## ドロップアウト・イン・プラクティス

:numref:`fig_mlp` の隠れ層と 5 つの隠れユニットを持つ MLP を思い出してください。隠れ層にドロップアウトを適用し、隠れユニットを確率 $p$ でゼロにすると、その結果は元のニューロンのサブセットのみを含むネットワークと見なすことができます。:numref:`fig_dropout2` では、$h_2$ と $h_5$ は削除されています。その結果、出力の計算が $h_2$ または $h_5$ に依存しなくなり、逆伝播を実行するとそれぞれの勾配も消滅します。このように、出力層の計算が $h_1, \ldots, h_5$ の 1 つの要素に過度に依存しすぎることはありません。 

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

通常、ドロップアウトはテスト時に無効にします。トレーニング済みのモデルと新しい例があれば、ノードをドロップアウトしないため、正規化する必要はありません。ただし、いくつかの例外があります。一部の研究者は、ニューラルネットワーク予測の*不確実性*を推定するためのヒューリスティックとしてテスト時にドロップアウトを使用します。予測が多数の異なるドロップアウトマスクで一致すれば、ネットワークの信頼性が高いと言えるかもしれません。 

## ゼロからの実装

単一層にドロップアウト関数を実装するには、層の次元数と同じ数のサンプルをベルヌーイ (バイナリ) 確率変数から引き出さなければなりません。ここで、確率変数は $1-p$ の値 $1$ (保持) と確率 $p$ の $0$ (drop) を取ります。これを実装する簡単な方法の 1 つは、一様分布 $U[0, 1]$ から標本を抽出することです。次に、対応するサンプルが $p$ より大きいノードを保持し、残りを削除できます。 

次のコードでは (** テンソル入力 `X` の要素を確率で `dropout` でドロップアウトする `dropout_layer` 関数を実装**)、上記のように余りを再スケーリングします:生存者を `1.0-dropout` で割ります。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

[**`dropout_layer` 関数をいくつかの例でテストできます**]。次のコード行では、入力 `X` をドロップアウト演算にそれぞれ確率 0、0.5、1 で渡しています。

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### モデルパラメーターの定義

ここでも、:numref:`sec_fashion_mnist` で導入された Fashion-MNIST データセットを使用します。[**それぞれ 256 単位を含む 2 つの隠れ層をもつ MLP を定義します**]

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### モデルを定義する

以下のモデルは、各隠れ層の出力にドロップアウトを適用します (アクティベーション関数に従う)。各層にドロップアウト確率を個別に設定できます。一般的な傾向として、ドロップアウト確率を低く設定すると、入力レイヤーに近づきます。以下では、1 番目と 2 番目の隠れレイヤーをそれぞれ 0.2 と 0.5 に設定します。ドロップアウトはトレーニング中のみ有効になるようにしています。

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### [**トレーニングとテスト**]

これは、前に説明した MLP のトレーニングとテストと似ています。

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [**簡潔な実装**]

高レベル API では、完全接続された各レイヤーの後に `Dropout` レイヤーを追加し、ドロップアウト確率をコンストラクターの唯一の引数として渡すだけで済みます。学習中、`Dropout` 層は、指定されたドロップアウト確率に従って、前の層の出力 (またはそれと同等に後続の層への入力) をランダムにドロップアウトします。トレーニングモードでない場合、`Dropout` 層はテスト中にデータを渡すだけです。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

次に、[**モデルのトレーニングとテスト**] を行います。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [概要

* ドロップアウトは、次元数と重みベクトルのサイズを制御するだけでなく、過適合を回避するためのもう 1 つのツールです。多くの場合、それらは共同で使用されます。
* ドロップアウトは、アクティベーション $h$ を予想値 $h$ の確率変数に置き換えます。
* ドロップアウトはトレーニング中にのみ使用されます。

## 演習

1. 第1層と第2層のドロップアウト確率を変更するとどうなりますか？特に、両方のレイヤーのものを切り替えるとどうなりますか？これらの質問に答える実験を計画し、結果を定量的に説明し、定性的な要点をまとめます。
1. エポック数を増やし、dropout を使用した場合と使用しない場合の結果を比較します。
1. ドロップアウトが適用された場合と適用されない場合の各隠れレイヤーでのアクティベーションのばらつきはどれくらいですか？プロットを描画して、この量が両方のモデルで経時的にどのように変化するかを示します。
1. テスト時にドロップアウトが一般的に使用されないのはなぜですか？
1. このセクションのモデルを例として使用して、ドロップアウトとウェイトディケイを使用した場合の効果を比較します。ドロップアウトとウェイトディケイを同時に使用するとどうなりますか？結果は加法性ですか？リターンの減少（またはそれより悪い）はありますか？彼らはお互いをキャンセルしますか?
1. 活性化ではなく重みマトリックスの個々の重みにドロップアウトを適用するとどうなりますか？
1. 各層にランダムノイズを注入する、標準のドロップアウト手法とは異なる、もう 1 つの手法を考案します。Fashion-MNIST データセット (固定アーキテクチャの場合) のドロップアウトよりも優れた方法を開発できますか?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:

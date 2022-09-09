```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# ドロップアウト
:label:`sec_dropout`

良い予測モデルに期待されることについて簡単に考えてみましょう。私たちは、目に見えないデータに対してうまく機能することを望んでいます。古典的汎化理論は、列車と試験性能のギャップを埋めるために、単純なモデルを目指すべきであることを示唆しています。シンプルさは、少数の次元の形でもたらされます。:numref:`sec_generalization_basics`の線形モデルの単項基底関数を議論する際にこれを探りました。さらに、:numref:`sec_weight_decay`で重量の減衰（$\ell_2$正則化）について説明したときに見たように、パラメータの（逆）ノルムも単純さの有用な尺度を表しています。シンプルさのもう1つの有用な概念は滑らかさです。つまり、関数は入力の小さな変化に敏感であってはならないということです。たとえば、画像を分類する場合、ピクセルにランダムノイズを追加してもほとんど無害であると予想されます。 

1995年、クリストファー・ビショップは、入力ノイズによるトレーニングがティホノフの正則化:cite:`Bishop.1995`と同等であることを証明したときに、このアイデアを形式化しました。この作業は、関数が滑らかである（したがって単純である）という要件と、入力の摂動に対して回復力があるという要件との間に明確な数学的関連性を引き出しました。 

そして、2014年、Srivastavaら:cite:`Srivastava.Hinton.Krizhevsky.ea.2014`は、ビショップのアイデアをネットワークの内部レイヤーにも適用する方法について巧妙なアイデアを開発しました。彼らのアイデアは*ドロップアウト*と呼ばれ、フォワードプロパゲーション中に各内部レイヤーを計算しながらノイズを注入することを含み、ニューラルネットワークをトレーニングするための標準的な手法になりました。この方法は*dropout* と呼ばれています。なぜなら、私たちは文字通り
*トレーニング中にいくつかのニューロンを落とす*。
トレーニング中、各反復で、標準ドロップアウトは、後続のレイヤーを計算する前に、各レイヤーのノードの一部をゼロにすることで構成されます。 

明確にするために、私たちはビショップへのリンクで私たち自身の物語を押し付けています。ドロップアウトに関する元の論文は、有性生殖の驚くべき類推を通して直感を提供します。著者らは、ニューラルネットワークの過剰適合は、各層が前の層の特定の活性化パターンに依存している状態によって特徴付けられ、この状態を*共適応*と呼んでいると主張している。ドロップアウトは、有性生殖が共適応を解散すると主張されているのと同じように、共適応を壊すと主張している遺伝子。この理論の説明は確かに議論の余地がありますが、ドロップアウト技術自体は永続的であることが証明されており、さまざまな形式のドロップアウトがほとんどのディープラーニングライブラリに実装されています。  

重要な課題は、このノイズをいかに注入するかです。1つのアイデアは、ノイズを*偏りのない*方法で注入することです。これにより、各レイヤーの期待値は、他のレイヤーを固定しながら、ノイズがない場合と同じになります。ビショップの研究では、線形モデルへの入力にガウスノイズを追加しました。各トレーニング反復で、彼は平均ゼロの分布からサンプリングされたノイズを入力 $\mathbf{x}$ に追加し、摂動点 $\mathbf{x}' = \mathbf{x} + \epsilon$ を生成します。予想通り、$E[\mathbf{x}'] = \mathbf{x}$。 

標準のドロップアウト正則化では、各レイヤーのノードの一部をゼロにし、保持された（ドロップアウトされていない）ノードの割合で正規化することにより、各レイヤーを*debiases* します。つまり、*ドロップアウト確率* $p$ では、各中間アクティベーション $h$ は次のように確率変数 $h'$ に置き換えられます。 

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

設計上、期待値は変わりません、つまり $E[h'] = h$。 

## ドロップアウト・イン・プラクティス

:numref:`fig_mlp`の隠れ層と5つの隠れユニットを持つMLPを思い出してください。隠れ層にドロップアウトを適用し、各隠れユニットを確率$p$でゼロにすると、結果は元のニューロンのサブセットのみを含むネットワークとして見ることができます。:numref:`fig_dropout2` では、$h_2$ と $h_5$ が削除されました。その結果、出力の計算は $h_2$ または $h_5$ に依存しなくなり、バックプロパゲーションの実行時にそれぞれの勾配も消失します。この方法では、出力層の計算が $h_1, \ldots, h_5$ のいずれかの要素に過度に依存することはありません。 

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

通常、テスト時にドロップアウトを無効にします。訓練されたモデルと新しい例を考えれば、ノードをドロップアウトしないため、正規化する必要はありません。ただし、いくつかの例外があります。一部の研究者は、ニューラルネットワーク予測の*不確実性*を推定するためのヒューリスティックとして、テスト時にドロップアウトを使用します。予測が多くの異なるドロップアウトマスク間で一致する場合、ネットワークの信頼性が高いと言えます。 

## ゼロからの実装

単一レイヤーにドロップアウト関数を実装するには、レイヤーの次元数と同じ数のベルヌーイ (バイナリ) 確率変数からサンプルを描画する必要があります。ここで、確率変数は値 $1$ (keep) と確率 $1-p$、$0$ (drop) と確率 $p$。これを実装する簡単な方法の 1 つは、まず一様分布 $U[0, 1]$ からサンプルを抽出することです。次に、対応するサンプルが$p$より大きいノードを保持し、残りを削除できます。 

次のコードでは、(**テンソル入力 `X` の要素を確率 `dropout` で削除する `dropout_layer` 関数を実装する**)、上記のように余りを再スケーリングします:生存者を `1.0-dropout` で割ります。

```{.python .input  n=5}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

[**いくつかの例で`dropout_layer`関数をテストする**] ことができます。次のコード行では、入力`X`をドロップアウト操作に渡します。確率はそれぞれ0、0.5、1です。

```{.python .input  n=6}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### モデルを定義する

以下のモデルは、（活性化関数に従って）各隠れ層の出力にドロップアウトを適用します。脱落確率はレイヤーごとに個別に設定できます。一般的な傾向は、入力レイヤーの近くでドロップアウトの確率を低く設定することです。ドロップアウトはトレーニング中のみアクティブになるようにしています。

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)
        
    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**トレーニング**]

以下は、前に説明した MLP のトレーニングと似ています。

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256, 
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**簡潔な実装**]

高レベルAPIでは、全結合層の後に`Dropout`層を追加し、ドロップアウト確率をコンストラクタの唯一の引数として渡すだけで済みます。トレーニング中、`Dropout` 層は、指定されたドロップアウト確率に従って、前の層の出力 (または同等に後続のレイヤーへの入力) をランダムにドロップアウトします。トレーニングモードではない場合、`Dropout` 層はテスト中にデータを渡すだけです。

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

次に、[**モデルをトレーニングする**]。

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## まとめ

* ドロップアウトは、次元の数と重みベクトルのサイズを制御するだけでなく、過剰適合を避けるためのもう1つのツールです。多くの場合、それらは共同で使用されます。
* ドロップアウトは、アクティベーション$h$を、期待値$h$のランダム変数に置き換えます。
* ドロップアウトはトレーニング中にのみ使用されます。

## 演習

1. 第1層と第2層の脱落確率を変更するとどうなりますか？特に、両方のレイヤーを切り替えるとどうなりますか？これらの質問に答える実験を計画し、結果を定量的に説明し、定性的な要点を要約します。
1. エポック数を増やし、dropoutを使用した場合と使用しない場合の結果を比較します。
1. ドロップアウトが適用されている場合と適用されていない場合の各非表示レイヤーのアクティベーションの差異はどれくらいですか？両方のモデルについて、この量が時間とともにどのように変化するかを示すプロットを描画します。
1. ドロップアウトは通常、テスト時に使用されないのはなぜですか？
1. このセクションのモデルを例として使用して、ドロップアウトと重量減衰を使用した場合の効果を比較します。ドロップアウトとウェイトディケイを同時に使用するとどうなりますか？結果は加算的ですか？リターンの減少（またはもっと悪い）はありますか？彼らはお互いをキャンセルしますか?
1. 活性化ではなくウェイトマトリックスの個々のウェイトにドロップアウトを適用するとどうなりますか？
1. 標準的なドロップアウト手法とは異なる、各層にランダムノイズを注入する別の手法を考案する。Fashion-mnist データセット (固定アーキテクチャー用) でドロップアウトよりも優れた方法を開発できますか?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:

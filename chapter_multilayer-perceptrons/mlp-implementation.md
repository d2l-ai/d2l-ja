```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 多層パーセプトロンの実装
:label:`sec_mlp-implementation`

多層パーセプトロン (MLP) は、単純な線形モデルほど実装が複雑ではありません。概念上の重要な違いは、複数のレイヤーを連結するようになったことです。

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## ゼロからの実装

このようなネットワークをゼロから実装することから始めましょう。 

### モデルパラメーターの初期化

Fashion-mnist には 10 個のクラスが含まれており、各イメージはグレースケールピクセル値の $28 \times 28 = 784$ グリッドで構成されていることを思い出してください。前と同じように、ここではピクセル間の空間構造を無視するので、これは 784 個の入力フィーチャと 10 個のクラスを持つ分類データセットと考えることができます。はじめに、[**1つの隠れ層と256の隠れユニットを持つMLPを実装する。**] 層の数と幅はどちらも調整可能（ハイパーパラメータとみなされる）。通常、層の幅は 2 の累乗で割り切れるように選択します。これは、メモリがハードウェアで割り当てられ、アドレス指定される方法により、計算効率が向上します。 

ここでも、パラメータをいくつかのテンソルで表現します。*すべてのレイヤー*について、1つの重み行列と1つのバイアスベクトルを追跡しなければならないことに注意してください。いつものように、これらのパラメータに関して損失の勾配にメモリを割り当てます。

```{.python .input  n=5}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input  n=6}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input  n=7}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

### モデル

すべてがどのように機能するかを確実に知るために、組み込みの`relu`関数を直接呼び出すのではなく、[**ReLUアクティベーション**を実装する**] します。

```{.python .input  n=8}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input  n=9}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input  n=10}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

空間構造を無視しているので、各2次元画像を長さ`num_inputs`のフラットベクトルに`reshape`します。最後に、ほんの数行のコードで (**モデルを実装**) します。私たちはフレームワークの組み込みオートグラードを使っているので、これだけで十分です。

```{.python .input  n=11}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### トレーニング

幸い、[**MLPの学習ループはソフトマックス回帰とまったく同じです。**] モデル、データ、トレーナーを定義し、最後にモデルとデータに対して関数 `fit` を呼び出します。

```{.python .input  n=12}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 簡潔な実装

ご想像のとおり、高レベル API に頼ることで、MLP をさらに簡潔に実装できます。 

### モデル

ソフトマックス回帰実装の簡潔な実装 (:numref:`sec_softmax_concise`) と比べると、唯一の違いは
*以前に*1つ*だけ追加した2つの*完全接続レイヤー。
1つ目は [**非表示レイヤー**] で、2つ目は出力レイヤーです。

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

### トレーニング

[**トレーニングループ**] は、ソフトマックス回帰を実装したときとまったく同じです。このモジュール性により、モデルアーキテクチャに関する事項を直交的な考慮事項から分離することができます。

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## まとめ

ディープネットワークを設計する実践が増えた今、ディープネットワークの単一レイヤーから複数レイヤーへのステップは、もはやそれほど大きな課題にはなりません。特に、トレーニングアルゴリズムとデータローダーを再利用できます。ただし、MLP をゼロから実装するのは面倒です。モデルパラメータの名前付けと追跡を行うと、モデルの拡張が困難になります。たとえば、レイヤー 42 と 43 の間に別のレイヤーを挿入するとします。これは、順番に名前を変更する意思がない限り、レイヤー42bになる可能性があります。さらに、ネットワークをゼロから実装すると、フレームワークが有意義なパフォーマンスの最適化を実行することははるかに困難になります。 

それでも、完全に接続されたディープネットワークがニューラルネットワークモデリングの選択方法であった1980年代後半の最先端に到達しました。次の概念的なステップは、画像を考えることです。その前に、いくつかの統計の基礎と、モデルを効率的に計算する方法の詳細を確認する必要があります。 

## 演習

1. 非表示ユニットの数 `num_hiddens` を変更し、その数がモデルの精度にどのように影響するかをプロットします。このハイパーパラメータの最大の価値は何ですか？
1. 非表示のレイヤーを追加して、結果にどのような影響があるかを確認してください。
1. 単一のニューロンで隠れ層を挿入するのはなぜ悪い考えですか？何が悪くなる可能性がありますか？
1. 学習率を変えると結果はどう変わりますか？他のすべてのパラメータを固定した状態で、どの学習率が最も良い結果が得られますか？これはエポック数とどのように関係していますか？
1. 学習率、エポック数、隠れ層の数、層ごとの隠れユニットの数など、すべてのハイパーパラメータを合わせて最適化しましょう。
    1. それらすべてを最適化することで得られる最高の結果は何ですか？
    1. 複数のハイパーパラメータを扱うのがはるかに難しいのはなぜですか？
    1. 複数のパラメータを共同で最適化する効率的な戦略を説明する。
1. 困難な問題について、フレームワークの速度とゼロからの実装を比較します。ネットワークの複雑さによってどのように変化しますか？
1. 整列した行列と整列していない行列のテンソル行列の乗算の速度を測定します。たとえば、次元 1024、1025、1026、1028、および 1032 の行列をテストします。
    1. これはGPUとCPUの間でどのように変化しますか？
    1. CPU と GPU のメモリバス幅を決定します。
1. さまざまなアクティベーション機能を試してみてください。どれが一番いいの？
1. ネットワークの重み付け初期化に違いはありますか?それは問題なの？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:

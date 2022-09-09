```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 線形回帰の簡潔な実装
:label:`sec_linear_concise`

ディープラーニングは、過去10年間にカンブリア紀の爆発的な爆発を目の当たりにしてきました。膨大な数の技術、アプリケーション、アルゴリズムは、過去数十年の進歩をはるかに上回っています。これは、複数の要因が偶然に組み合わされているためです。そのうちの1つは、多数のオープンソースのディープラーニングフレームワークによって提供される強力な無料ツールです。Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`、DistBelief :cite:`Dean.Corrado.Monga.ea.2012`、およびCaffe :cite:`Jia.Shelhamer.Donahue.ea.2014`は、間違いなく広く採用された第1世代のモデルを代表しています。Lispのようなプログラミング体験を提供するSN2（Simulateur Neuristique）:cite:`Bottou.Le-Cun.1988`のような以前の（独創的な）作品とは対照的に、最新のフレームワークはPythonの自動差別化と利便性を提供します。これらのフレームワークにより、勾配ベースの学習アルゴリズムを実装する反復作業を自動化およびモジュール化できます。 

:numref:`sec_linear_scratch`では、（i）データストレージと線形代数のテンソル、および（ii）勾配の計算には自動微分のみに依存していました。実際には、データイテレータ、損失関数、オプティマイザ、ニューラルネットワーク層は非常に一般的であるため、現代のライブラリもこれらのコンポーネントを実装しています。このセクションでは、ディープラーニングフレームワークの:numref:`sec_linear_scratch`（**高レベルAPIを使用して簡潔に**）から（**線形回帰モデルの実装方法を説明します**）。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=1}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input  n=1}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

## モデルを定義する

:numref:`sec_linear_scratch` でゼロから線形回帰を実装したとき、モデルパラメーターを明示的に定義し、基本的な線形代数演算を使用して出力を生成するように計算をコード化しました。あなたはこれを行う方法を知っているべきです*。しかし、モデルがより複雑になり、ほぼ毎日これを行う必要がある場合は、喜んで支援を受けるでしょう。状況は、自分のブログをゼロからコーディングするのと似ています。それを1回か2回行うことはやりがいがあり、有益ですが、一ヶ月かけて車輪の再発明をすれば、お粗末なWeb開発者になるでしょう。 

標準的な操作では、[**フレームワークの事前定義されたレイヤーを使用**] できます。これにより、実装について心配することなく、モデルの構築に使用されるレイヤーに集中できます。:numref:`fig_single_neuron`で説明されている単層ネットワークのアーキテクチャを思い出してください。この層は、各入力が行列ベクトル乗算によって各出力に接続されるため、*完全接続* と呼ばれます。

:begin_tab:`mxnet`
Gluonでは、全結合層は`Dense`クラスで定義されています。単一のスカラー出力のみを生成したいので、その数を 1 に設定します。便宜上、Gluonでは各レイヤーの入力形状を指定する必要がないことは注目に値します。したがって、この線形層に入る入力の数をGluonに伝える必要はありません。モデルに初めてデータを渡すとき、例えば`net(X)`を後で実行すると、Gluonは各レイヤーへの入力数を自動的に推測し、正しいモデルをインスタンス化します。これがどのように機能するかについては、後で詳しく説明します。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、全結合層は `Linear` と `LazyLinear` (バージョン 1.8.0 以降で利用可能) のクラスで定義されています。後者では、ユーザーは出力次元を*のみ*指定できますが、前者はさらにこの層に入る入力の数を要求します。入力シェイプの指定は不便で、(畳み込み層などで) 自明でない計算が必要になる場合があります。したがって、簡単にするために、できる限りこのような「怠惰な」レイヤーを使用します。
:end_tab:

:begin_tab:`tensorflow`
Kerasでは、全結合層は`Dense`クラスで定義されています。単一のスカラー出力のみを生成したいので、その数を 1 に設定します。便宜上、Kerasでは各レイヤーの入力形状を指定する必要がないことは注目に値します。この線形層に入る入力の数をKerasに伝える必要はありません。最初にモデルにデータを渡そうとするとき、例えば`net(X)`を後で実行すると、Kerasは各レイヤーへの入力数を自動的に推測します。これがどのように機能するかについては、後で詳しく説明します。
:end_tab:

```{.python .input}
%%tab all
class LinearRegression(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

`forward` メソッドでは、定義済みレイヤーの組み込み関数 `__call__` を呼び出して出力を計算します。

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    """The linear regression model."""
    return self.net(X)
```

## 損失関数の定義

:begin_tab:`mxnet`
`loss`モジュールは、多くの有用な損失関数を定義しています。スピードと利便性のために、私たちは独自の実装を忘れて、代わりに組み込みの`loss.L2Loss`を選択します。返される`loss`は各例の二乗誤差であるため、`mean`を使用してミニバッチ全体の損失を平均します。
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` クラスは平均二乗誤差 (:eqref:`eq_mse` の $1/2$ 係数を含まない) を計算します。**] 既定では、`MSELoss` は例に対する平均損失を返します。独自に実装するよりも速い (そして使いやすい)。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` クラスは、平均二乗誤差 (:eqref:`eq_mse` の $1/2$ 係数を含まない) を計算します。デフォルトでは、例の平均損失を返します。
:end_tab:

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

## 最適化アルゴリズムの定義

:begin_tab:`mxnet`
Minibatch SGD はニューラルネットワークを最適化するための標準ツールであるため、Gluon は `Trainer` クラスを通じて、このアルゴリズムの多くのバリエーションと共にそれをサポートしています。Gluonの`Trainer`クラスは最適化アルゴリズムを表し、:numref:`sec_oo-design`で作成した`Trainer`クラスにはトレーニング関数が含まれています。つまり、オプティマイザを繰り返し呼び出してモデルパラメータを更新します。`Trainer`をインスタンス化するとき、`net.collect_params()`を介してモデル`net`から取得できる、最適化するパラメータ、使用する最適化アルゴリズム（`sgd`）、および最適化アルゴリズムに必要なハイパーパラメータの辞書を指定します。
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD はニューラルネットワークを最適化するための標準ツールであるため、PyTorch は `optim` モジュールでこのアルゴリズムの多くのバリエーションと共にこれをサポートします。（**`SGD`インスタンスをインスタンス化**）するとき、`self.parameters()`を介してモデルから取得できる、最適化するパラメーターと、最適化アルゴリズムに必要な学習率（`self.lr`）を指定します。
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGDはニューラルネットワークを最適化するための標準ツールであるため、Kerasは`optimizers`モジュールでこのアルゴリズムの多くのバリエーションと共にそれをサポートしています。
:end_tab:

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
```

## トレーニング

ディープラーニングフレームワークの高レベル API を使用してモデルを表現するには、必要なコード行が少なくて済むことに気づいたかもしれません。パラメーターを個別に割り当てたり、損失関数を定義したり、ミニバッチ SGD を実装したりする必要はありませんでした。もっと複雑なモデルで作業を始めると、高レベル API の利点はかなり大きくなるでしょう。これで基本的な要素がすべて揃ったので、[**トレーニングループ自体は、ゼロから実装したものと同じです。**] そこで、:numref:`sec_linear_scratch` の `fit_epoch` メソッドの実装に依存する `fit` メソッド (:numref:`oo-design-training` で導入) を呼び出して、モデルをトレーニングします。

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

以下では、データセットを生成した [**有限データのトレーニングによって学習したモデルパラメータと実際のパラメータを比較する**]。パラメータにアクセスするには、必要な層の重みと偏りにアクセスします。ゼロからの実装と同様に、推定されたパラメータは真の対応パラメータに近いことに注意してください。

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## まとめ

このセクションには、Gluon `Chen.Li.Li.ea.2015`、JAX :cite:`Frostig.Johnson.Leary.2018`、PyTorch :cite:`Paszke.Gross.Massa.ea.2019`、Tensorflow :cite:`Abadi.Barham.Chen.ea.2016` などの最新のディープラーニングフレームワークによって提供される便利さを活用するためのディープネットワーク（本書内）の最初の実装が含まれています。データのロード、レイヤー、損失関数、オプティマイザー、およびトレーニングループの定義にフレームワークのデフォルトを使用しました。フレームワークが必要な機能をすべて提供する場合は常に、それらを使用することをお勧めします。これらのコンポーネントのライブラリ実装は、パフォーマンスが大幅に最適化され、信頼性が適切にテストされる傾向があるためです。同時に、これらのモジュールは直接実装可能であることを忘れないようにしてください。これは、現在のどのライブラリにも存在し得ない新しいコンポーネントを発明するモデル開発の最先端を生きたいと願う意欲的な研究者にとって特に重要です。

:begin_tab:`mxnet`
Gluonでは、`data`モジュールはデータ処理のためのツールを提供し、`nn`モジュールは多数のニューラルネットワーク層を定義し、`loss`モジュールは多くの一般的な損失関数を定義します。さらに、`initializer`は、パラメータ初期化のための多くの選択肢へのアクセスを提供します。ユーザーにとって便利なことに、次元とストレージは自動的に推測されます。この遅延初期化の結果、パラメータがインスタンス化 (および初期化) される前にパラメータにアクセスしようとしないでください。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、`data`モジュールはデータ処理のためのツールを提供し、`nn`モジュールは多数のニューラルネットワーク層と一般的な損失関数を定義します。パラメータの値を `_` で終わるメソッドに置き換えることで、パラメータを初期化できます。ネットワークの入力次元を指定する必要があることに注意してください。これは今のところ些細なことですが、多くの層を持つ複雑なネットワークを設計する場合、大きな効果をもたらす可能性があります。移植性を確保するには、これらのネットワークをどのようにパラメータ化するかについて慎重に検討する必要があります。
:end_tab:

:begin_tab:`tensorflow`
TensorFlowでは、`data`モジュールはデータ処理のためのツールを提供し、`keras`モジュールは多数のニューラルネットワーク層と一般的な損失関数を定義します。さらに、`initializers` モジュールは、モデルパラメーターを初期化するためのさまざまな方法を提供します。ネットワークの次元とストレージは自動的に推測されます (ただし、初期化される前にパラメーターにアクセスしようとしないように注意してください)。
:end_tab:

## 演習

1. ミニバッチの総損失をミニバッチの損失の平均に置き換える場合、学習率をどのように変更する必要がありますか？
1. フレームワークのドキュメントを確認して、どの損失関数が提供されているかを確認します。特に、二乗損失をHuberのロバストな損失関数に置き換えます。つまり、損失関数$$l (y, y') =\ begin {case} |y-y'|-\ frac {\ sigma} {2} &\ text {if} |y-y'| >\ sigma\\ frac {1} {2\ sigma} (y-y') ^2 &\ text {そうでなければ}\ end {case} $$
1. モデルの重みの勾配にはどのようにアクセスしますか？
1. 学習率とエポック数を変えると、解はどのように変化しますか？改善し続けていますか？
1. 生成されるデータ量を変更すると、ソリューションはどのように変化しますか?
    1. $\hat{\mathbf{w}} - \mathbf{w}$ と $\hat{b} - b$ の推定誤差をデータ量の関数としてプロットします。ヒント：データ量を直線的にではなく対数的に増やします。つまり、5、10、20、50、...、1,000、2,000、...、10,000ではなく10,000です。
    2. ヒントの提案が適切なのはなぜですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

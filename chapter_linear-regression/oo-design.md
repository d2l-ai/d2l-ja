```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 実装のためのオブジェクト指向設計
:label:`sec_oo-design`

線形回帰の概要では、データ、モデル、損失関数、最適化アルゴリズムなど、さまざまなコンポーネントについて説明しました。実際、線形回帰は最も単純な機械学習モデルの1つです。しかし、それをトレーニングするには、この本の他のモデルが必要とするものと同じコンポーネントの多くを使用します。したがって、実装の詳細を掘り下げる前に、本書全体で使用されているいくつかの API を設計する価値があります。ディープラーニングのコンポーネントをオブジェクトとして扱う場合、これらのオブジェクトとその相互作用のクラスを定義することから始めることができます。このオブジェクト指向の実装設計により、プレゼンテーションが大幅に合理化され、プロジェクトで使用することもできます。 

[PyTorch Lightning](https://www.pytorchlightning.ai/)などのオープンソースライブラリに触発され、高レベルでは、3つのクラスを用意したいと考えています。（i）`Module`にはモデル、損失、および最適化メソッドが含まれています。（ii）`DataModule`はトレーニングと検証のためのデータローダーを提供します。（iii）両方のクラスは`Trainer`クラスを使用して結合され、トレーニングが可能になりますさまざまなハードウェアプラットフォーム上のモデル。この本のほとんどのコードは、`Module`と`DataModule`に適合しています。`Trainer` クラスについて触れるのは、GPU、CPU、並列トレーニング、および最適化アルゴリズムについて説明するときだけです。

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import torch as d2l
import tensorflow as tf
```

## ユーティリティ
:label:`oo-design-utilities`

Jupyter ノートブックのオブジェクト指向プログラミングを簡略化するには、いくつかのユーティリティが必要です。課題の 1 つは、クラス定義がかなり長いコードブロックになる傾向があることです。ノートブックの可読性には、説明が散在する短いコード断片が必要です。これは、Pythonライブラリに共通のプログラミングスタイルと両立しない要件です。最初のユーティリティ関数では、クラスが作成された *後* に、関数をメソッドとしてクラスに登録することができます。実際、クラスのインスタンスを作成した後でも、そうすることができます。これにより、クラスの実装を複数のコードブロックに分割できます。

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

それでは、使い方を簡単に見てみましょう。クラス`A`をメソッド`do`で実装する予定です。同じコードブロックに `A` と `do` の両方のコードを含める代わりに、まずクラス `A` を宣言し、インスタンス `a` を作成します。

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

次に、通常どおりにメソッド `do` を定義しますが、クラス `A` のスコープでは定義しません。代わりに、引数としてクラス `A` を使用して `add_to_class` によってこのメソッドを修飾します。そうすることで、このメソッドは `A` の定義の一部として定義されていた場合に予想されるように、`A` のメンバー変数にアクセスできます。インスタンス `a` に対して呼び出すとどうなるか見てみましょう。

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

2 つ目は、クラスの `__init__` メソッドのすべての引数をクラス属性として保存するユーティリティクラスです。これにより、追加のコードなしでコンストラクタ呼び出しシグネチャを暗黙的に拡張できます。

```{.python .input}
%%tab all
class HyperParameters:  #@save
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

その実装は:numref:`sec_utils`に延期されます。これを使用するには、`HyperParameters` を継承し、`__init__` メソッドで `save_hyperparameters` を呼び出すクラスを定義します。

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

最後のユーティリティは、実験の進行中にインタラクティブに実験の進行状況をプロットすることができます。はるかに強力な（そして複雑な）[TensorBoard](https://www.tensorflow.org/tensorboard)に敬意を表して、`ProgressBoard`と名付けました。実装は:numref:`sec_utils`に延期されます。とりあえず、動作を簡単に見てみましょう。 

関数 `draw` は、凡例で `label` を指定して、図の点 `(x, y)` をプロットします。オプションの`every_n`は、図に$1/n$点のみを表示することでラインを滑らかにします。これらの値は、元の図の $n$ の近傍点から平均化されています。

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """Plot data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

次の例では、`sin`と`cos`を異なる滑らかさで描画します。このコードブロックを実行すると、アニメーションで線が大きくなるのがわかります。

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## モデル
:label:`oo-design-models`

`Module` クラスは、実装するすべてのモデルの基本クラスです。少なくとも 3 つの方法を定義する必要があります。`__init__` メソッドは学習可能なパラメーターを格納し、`training_step` メソッドはデータバッチを受け入れて損失値を返し、`configure_optimizers` メソッドは学習可能なパラメーターの更新に使用される最適化メソッドまたはそのリストを返します。オプションで、評価尺度を報告する `validation_step` を定義できます。再利用性を高めるために、出力を計算するコードを別の`forward`メソッドに入れることがあります。

```{.python .input}
%%tab all
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('pytorch'):
            self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
`Module`は、Gluonのニューラルネットワークの基本クラスである`nn.Block`のサブクラスであることに気付くかもしれません。ニューラルネットワークを処理する便利な機能を提供します。たとえば、`forward(self, X)`などの`forward`メソッドを定義すると、インスタンス`a`に対して`a(X)`によってこの関数を呼び出すことができます。これは、組み込みの`__call__`メソッドで`forward`メソッドを呼び出すため機能します。`nn.Block` の詳細と例については、:numref:`sec_model_construction` を参照してください。
:end_tab:

:begin_tab:`pytorch`
`Module`は、PyTorchのニューラルネットワークの基本クラスである`nn.Module`のサブクラスであることに気付くかもしれません。ニューラルネットワークを処理する便利な機能を提供します。たとえば、`forward(self, X)`などの`forward`メソッドを定義すると、インスタンス`a`に対して`a(X)`によってこの関数を呼び出すことができます。これは、組み込みの`__call__`メソッドで`forward`メソッドを呼び出すため機能します。`nn.Module` の詳細と例については、:numref:`sec_model_construction` を参照してください。
:end_tab:

:begin_tab:`tensorflow`
`Module`は、TensorFlowのニューラルネットワークの基本クラスである`tf.keras.Model`のサブクラスであることに気付くかもしれません。ニューラルネットワークを処理する便利な機能を提供します。たとえば、組み込みの `__call__` メソッドの `call` メソッドを呼び出します。ここでは、`call` を `forward` 関数にリダイレクトし、引数をクラス属性として保存します。これは、コードを他のフレームワーク実装とより類似させるために行います。
:end_tab:

##  データ
:label:`oo-design-data`

`DataModule` クラスは、データの基本クラスです。データの準備には `__init__` メソッドがよく使用されます。これには、必要に応じてダウンロードと前処理が含まれます。`train_dataloader` は、トレーニングデータセットのデータローダーを返します。データローダーは、使用されるたびにデータバッチを生成する (Python) ジェネレーターです。このバッチは、`Module` の `training_step` メソッドに入力され、損失が計算されます。検証データセットローダーを返すオプションの `val_dataloader` があります。これは、`Module` の `validation_step` メソッドのデータバッチを生成することを除いて、同じように動作します。

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## トレーニング
:label:`oo-design-training`

`Trainer` クラスは、`DataModule` で指定されたデータを使用して `Module` クラスの学習可能なパラメーターを学習させます。キーメソッドは`fit`で、2つの引数を受け取ります。`model`は`Module`のインスタンスであり、`DataModule`のインスタンスである`data`です。次に、データセット全体を `max_epochs` 回反復してモデルをトレーニングします。前と同じように、この関数の実装は後の章に任せます。

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## まとめ

将来のディープラーニング実装のためのオブジェクト指向設計を強調するために、上記のクラスは、オブジェクトがどのようにデータを格納し、相互に作用するかを示すだけです。本の残りの部分では、`@add_to_class `を介するなどして、これらのクラスの実装を充実させ続ける。さらに、これらの完全に実装されたクラスは、ディープラーニングのための構造化モデリングを容易にする*軽量ツールキット*である[d2l library](https://github.com/d2l-ai/d2l-en/tree/master/d2l)に保存されています。特に、あまり変更することなく、プロジェクト間で多くのコンポーネントを再利用することが容易になります。たとえば、オプティマイザだけ、モデルだけ、データセットだけを置き換えることができます。この程度のモジュール性は、簡潔さと単純さ（これが私たちがそれを追加した理由です）の点で本全体に配当をもたらし、あなた自身のプロジェクトでも同じことをすることができます。  

## 演習

1. [d2l library](https://github.com/d2l-ai/d2l-en/tree/master/d2l) に保存されている上記のクラスの完全な実装を見つけます。ディープラーニングモデリングに慣れてきたら、実装の詳細を確認することを強くお勧めします。
1. `B` クラスの `save_hyperparameters` ステートメントを削除します。`self.a` と `self.b` をまだ印刷できますか？オプション:`HyperParameters` クラスの完全な実装に没頭したことがあるなら、その理由を説明できますか?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6647)
:end_tab:

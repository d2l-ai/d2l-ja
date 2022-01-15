# 画層とブロック
:label:`sec_model_construction`

ニューラルネットワークを初めて導入したときは、単一出力の線形モデルに注目しました。ここでは、モデル全体が単一のニューロンだけで構成されています。1 つのニューロン (i) が何らかの入力を受け取り、(ii) 対応するスカラー出力を生成し、(iii) 関心のある目的関数を最適化するために更新可能な関連パラメーターのセットがあることに注意してください。その後、複数の出力を持つネットワークについて考え始めると、ベクトル化された演算を利用してニューロンの層全体の特性を評価しました。個々のニューロンと同様に、層 (i) は一連の入力を受け取り、(ii) 対応する出力を生成し、(iii) 一連の調整可能なパラメーターによって記述されます。ソフトマックス回帰を行ったとき、単層自体がモデルでした。しかし、その後にMLPを導入したときも、このモデルはこれと同じ基本構造を保持していると考えることができた。 

興味深いことに、MLP では、モデル全体とその構成層の両方がこの構造を共有しています。モデル全体が生の入力 (特徴) を取り込み、出力 (予測) を生成し、パラメーター (すべての構成層からの結合パラメーター) を持ちます。同様に、個々の層は (前の層から供給された) 入力を取り込み、出力 (後続層への入力) を生成し、後続の層から逆方向に流れる信号に従って更新される一連の調整可能なパラメーターを持ちます。 

ニューロン、層、モデルが私たちのビジネスを進めるのに十分な抽象化をもたらすと考えるかもしれませんが、個々のレイヤーよりも大きいがモデル全体よりも小さいコンポーネントについて話すと便利なことがよくあります。たとえば、コンピュータビジョンで非常に普及しているResNet-152アーキテクチャは、数百のレイヤーを所有しています。これらのレイヤーは、*レイヤーのグループ* の繰り返しパターンで構成されます。このようなネットワークを一度に 1 つのレイヤで実装するのは面倒な作業になることがあります。この懸念は単なる仮説的なものではなく、実際にはこのようなデザインパターンが一般的です。上記の ResNet アーキテクチャは、認識と検出の両方で 2015 年の ImageNet と COCO のコンピュータビジョンコンペティションで優勝し、多くのビジョンタスクで今でも頼りになるアーキテクチャです。レイヤーがさまざまな繰り返しパターンで配置される同様のアーキテクチャは、自然言語処理や音声処理などの他の領域でも広く普及しています。 

これらの複雑なネットワークを実装するために、ニューラルネットワーク「ブロック」という概念を導入します。ブロックは、1 つのレイヤー、複数のレイヤーで構成されるコンポーネント、またはモデル全体を記述できます。ブロック抽象化を使用する利点の 1 つは、それらを結合してより大きなアーティファクトに (多くの場合、再帰的に) できることです。これは :numref:`fig_blocks` で説明されています。任意の複雑さのブロックをオンデマンドで生成するコードを定義することで、驚くほどコンパクトなコードを作成しながら、複雑なニューラルネットワークを実装できます。 

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

プログラミングの観点からは、ブロックは*class* で表されます。そのサブクラスは、入力を出力に変換し、必要なパラメーターを格納する前方伝播関数を定義する必要があります。ブロックによってはパラメーターをまったく必要としないものがあることに注意してください。最後に、勾配を計算するために、ブロックは逆伝播関数を持たなければなりません。幸いなことに、独自のブロックを定義するときに自動微分 (:numref:`sec_autograd` で導入) によってもたらされるいくつかの舞台裏の魔法により、パラメーターと前方伝播関数について心配するだけで済みます。 

[**はじめに、MLP の実装に使用したコードを再検討します**](:numref:`sec_mlp_concise`)。次のコードは、256 ユニットと ReLU アクティベーションを持つ 1 つの完全接続された隠れ層をもつネットワークを生成し、その後に 10 ユニットの完全接続された出力層 (アクティベーション関数なし) をもつネットワークを生成します。

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
この例では、`nn.Sequential` をインスタンス化し、返されたオブジェクトを `net` 変数に代入してモデルを構築しました。次に、`add` 関数を繰り返し呼び出し、実行すべき順序でレイヤーを追加します。つまり、`nn.Sequential` は特別な種類の `Block` を定義しています。このクラスは、Gluon でブロックを表すクラスです。構成要素 `Block` の順序付きリストを維持します。`add` 関数は、連続する各 `Block` をリストに追加しやすくします。各層は `Dense` クラスのインスタンスであり、それ自体が `Block` のサブクラスであることに注意してください。順伝播 (`forward`) 関数も非常に単純です。リスト内の各 `Block` を連結し、それぞれの出力を入力として次の関数に渡します。ここまでは、`net(X)` コンストラクションを介してモデルを呼び出して、その出力を取得してきました。これは実際には `Block` クラスの `__call__` 関数によって実現された Python の巧妙なトリックである `net.forward(X)` の省略表現です。
:end_tab:

:begin_tab:`pytorch`
この例では、`nn.Sequential` をインスタンス化してモデルを構築しました。レイヤーは実行順序どおりに引数として渡されます。つまり、(**`nn.Sequential` は特別な種類の `Module` を定義します**)、PyTorch でブロックを提示するクラスです。構成要素 `Module` の順序付きリストを維持します。2 つの完全接続層はそれぞれ `Linear` クラスのインスタンスであり、それ自体が `Module` のサブクラスであることに注意してください。順伝播 (`forward`) 関数も非常に単純です。リスト内の各ブロックを連結し、それぞれの出力を入力として次のブロックに渡します。ここまでは、コンストラクション `net(X)` を介してモデルを呼び出して、その出力を取得してきました。これは実際には `net.__call__(X)` の省略形です。
:end_tab:

:begin_tab:`tensorflow`
この例では、`keras.models.Sequential` をインスタンス化してモデルを構築しました。レイヤーは実行順序どおりに引数として渡されます。つまり、`Sequential` は、Keras でブロックを表すクラスである `keras.Model` という特別な種類を定義しています。構成要素 `Model` の順序付きリストを維持します。2 つの完全接続層はそれぞれ `Dense` クラスのインスタンスであり、それ自体が `Model` のサブクラスであることに注意してください。順伝播 (`call`) 関数も非常に単純です。リスト内の各ブロックを連結し、それぞれの出力を入力として次のブロックに渡します。ここまでは、コンストラクション `net(X)` を介してモデルを呼び出して出力を取得してきました。これは実際には、Block クラスの `__call__` 関数によって実現された Python の巧妙なトリックである `net.call(X)` の省略表現です。
:end_tab:

## [**カスタムブロック**]

ブロックがどのように機能するかを直感的に理解する最も簡単な方法は、ブロックを自分で実装することでしょう。独自のカスタムブロックを実装する前に、各ブロックが提供しなければならない基本機能を簡単にまとめます。

:begin_tab:`mxnet, tensorflow`
1. 入力データを前方伝播関数の引数として取り込みます。
1. 順伝播関数が値を返すようにして、出力を生成します。出力の形状が入力と異なる場合があることに注意してください。たとえば、上のモデルの最初の全結合層は、任意の次元の入力を取り込みますが、次元 256 の出力を返します。
1. 入力に対する出力の勾配を計算します。この勾配は、バックプロパゲーション関数を介してアクセスできます。通常、これは自動的に行われます。
1. 順伝播計算の実行に必要なパラメーターを保存し、そのパラメーターへのアクセスを提供します。
1. 必要に応じてモデルパラメーターを初期化します。
:end_tab:

:begin_tab:`pytorch`
1. 入力データを前方伝播関数の引数として取り込みます。
1. 順伝播関数が値を返すようにして、出力を生成します。出力の形状が入力と異なる場合があることに注意してください。たとえば、上のモデルの最初の全結合層は次元 20 の入力を取り込みますが、次元 256 の出力を返します。
1. 入力に対する出力の勾配を計算します。この勾配は、バックプロパゲーション関数を介してアクセスできます。通常、これは自動的に行われます。
1. 順伝播計算の実行に必要なパラメーターを保存し、そのパラメーターへのアクセスを提供します。
1. 必要に応じてモデルパラメーターを初期化します。
:end_tab:

次のスニペットでは、256 個の隠れ単位を持つ 1 つの隠れ層と 10 次元の出力層をもつ MLP に対応するブロックをゼロからコード化します。以下の `MLP` クラスは、ブロックを表すクラスを継承していることに注意してください。親クラスの関数に大きく依存し、独自のコンストラクタ (Python では `__init__` 関数) と前方伝播関数のみを提供します。

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Model` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

まず、順伝播関数に注目しましょう。`X` を入力として取り、アクティベーション関数を適用して隠れ表現を計算し、そのロジットを出力することに注意してください。この `MLP` の実装では、両方のレイヤーがインスタンス変数です。これが妥当な理由を理解するために、2 つの MLP `net1` と `net2` をインスタンス化し、異なるデータでそれらをトレーニングすることを想像してみてください。当然、それらは2つの異なる学習モデルを表すと予想されます。 

順伝播関数の呼び出しごとに、コンストラクターで [**MLP のレイヤーをインスタンス化**] します (**そしてこれらのレイヤーを呼び出します**)。いくつかの重要な詳細に注意してください。まず、カスタマイズした `__init__` 関数は `super().__init__()` を介して親クラスの `__init__` 関数を呼び出します。これにより、ほとんどのブロックに適用できるボイラープレートコードを再記述する手間が省けます。次に、2 つの完全に接続された Layer をインスタンス化し、`self.hidden` と `self.out` に割り当てます。new 演算子を実装しない限り、バックプロパゲーション関数やパラメーターの初期化について心配する必要はありません。これらの関数はシステムによって自動的に生成されます。これを試してみよう。

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

ブロック抽象化の主な利点は、その汎用性にあります。ブロックをサブクラス化して、層 (全結合層クラスなど)、モデル全体 (上記の `MLP` クラスなど)、または中程度の複雑度のさまざまなコンポーネントを作成できます。畳み込みニューラルネットワークを扱う場合など、この多様性を次の章で活用しています。 

## [**シーケンシャルブロック**]

ここで、`Sequential` クラスがどのように機能するのかを詳しく見てみましょう。`Sequential` は他のブロックをデイジーチェーン接続するように設計されていたことを思い出してください。単純化された `MySequential` を独自に構築するには、次の 2 つのキー関数を定義するだけです。
1. ブロックを 1 つずつリストに追加する関数。
2. 追加された順序と同じ順序で、ブロックのチェーンを介して入力を渡す順伝播関数。

次の `MySequential` クラスは、デフォルトの `Sequential` クラスと同じ機能を提供します。

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` 関数は、順序付きディクショナリ `_children` に 1 つのブロックを追加します。すべてのGluon `Block`が`_children`属性を持っている理由と、Pythonのリストを自分で定義するのではなく、なぜそれを使ったのか不思議に思うかもしれません。つまり `_children` の主な利点は、ブロックのパラメーターの初期化中に、Gluon は `_children` ディクショナリ内を調べて、パラメーターも初期化する必要のあるサブブロックを見つけることがわかっていることです。
:end_tab:

:begin_tab:`pytorch`
`__init__` メソッドでは、すべてのモジュールを順序付き辞書 `_modules` に 1 つずつ追加します。すべての `Module` がなぜ `_modules` 属性を持ち、Python リストを自分で定義するのではなくなぜそれを使ったのか不思議に思うかもしれません。つまり `_modules` の主な利点は、モジュールのパラメータ初期化中に、システムが `_modules` ディクショナリを調べて、パラメータも初期化する必要のあるサブモジュールを見つけることがわかっていることです。
:end_tab:

`MySequential` の前方伝播関数が呼び出されると、追加された各ブロックは追加された順に実行されます。これで、`MySequential` クラスを使用して MLP を再実装できます。

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

この `MySequential` の使い方は、`Sequential` クラス用に以前に書いたコード (:numref:`sec_mlp_concise` で説明) と同じであることに注意してください。 

## [**フォワード伝播関数でのコードの実行**]

`Sequential` クラスを使用するとモデルの構築が容易になり、独自のクラスを定義しなくても新しいアーキテクチャをアセンブルできます。ただし、すべてのアーキテクチャが単純なデイジーチェーンであるとは限りません。より柔軟性が必要な場合は、独自のブロックを定義する必要があります。たとえば、順伝播関数内で Python の制御フローを実行したい場合があります。さらに、定義済みのニューラルネットワーク層に頼るのではなく、任意の数学演算を実行したい場合もあります。 

これまで、私たちのネットワークのすべての操作が、ネットワークのアクティベーションとそのパラメーターに基づいて動作していたことに気付いたかもしれません。ただし、前のレイヤーの結果でも更新可能なパラメーターでもない用語を取り入れたい場合もあります。これらを*定数パラメータ*と呼びます。たとえば、関数 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ を計算するレイヤーが必要だとします。$\mathbf{x}$ は入力、$\mathbf{w}$ はパラメーター、$c$ は最適化中に更新されない指定された定数です。そこで `FixedHiddenMLP` クラスを以下のように実装します。

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

この `FixedHiddenMLP` モデルでは、重み (`self.rand_weight`) がインスタンス化時にランダムに初期化され、その後は一定になる隠れ層を実装します。この重みはモデルパラメータではないため、バックプロパゲーションによって更新されることはありません。その後、ネットワークはこの「固定」層の出力を全結合層に渡します。 

出力を返す前に、モデルが異常なことをしたことに注意してください。while ループを実行し、$L_1$ ノルムが $1$ より大きいという条件をテストし、条件を満たすまで出力ベクトルを $2$ で割ります。最後に、`X` のエントリの合計を返しました。われわれの知る限り、標準的なニューラルネットワークはこの操作を実行しません。この特定の操作は、実際のタスクでは役に立たない場合があることに注意してください。ここでのポイントは、ニューラルネットワーク計算の流れに任意のコードを統合する方法を示すことだけです。

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

[**ブロックをさまざまな方法で組み合わせて組み合わせる**] 次の例では、いくつかの創造的な方法でブロックをネストしています。

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## 効率性

:begin_tab:`mxnet`
熱心な読者は、これらの操作の一部の効率を心配し始めるかもしれません。結局のところ、高性能のディープラーニングライブラリと思われるものでは、辞書のルックアップ、コードの実行、その他多くのPythonicの処理が行われています。Python の [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) の問題はよく知られています。ディープラーニングのコンテキストでは、非常に高速な GPU が、別のジョブを実行する前に、ちっぽけな CPU が Python コードを実行するまで待たなければならないのではないかと心配するかもしれません。Python を高速化する一番良い方法は、Python を完全に避けることです。 

Gluonがこれを行う1つの方法は、
*ハイブリダイゼーション*。これについては後述する。
ここで、Python インタプリタは最初に呼び出されたときにブロックを実行します。Gluon ランタイムは何が起きているかを記録し、次回 Gluon ランタイムがそれを回避したときに Python の呼び出しをショートこれにより、場合によってはかなり高速化される可能性がありますが、制御フロー（上記のように）がネットを通るさまざまなパスで異なるブランチを下る場合は注意が必要です。興味のある読者は、現在の章を終えた後、ハイブリダイゼーションのセクション (:numref:`sec_hybridize`) をチェックしてコンパイルについて学ぶことを勧めます。
:end_tab:

:begin_tab:`pytorch`
熱心な読者は、これらの操作の一部の効率を心配し始めるかもしれません。結局のところ、高性能のディープラーニングライブラリと思われるものでは、辞書のルックアップ、コードの実行、その他多くのPythonicの処理が行われています。Python の [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) の問題はよく知られています。ディープラーニングのコンテキストでは、非常に高速な GPU が、別のジョブを実行する前に、ちっぽけな CPU が Python コードを実行するまで待たなければならないのではないかと心配するかもしれません。
:end_tab:

:begin_tab:`tensorflow`
熱心な読者は、これらの操作の一部の効率を心配し始めるかもしれません。結局のところ、高性能のディープラーニングライブラリと思われるものでは、辞書のルックアップ、コードの実行、その他多くのPythonicの処理が行われています。Python の [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) の問題はよく知られています。ディープラーニングのコンテキストでは、非常に高速な GPU が、別のジョブを実行する前に、ちっぽけな CPU が Python コードを実行するまで待たなければならないのではないかと心配するかもしれません。Python を高速化する一番良い方法は、Python を完全に避けることです。
:end_tab:

## [概要

* 画層はブロックです。
* 多くの画層が 1 つのブロックを構成できます。
* 多くのブロックが 1 つのブロックを構成できます。
* ブロックにはコードを含めることができます。
* ブロックは、パラメーターの初期化やバックプロパゲーションなど、多くのハウスキーピングを処理します。
* 層とブロックの連続的な連結は `Sequential` ブロックによって処理されます。

## 演習

1. `MySequential` を変更して Python リストにブロックを格納すると、どのような問題が発生しますか？
1. 2 つのブロック (`net1` と `net2` など) を引数として取り、両方のネットワークの連結された出力を順伝播で返すブロックを実装します。これはパラレルブロックとも呼ばれます。
1. 同じネットワークの複数のインスタンスを連結すると仮定します。同じブロックの複数のインスタンスを生成し、そこからより大きなネットワークを構築するファクトリ関数を実装します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:

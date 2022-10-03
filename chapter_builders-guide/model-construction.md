# レイヤーとモジュール
:label:`sec_model_construction`

ニューラルネットワークを初めて導入したとき、私たちは単一出力の線形モデルに焦点を当てました。ここでは、モデル全体が単一のニューロンだけで構成されています。単一のニューロンが (i) いくつかの入力セットを受け取り、(ii) 対応するスカラー出力を生成し、(iii) 関心のある目的関数を最適化するために更新できる一連の関連パラメーターがあることに注意してください。次に、複数の出力を持つネットワークについて考え始めると、ベクトル化された算術演算を利用してニューロンの層全体を特徴付けました。個々のニューロンと同様に、層 (i) は一連の入力を受け取り、(ii) 対応する出力を生成し、(iii) 調整可能なパラメーターのセットによって記述されます。ソフトマックス回帰に取り組んだとき、単一レイヤー自体がモデルでした。しかし、その後MLPを導入したときでも、このモデルは同じ基本構造を保持していると考えることができます。 

興味深いことに、MLPでは、モデル全体とその構成層の両方がこの構造を共有しています。モデル全体が生の入力 (フィーチャ) を取り込み、出力 (予測) を生成し、パラメーター (すべての構成レイヤーから組み合わされたパラメーター) を持ちます。同様に、個々の層は (前の層によって供給される) 入力を取り込み、出力 (後続のレイヤーへの入力) を生成し、後続のレイヤーから逆方向に流れる信号に従って更新される一連の調整可能なパラメーターを持ちます。 

ニューロン、レイヤー、モデルが私たちのビジネスを進めるのに十分な抽象化を与えると思うかもしれませんが、個々のレイヤーよりも大きく、モデル全体よりも小さいコンポーネントについて話すと便利なことがよくあります。たとえば、コンピュータービジョンで非常に普及しているResNet-152アーキテクチャは、数百のレイヤーを持っています。これらのレイヤーは、*レイヤーのグループ* の繰り返しパターンで構成されています。このようなネットワークを一度に 1 層ずつ実装するのは面倒です。この懸念は単なる仮説ではありません。このような設計パターンは実際には一般的です。上記のResNetアーキテクチャは、認識と検出の両方で2015年のImageNetとCOCOのコンピュータービジョンコンペティションで優勝し、多くのビジョンタスクで頼りになるアーキテクチャであり続けています。レイヤーがさまざまな繰り返しパターンで配置される同様のアーキテクチャは、現在、自然言語処理や音声を含む他のドメインに遍在しています。 

これらの複雑なネットワークを実装するために、ニューラルネットワーク*モジュール*の概念を紹介します。モジュールは、単一のレイヤー、複数のレイヤーで構成されるコンポーネント、またはモデル自体を記述できます。モジュール抽象化を使用する利点の 1 つは、多くの場合再帰的に、より大きな成果物に結合できることです。これは:numref:`fig_blocks`に示されています。オンデマンドで任意の複雑さのモジュールを生成するコードを定義することで、驚くほどコンパクトなコードを書くことができ、複雑なニューラルネットワークを実装できます。 

![Multiple layers are combined into modules, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`

プログラミングの観点から、モジュールは*クラス*で表されます。そのサブクラスは、入力を出力に変換する順伝播メソッドを定義し、必要なパラメータを格納する必要があります。一部のモジュールはパラメータをまったく必要としないことに注意してください。最後に、勾配を計算するために、モジュールはバックプロパゲーションメソッドを備えている必要があります。幸いなことに、独自のモジュールを定義するときに自動微分（:numref:`sec_autograd`で導入された）によって提供されるいくつかの舞台裏の魔法のために、パラメータと順伝播方法について心配するだけで済みます。 

[**はじめに、MLPの実装に使用したコードを再検討します**](:numref:`sec_mlp`)。次のコードは、256 ユニットと ReLU アクティベーション、続いて 10 ユニット (アクティベーション関数なし) の全接続出力層が続くネットワークを生成します。

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input  n=3}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input  n=4}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

:begin_tab:`mxnet`
この例では、`nn.Sequential` をインスタンス化してモデルを構築し、返されたオブジェクトを変数 `net` に割り当てます。次に、`add` メソッドを繰り返し呼び出し、実行される順序でレイヤーを追加します。要するに、`nn.Sequential`は、Gluonで*モジュール*を提示するクラスである`Block`の特殊な種類を定義しています。構成要素`Block`の順序付きリストを維持します。`add` メソッドは、連続する各 `Block` をリストに追加するのを容易にします。各レイヤーは `Dense` クラスのインスタンスであり、それ自体が `Block` のサブクラスであることに注意してください。フォワードプロパゲーション (`forward`) メソッドも非常に簡単です。リスト内の各`Block`を連結し、それぞれの出力を入力として次のメソッドに渡します。これまで、出力を取得するために、構築 `net(X)` を介してモデルを呼び出していたことに注意してください。これは実際には `net.forward(X)` の省略形にすぎません。これは、`Block` クラスの `__call__` メソッドを介して達成された Python の巧妙なトリックです。
:end_tab:

:begin_tab:`pytorch`
この例では、`nn.Sequential`をインスタンス化してモデルを構築し、実行する順序のレイヤーを引数として渡します。要するに、(**`nn.Sequential`は特別な種類の `Module`** を定義します)、PyTorchでモジュールを提示するクラスです。構成要素`Module`の順序付きリストを維持します。2つの全結合層はそれぞれ、`Linear`クラスのインスタンスであり、それ自体が`Module`のサブクラスであることに注意してください。フォワードプロパゲーション (`forward`) メソッドも非常に簡単です。リスト内の各モジュールを連結し、それぞれの出力を次のモジュールへの入力として渡します。これまで、出力を取得するために、構築 `net(X)` を介してモデルを呼び出していたことに注意してください。これは実際には `net.__call__(X)` の省略形にすぎません。
:end_tab:

:begin_tab:`tensorflow`
この例では、`keras.models.Sequential`をインスタンス化してモデルを構築し、実行する順序のレイヤーを引数として渡します。要するに、`Sequential`は、Kerasでモジュールを提示するクラスである`keras.Model`の特別な種類を定義しています。構成要素`Model`の順序付きリストを維持します。2つの全結合層はそれぞれ、`Dense`クラスのインスタンスであり、それ自体が`Model`のサブクラスであることに注意してください。フォワードプロパゲーション (`call`) メソッドも非常に簡単です。リスト内の各モジュールを連結し、それぞれの出力を次のモジュールへの入力として渡します。これまで、出力を取得するために、構築 `net(X)` を介してモデルを呼び出していたことに注意してください。これは実際には `net.call(X)` の省略形に過ぎません。これは、モジュールクラスの `__call__` メソッドを介して達成された Python の巧妙なトリックです。
:end_tab:

## [**カスタムモジュール**]

おそらく、モジュールがどのように機能するかについての直感を養う最も簡単な方法は、モジュールを自分で実装することです。独自のカスタムモジュールを実装する前に、各モジュールが提供しなければならない基本機能を簡単に要約します。 

1. 入力データをそのフォワードプロパゲーションメソッドの引数として取り込みます。
1. フォワードプロパゲーションメソッドが値を返すようにして出力を生成します。出力は入力とは異なる形状になる場合があることに注意してください。たとえば、上記のモデルの最初の全結合層は、任意の次元の入力を取り込みますが、次元 256 の出力を返します。
1. 入力に対する出力の勾配を計算します。この勾配は、バックプロパゲーションメソッドを介してアクセスできます。通常、これは自動的に行われます。
1. フォワードプロパゲーション計算の実行に必要なパラメーターを保存し、そのパラメーターへのアクセスを提供します。
1. 必要に応じてモデルパラメーターを初期化します。

次のスニペットでは、256 の隠れユニットを持つ 1 つの隠れ層と 10 次元の出力層を持つ MLP に対応するモジュールをゼロからコーディングします。以下の `MLP` クラスは、モジュールを表すクラスを継承していることに注意してください。親クラスのメソッドに大きく依存し、独自のコンストラクタ (Python では `__init__` メソッド) とフォワード伝播メソッドのみを提供します。

```{.python .input  n=5}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Call the constructor of the MLP parent class nn.Block to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input  n=6}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input  n=7}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # Call the constructor of the parent class tf.keras.Model to perform
        # the necessary initialization
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def call(self, X):
        return self.out(self.hidden((X)))
```

まず、順伝播方法に焦点を当てましょう。`X`を入力として受け取り、活性化関数を適用して隠れ表現を計算し、そのロジットを出力することに注意してください。この`MLP`の実装では、両方のレイヤーがインスタンス変数です。これが妥当な理由を理解するために、`net1` と `net2` の 2 つの MLP をインスタンス化し、異なるデータで学習させることを想像してみてください。当然、それらは2つの異なる学習モデルを表すと予想されます。 

順伝播メソッドを呼び出すたびに、コンストラクターで [**MLPのレイヤーをインスタンス化する**]（**そしてこれらのレイヤーを呼び出す**）。いくつかの重要な詳細をメモしておきます。まず、カスタマイズされた`__init__`メソッドは、`super().__init__()`を介して親クラスの`__init__`メソッドを呼び出し、ほとんどのモジュールに適用できる定型コードを再記述する手間を省きます。次に、完全に接続された 2 つのレイヤーをインスタンス化し、それらを `self.hidden` と `self.out` に割り当てます。新しいレイヤーを実装しない限り、バックプロパゲーションメソッドやパラメーターの初期化について心配する必要はありません。システムは、これらのメソッドを自動的に生成します。これやってみよう。

```{.python .input  n=8}
%%tab all
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

モジュール抽象化の重要な長所は、その汎用性にあります。モジュールをサブクラス化して、レイヤー (全結合レイヤークラスなど)、モデル全体 (上記の `MLP` クラスなど)、または中程度の複雑さのさまざまなコンポーネントを作成できます。畳み込みニューラルネットワークを扱う場合など、次の章でこの汎用性を活用します。 

## [**シーケンシャルモジュール**]

ここで、`Sequential` クラスがどのように機能するかを詳しく見てみましょう。`Sequential`は、他のモジュールをデイジーチェーン接続するように設計されていることを思い出してください。独自の簡略化された `MySequential` を構築するには、次の 2 つの主要なメソッドを定義する必要があります。
1. モジュールを一つずつリストに追加するメソッド。
2. 追加されたのと同じ順序でモジュールのチェーンを介して入力を渡すフォワードプロパゲーションメソッド。

次の `MySequential` クラスは、デフォルトの `Sequential` クラスと同じ機能を提供します。

```{.python .input  n=10}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume that
        # it has a unique name. We save it in the member variable _children of
        # the Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize method, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input  n=11}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input  n=12}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` メソッドは、順序付きディクショナリ `_children` に 1 つのブロックを追加します。すべての Gluon `Block` がなぜ `_children` 属性を持っているのか、そしてなぜ私たちがPythonリストを定義するのではなくそれを使ったのか不思議に思うかもしれません。要するに、`_children`の主な利点は、ブロックのパラメーターの初期化中に、Gluonが`_children`ディクショナリ内を見て、パラメーターも初期化する必要があるサブブロックを見つけることがわかっていることです。
:end_tab:

:begin_tab:`pytorch`
`__init__` メソッドでは、`add_modules` メソッドを呼び出してすべてのモジュールを追加します。これらのモジュールには、後で `children` メソッドでアクセスできます。このようにして、システムは追加されたモジュールを認識し、各モジュールのパラメータを適切に初期化します。
:end_tab:

`MySequential`のフォワード伝播メソッドが呼び出されると、追加された各モジュールは、追加された順序で実行されます。これで、`MySequential` クラスを使用して MLP を再実装できます。

```{.python .input  n=13}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input  n=14}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input  n=15}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

この`MySequential`の使用法は、`Sequential`クラス用に以前に記述したコード（:numref:`sec_mlp`で説明されているように）と同じであることに注意してください。 

## [**フォワードプロパゲーション方式でコードを実行する**]

`Sequential` クラスを使用すると、モデルの構築が容易になり、独自のクラスを定義しなくても新しいアーキテクチャを組み立てることができます。ただし、すべてのアーキテクチャが単純なデイジーチェーンであるとは限りません。より高い柔軟性が必要な場合は、独自のブロックを定義したいと思うでしょう。たとえば、Python の制御フローをフォワードプロパゲーションメソッド内で実行するとします。さらに、単に定義済みのニューラルネットワーク層に依存するのではなく、任意の数学的演算を実行したい場合があります。 

今まで、ネットワーク内のすべての操作が、ネットワークのアクティベーションとそのパラメータに基づいて動作していたことに気づいたかもしれません。ただし、前のレイヤーの結果でも更新可能なパラメーターでもない用語を取り入れたい場合があります。これらを*定数パラメータ*と呼びます。たとえば、関数 $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ を計算するレイヤーが必要であるとします。ここで、$\mathbf{x}$ は入力、$\mathbf{w}$ はパラメーター、$c$ は最適化中に更新されない特定の定数です。そこで、`FixedHiddenMLP` クラスを以下のように実装します。

```{.python .input  n=16}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # Random weight parameters created with the `get_constant` method
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
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
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

この`FixedHiddenMLP`モデルでは、重み（`self.rand_weight`）がインスタンス化時にランダムに初期化され、その後は一定になる隠れ層を実装します。この重みはモデルパラメータではないため、バックプロパゲーションによって更新されることはありません。次に、ネットワークはこの「固定」層の出力を全結合層に渡します。 

出力を返す前に、私たちのモデルは何か変わったことをしたことに注意してください。whileループを実行し、$\ell_1$ノルムが$1$より大きいという条件でテストし、条件を満たすまで出力ベクトルを $2$ で割りました。最後に、`X` のエントリの合計を返しました。私たちの知る限りでは、この操作を実行する標準的なニューラルネットワークはありません。この特定の操作は、実際のタスクでは役に立たないことに注意してください。ここでのポイントは、ニューラルネットワーク計算のフローに任意のコードを統合する方法を示すことだけです。

```{.python .input}
%%tab all
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

[**モジュールを組み立てるさまざまな方法を組み合わせて組み合わせることができます。**] 次の例では、いくつかの創造的な方法でモジュールをネストします。

```{.python .input}
%%tab mxnet
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
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
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

## まとめ

* レイヤーはモジュールです。
* 多くのレイヤーが 1 つのモジュールを構成できます。
* 多くのモジュールが 1 つのモジュールを構成できます。
* モジュールにはコードを含めることができます。
* モジュールは、パラメータの初期化やバックプロパゲーションなど、多くのハウスキーピングを処理します。
* レイヤーとモジュールの連続的な連結は、`Sequential` モジュールによって処理されます。

## 演習

1. `MySequential` を Python リストにモジュールを格納するように変更すると、どのような問題が発生しますか?
1. `net1`と`net2`の2つのモジュールを引数として取り、両方のネットワークの連結された出力を順伝播で返すモジュールを実装します。これは並列モジュールとも呼ばれます。
1. 同じネットワークの複数のインスタンスを連結するとします。同じモジュールの複数のインスタンスを生成するファクトリ関数を実装し、そこからより大きなネットワークを構築します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:

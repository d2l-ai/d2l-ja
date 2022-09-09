# ファイル I/O

ここまで、データの処理方法と、ディープラーニングモデルの構築、トレーニング、およびテスト方法について説明しました。しかし、ある時点で、学習したモデルに十分満足して、後でさまざまな状況で使用できるように結果を保存したいと思うでしょう（おそらく展開の予測を行うため）。さらに、長いトレーニングプロセスを実行する場合、サーバーの電源コードをつまずいても、数日分の計算が失われないように、中間結果を定期的に保存する（チェックポイント）ことがベストプラクティスです。したがって、個々の重みベクトルとモデル全体の両方をロードして保存する方法を学ぶときです。このセクションでは、両方の問題について説明します。 

## (**テンソルの読み込みと保存**)

個々のテンソルについては、`load`と`save`の関数を直接呼び出して、それぞれ読み書きすることができます。どちらの関数も名前を指定する必要があり、`save`では入力として変数を保存する必要があります。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save('x-file.npy', x)
```

これで、保存されたファイルのデータをメモリに読み戻すことができます。

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

[**テンソルのリストを保存し、メモリに読み戻すことができます。**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

[**文字列からテンソルにマップする辞書を書いて読むこともできます。**] これは、モデル内のすべての重みを読み書きしたい場合に便利です。

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**モデルパラメータのロードと保存**]

個々のウェイトベクトル (または他のテンソル) を保存することは便利ですが、モデル全体を保存 (そして後でロード) したい場合は非常に面倒です。結局のところ、何百ものパラメータグループが散在している可能性があります。このため、ディープラーニングフレームワークは、ネットワーク全体をロードおよび保存するための組み込み機能を提供します。注意すべき重要な点は、これによりモデル全体ではなくモデル*パラメータ*が保存されることです。たとえば、3 層の MLP がある場合、アーキテクチャを個別に指定する必要があります。これは、モデル自体に任意のコードを含めることができるため、自然にシリアル化できないためです。したがって、モデルを復元するには、コードでアーキテクチャを生成し、ディスクからパラメーターをロードする必要があります。(**おなじみのMLPから始めましょう。**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

次に、「mlp.params」という名前で [**モデルのパラメータをファイルとして保存します**]。

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

モデルを復元するために、元の MLP モデルのクローンをインスタンス化します。モデルパラメータをランダムに初期化する代わりに、[**ファイルに保存されているパラメータを直接読み取る**]。

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

両方のインスタンスが同じモデルパラメーターをもつため、同じ入力 `X` の計算結果は同じになるはずです。これを確認しましょう。

```{.python .input}
%%tab all
Y_clone = clone(X)
Y_clone == Y
```

## まとめ

* `save`および`load`関数は、テンソルオブジェクトのファイル入出力を実行するために使用できます。
* パラメータディクショナリを介して、ネットワークのパラメータセット全体を保存およびロードできます。
* アーキテクチャの保存は、パラメータではなくコードで行う必要があります。

## 演習

1. トレーニング済みのモデルを別のデバイスに展開する必要がない場合でも、モデルパラメータを保存することの実際的な利点は何ですか？
1. ネットワークの一部だけを再利用して、異なるアーキテクチャのネットワークに組み込むと仮定します。以前のネットワークの最初の2つのレイヤーを新しいネットワークでどのように使用しますか？
1. ネットワークアーキテクチャとパラメータを保存するにはどうしたらいいですか？アーキテクチャにどのような制限を課しますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:

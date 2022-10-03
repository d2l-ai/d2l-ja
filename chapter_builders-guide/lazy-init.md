# 遅延初期化
:label:`sec_lazy_init`

これまでのところ、ネットワークの設定がずさんなことで逃げ出したように思えるかもしれません。具体的には、次の直感的でないことを行いましたが、動作するようには思えないかもしれません。 

* 入力の次元を指定せずにネットワークアーキテクチャを定義しました。
* 前のレイヤーの出力ディメンションを指定せずにレイヤーを追加しました。
* モデルに含めるべきパラメータの数を決定するのに十分な情報を提供する前に、これらのパラメータを「初期化」しました。

私たちのコードがまったく動作することに驚くかもしれません。結局のところ、ディープラーニングフレームワークは、ネットワークの入力次元がどうなるかを判断する方法はありません。ここでの秘訣は、フレームワークが初期化を*延期し、モデルに初めてデータを渡すまで待って、各レイヤーのサイズをその場で推測することです。 

その後、畳み込みニューラルネットワークを扱う場合、入力次元（画像の解像度）が後続の各レイヤーの次元性に影響を与えるため、この手法はさらに便利になります。したがって、コードの作成時に次元が何であるかを知る必要なくパラメータを設定できると、モデルの指定とその後の変更作業が大幅に簡素化されます。次に、初期化の仕組みについて詳しく説明します。 

はじめに、MLP をインスタンス化しましょう。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

この時点では、入力の次元が不明であるため、ネットワークは入力層の重みの次元を知ることができない可能性があります。そのため、フレームワークはまだパラメータを初期化していません。以下のパラメータにアクセスして確認します。

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
パラメーターオブジェクトが存在する間、各レイヤーへの入力ディメンションは -1 としてリストされることに注意してください。MXNet は、パラメーターの次元が不明であることを示すために、特別な値 -1 を使用します。この時点で、`net[0].weight.data()` にアクセスしようとすると、パラメータにアクセスする前にネットワークを初期化する必要があることを示すランタイムエラーが発生します。ここで、`initialize` メソッドでパラメータを初期化しようとするとどうなるか見てみましょう。
:end_tab:

:begin_tab:`tensorflow`
各レイヤーオブジェクトは存在しますが、ウェイトは空です。`net.get_weights()` を使用すると、ウェイトがまだ初期化されていないため、エラーがスローされます。
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
ご覧のとおり、何も変わっていません。入力次元が不明な場合、initialize を呼び出してもパラメーターは真に初期化されません。代わりに、この呼び出しは、パラメーターを初期化する (およびオプションで、どのディストリビューションに応じて) MXNet に登録します。
:end_tab:

次に、ネットワークを介してデータを渡して、フレームワークが最終的にパラメータを初期化するようにします。

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

入力次元20がわかるとすぐに、フレームワークは20の値を入力することで第1レイヤーの重みマトリックスの形状を識別できます。最初のレイヤーの形状を認識したら、フレームワークは2番目のレイヤーに進み、すべての形状がわかるまで計算グラフを介して続きます。この場合、最初のレイヤーのみが遅延初期化を必要としますが、フレームワークは順次初期化されることに注意してください。すべてのパラメータ形状がわかれば、フレームワークは最終的にパラメータを初期化できます。

:begin_tab:`pytorch`
次のメソッドは、ネットワークを介してダミー入力を渡して予行運転を行い、すべてのパラメータ形状を推測し、続いてパラメータを初期化します。これは、後でデフォルトのランダム初期化が望ましくない場合に使用されます。
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

## まとめ

* 遅延初期化は便利で、フレームワークがパラメータ形状を自動的に推測できるため、アーキテクチャの変更が容易になり、一般的なエラーの原因を1つ排除できます。
* モデルを介してデータを渡して、フレームワークが最終的にパラメータを初期化するようにすることができます。

## 演習

1. 入力ディメンションを最初のレイヤーに指定し、後続のレイヤーには指定しないとどうなりますか？すぐに初期化できますか？
1. 不一致のディメンションを指定するとどうなりますか?
1. 様々な次元のインプットがあるとしたら、何をする必要がありますか？ヒント:パラメータ同士を見てください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:

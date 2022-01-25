# 多層パーセプトロンの簡潔な実装
:label:`sec_mlp_concise`

ご想像のとおり、(**高レベルAPIに頼ることで、MLPをより簡潔に実装できます**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## モデル

softmax 回帰実装 (:numref:`sec_softmax_concise`) の簡潔な実装と比較すると、唯一の違いは以下を追加することです。
*2つの* 完全に接続されたレイヤー
(以前は、*one* を追加しました)。1つ目は [**私たちの隠れ層**] で、(**256 個の隠しユニットを含み、ReLU アクティベーション機能を適用する**)。2 つ目は出力レイヤーです。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
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
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

[**トレーニングループ**] はソフトマックス回帰を実装したときとまったく同じです。このモジュール性により、モデルアーキテクチャに関する事項を直交的な考慮事項から切り離すことができます。

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [概要

* 高レベル API を使用することで、MLP をより簡潔に実装できます。
* 同じ分類問題では、MLP の実装はソフトマックス回帰の実装と同じですが、活性化関数をもつ隠れ層が追加されている点が異なります。

## 演習

1. 異なる数の隠れ層を追加してみてください (学習率を変更することもできます)。どの設定が最適ですか？
1. さまざまなアクティベーション機能を試してみてください。どれが一番効果的ですか？
1. ウェイトの初期化にはさまざまなスキームを試してください。どの方法が最適ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:

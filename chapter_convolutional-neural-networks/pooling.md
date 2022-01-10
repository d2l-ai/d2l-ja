# プーリング
:label:`sec_pooling`

多くの場合、画像を処理するときに、隠れた表現の空間分解能を徐々に下げて、情報を集約して、ネットワーク内で高くなるほど、各隠れノードが敏感な（入力の）受容場が大きくなるようにします。 

私たちの最終的なタスクは、画像について何かグローバルな質問をすることがよくあります。例えば、*猫が含まれていますか？* したがって、通常、最終層の単位は入力全体に影響を受けます。情報を徐々に集約し、より粗いマップと粗いマップを生成することで、畳み込み層の利点をすべて処理の中間層に維持しながら、最終的にグローバル表現を学習するというこの目標を達成します。 

さらに、エッジ (:numref:`sec_conv_layer` で説明) などの下位レベルの特徴を検出する場合、表現を翻訳に対して多少不変にしたいことがよくあります。たとえば、イメージ `X` を黒と白の間で鮮明に描写し、イメージ全体を 1 ピクセル右にシフトした場合 (つまり `Z[i, j] = X[i, j + 1]`)、新しいイメージ `Z` の出力は大きく異なる可能性があります。エッジが 1 ピクセルずれた状態になります。実際には、オブジェクトがまったく同じ場所に存在することはほとんどありません。実際、三脚や静止物があっても、シャッターの動きによるカメラの振動により、すべてが1ピクセルほどずれてしまうことがあります (ハイエンドカメラには、この問題に対処するための特別な機能が搭載されています)。 

このセクションでは、畳み込み層の位置に対する感度を軽減することと、表現を空間的にダウンサンプリングするという2つの目的を果たす*プーリング層*を紹介します。 

## 最大プーリングと平均プーリング

畳み込み層と同様に、*pooling* 演算子は固定形状ウィンドウで構成され、そのストライドに従って入力内のすべての領域にわたってスライドされ、固定形状ウィンドウ (*プーリングウィンドウ* とも呼ばれる) が通過する位置ごとに 1 つの出力を計算します。ただし、畳み込み層の入力とカーネルの相互相関計算とは異なり、プーリング層にはパラメーターは含まれません (*kernel* はありません)。その代わり、プーリング演算子は決定論的であり、通常はプーリングウィンドウ内の要素の最大値または平均値を計算します。これらの演算は、それぞれ*最大プーリング* (略して*最大プーリング*) と*平均プーリング* と呼ばれます。 

どちらの場合も、相互相関演算子と同様に、プーリングウィンドウは入力テンソルの左上から開始し、入力テンソルを左から右、上から下にスライドすると考えることができます。プーリングウィンドウがヒットする各位置で、最大プーリングが採用されているか平均プーリングが採用されているかに応じて、ウィンドウ内の入力サブテンソルの最大値または平均値が計算されます。 

![Maximum pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` の出力テンソルの高さは 2、幅は 2 です。4 つの要素は、各プーリングウィンドウの最大値から導出されます。 

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

プーリングウィンドウ形状が $p \times q$ のプーリング層は $p \times q$ プーリング層と呼ばれます。プーリング操作は $p \times q$ プーリングと呼ばれます。 

このセクションの冒頭で述べたオブジェクトのエッジ検出の例に戻りましょう。畳み込み層の出力を $2\times 2$ 最大プーリングの入力として使用します。畳み込み層の入力を `X` に設定し、プーリング層の出力を `Y` に設定します。`X[i, j]` と `X[i, j + 1]` の値が異なるかどうか、または `X[i, j + 1]` と `X[i, j + 2]` の値が異なるかどうかにかかわらず、プーリング層は常に `Y[i, j] = 1` を出力します。つまり、$2\times 2$ 最大プーリング層を使用すれば、畳み込み層によって認識されるパターンが、高さまたは幅が 1 要素しか移動しないかどうかを検出できます。 

以下のコードでは、関数 `pool2d` で (**プーリング層の順伝播を実装**) しています。この関数は :numref:`sec_conv_layer` の `corr2d` 関数と似ています。ただし、ここにはカーネルがなく、出力を入力内の各領域の最大値または平均値として計算しています。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
#@tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

:numref:`fig_pooling` に入力テンソル `X` を構築して [** 2 次元の最大プーリング層の出力を検証する**] ことができます。

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

また、(**平均プーリング層**) を試します。

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## [**パディングとストライド**]

畳み込み層と同様に、プーリング層も出力の形状を変えることができます。また、前述のように、入力をパディングしてストライドを調整することで、希望する出力形状が得られるように操作を変更できます。ディープラーニングフレームワークの組み込みの 2 次元最大プーリング層を使用して、プーリング層でのパディングとストライドの使用を実証できます。最初に、形状が 4 次元をもつ入力テンソル `X` を構築します。ここで、例の数 (バッチサイズ) とチャネル数はどちらも 1 です。

:begin_tab:`tensorflow`
テンソルフローは*channels-last* 入力を優先し、最適化されていることに注意することが重要です。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

デフォルトでは (**フレームワークのビルトインクラスのインスタンス内のストライドとプーリングウィンドウは同じ形状です。**) 以下では `(3, 3)` という形状のプーリングウィンドウを使用するため、ストライド形状はデフォルトで `(3, 3)` になります。

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

[**ストライドとパディングは手動で指定できます**]

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`mxnet`
もちろん、任意の矩形プーリングウィンドウを指定し、高さと幅のパディングとストライドをそれぞれ指定できます。
:end_tab:

:begin_tab:`pytorch`
もちろん、(**任意の矩形プーリングウィンドウを指定し、高さと幅にはパディングとストライドを指定**) できます。
:end_tab:

:begin_tab:`tensorflow`
もちろん、任意の矩形プーリングウィンドウを指定し、高さと幅のパディングとストライドをそれぞれ指定できます。
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

## 複数チャネル

マルチチャネル入力データを処理する場合、[**プーリング層は各入力チャネルを個別にプールします**]。畳み込み層のようにチャネルにわたって入力を合計するのではなく。つまり、プーリング層の出力チャネル数は入力チャネル数と同じになります。以下では、チャネル次元でテンソル `X` と `X + 1` を連結して、2 つのチャネルをもつ入力を作成します。

:begin_tab:`tensorflow`
channels-last 構文のため、TensorFlow では最後のディメンションに沿って連結する必要があることに注意してください。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

ご覧のとおり、プーリング後も出力チャンネル数はまだ 2 です。

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

:begin_tab:`tensorflow`
テンソルフロープーリングの出力は一見異なっているように見えますが、数値的には同じ結果が MXNet と PyTorch として表示されます。違いは次元性にあり、出力を垂直方向に読み取ると、他の実装と同じ出力が得られます。
:end_tab:

## [概要

* プーリングウィンドウ内の入力要素を取得すると、最大プーリング演算では最大値が出力として割り当てられ、平均プーリング演算では平均値が出力として割り当てられます。
* プーリング層の主な利点の 1 つは、畳み込み層の位置に対する過度の感度を軽減することです。
* プーリング層のパディングとストライドを指定できます。
* 最大プーリングと 1 より大きいストライドを組み合わせると、空間次元 (幅や高さなど) を減らすことができます。
* プーリング層の出力チャネル数は、入力チャネル数と同じです。

## 演習

1. 平均プーリングを畳み込み層の特殊なケースとして実装できますか？もしそうなら、それをしなさい。
1. 畳み込み層の特殊なケースとして最大プーリングを実装できますか？もしそうなら、それをしなさい。
1. プーリング層の計算コストはどれくらいですか？プーリング層への入力のサイズが $c\times h\times w$ で、プーリングウィンドウの形状が $p_h\times p_w$ で、パディングが $(p_h, p_w)$、ストライドが $(s_h, s_w)$ であると仮定します。
1. 最大プーリングと平均プーリングが異なると予想されるのはなぜですか。
1. 個別の最小プーリング層が必要ですか？他のオペレーションと交換してもらえますか？
1. 平均プーリングと最大プーリングの間に考慮できる別の操作がありますか（ヒント：softmaxを思い出してください）。どうしてそんなに人気がないのでしょう？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:

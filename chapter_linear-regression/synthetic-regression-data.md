```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 合成回帰データ
:label:`sec_synthetic-regression-data`

機械学習とは、データから情報を抽出することです。合成データから何を学べるのか不思議に思うかもしれません。私たち自身が人工的なデータ生成モデルに組み込んだパターンについては本質的に気にしないかもしれませんが、そのようなデータセットは教訓的な目的に役立ち、学習アルゴリズムの特性を評価し、実装が期待どおりに機能することを確認するのに役立ちます。たとえば、*アプリオリ*で正しいパラメータがわかっているデータを作成すると、モデルが実際にそれらを回復できることを検証できます。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## データセットの生成

この例では、簡潔にするために低次元で作業します。次のコードスニペットは、標準正規分布から抽出された2次元の特徴を含む1000の例を生成します。結果として得られる計画マトリックス $\mathbf{X}$ は $\mathbb{R}^{1000 \times 2}$ に属します。ここでは、*グラウンドトゥルース* 線形関数を適用して各ラベルを生成し、加法性ノイズ $\epsilon$ によってそれらを破壊し、各例で独立して同じように描画します。 

(**$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**) 

便宜上、$\epsilon$は平均$\mu= 0$と標準偏差$\sigma = 0.01$の正規分布から導出されると仮定します。オブジェクト指向設計では、`d2l.DataModule` (:numref:`oo-design-data` で導入) のサブクラスの `__init__` メソッドにコードを追加することに注意してください。追加のハイパーパラメータを設定することは良い習慣です。これを`save_hyperparameters()`で達成します。`batch_size`は後で決定されます。

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise            
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

以下では、真のパラメータを $\mathbf{w} = [2, -3.4]^\top$ と $b = 4.2$ に設定します。後で、これらの*グラウンドトゥルース*値と照らし合わせて推定パラメータを確認できます。

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**`features` の各行は $\mathbb{R}^2$ のベクトルで構成され、`labels` の各行はスカラーです。**] 最初のエントリを見てみましょう。

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## データセットの読み取り

機械学習モデルをトレーニングするには、多くの場合、データセットを複数回通過し、一度に 1 つのミニバッチの例を取得する必要があります。このデータは、モデルの更新に使用されます。これがどのように機能するかを説明するために、[**`get_dataloader`関数を実装し、**] `add_to_class`（:numref:`oo-design-utilities`で導入）を介して`SyntheticRegressionData`クラスのメソッドとして登録します。それ (**バッチサイズ、特徴の行列、およびラベルのベクトルを取り、サイズ`batch_size`のミニバッチを生成します**) そのため、各ミニバッチは特徴とラベルのタプルで構成されます。トレーニングモードか検証モードかに注意する必要があることに注意してください。前者ではランダムな順序でデータを読み取る必要があるのに対し、後者の場合、事前に定義された順序でデータを読み取ることができることがデバッグの目的で重要になる場合があります。

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet') or tab.selected('pytorch'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

直感を構築するために、データの最初のミニバッチを調べてみましょう。フィーチャの各ミニバッチは、そのサイズと入力フィーチャの次元の両方を提供します。同様に、ラベルのミニバッチは、`batch_size`によって与えられた一致する形状になります。

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

一見無害に見えますが、`iter(data.train_dataloader())`の呼び出しは、Pythonのオブジェクト指向設計の力を示しています。`SyntheticRegressionData` クラスにメソッドを追加したことに注意してください。
** `data` オブジェクトを作成した後。 
それにもかかわらず、オブジェクトは、クラスに機能を*事後*追加することで恩恵を受けます。 

反復を通して、データセット全体が使い果たされるまで、個別のミニバッチを取得します（これを試してください）。上記で実装された反復は教訓的な目的には適していますが、実際の問題で私たちを困らせるような方法では非効率的です。たとえば、すべてのデータをメモリにロードし、大量のランダムメモリアクセスを実行する必要があります。ディープラーニングフレームワークに実装されたビルトインイテレーターは、かなり効率的で、ファイルに格納されたデータ、ストリームを介して受信したデータ、オンザフライで生成または処理されたデータなどのソースを処理できます。次に、組み込みのイテレータを使って同じ関数を実装してみましょう。 

## データローダーの簡潔な実装

独自のイテレータを書く代わりに、[**フレームワーク内の既存のAPIを呼び出してデータをロードします。**] 前と同じように、機能`X`とラベル`y`を持つデータセットが必要です。それ以上に、組み込みのデータローダーに`batch_size`を設定し、サンプルを効率的にシャッフルできるようにします。

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

新しいデータローダーは、より効率的で機能が追加されている点を除いて、前のデータローダーと同じように動作します。

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

たとえば、フレームワーク API によって提供されるデータローダーは、組み込みの `__len__` メソッドをサポートしているため、長さ、つまりバッチ数をクエリできます。

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## まとめ

データローダーは、データのロードと操作のプロセスを抽象化する便利な方法です。このように、同じ機械学習*アルゴリズム*が、変更を必要とせずに多くの異なるタイプとデータソースを処理することができます。データローダーの優れた点の 1 つは、構成できることです。たとえば、画像を読み込んで、それらを切り抜いたり、別の方法で変更したりする後処理フィルターがあるとします。そのため、データローダーはデータ処理パイプライン全体を記述するために使用できます。  

モデル自体に関しては、2次元線形モデルは、私たちが遭遇するかもしれないほど単純なモデルです。これにより、データ量が不十分だったり、方程式系が不十分であることを心配することなく、回帰モデルの精度をテストできます。これを次のセクションで有効に活用します。   

## 演習

1. 例の数をバッチサイズで割ることができない場合はどうなりますか。フレームワークのAPIを使用して別の引数を指定してこの動作を変更するにはどうすればいいですか?
1. パラメータベクトル`w`のサイズと`num_examples`の例の数の両方が大きい巨大なデータセットを生成したい場合はどうなりますか？ 
    1. すべてのデータをメモリに保持できない場合はどうなりますか？
    1. データがディスク上に保持されている場合、どのようにデータをシャッフルしますか？あなたの仕事は、ランダムな読み取りまたは書き込みをあまり必要としない、*効率的な*アルゴリズムを設計することです。ヒント: [pseudorandom permutation generators](https://en.wikipedia.org/wiki/Pseudorandom_permutation) allow you to design a reshuffle without the need to store the permutation table explicitly :cite:`Naor.Reingold.1999`。 
1. イテレータが呼び出されるたびに、その場で新しいデータを生成するデータジェネレータを実装します。 
1. 呼び出されるたびに*同じ*データを生成するランダムデータジェネレータをどのように設計しますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6664)
:end_tab:

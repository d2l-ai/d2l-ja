# AutoRec: オートエンコーダによるレーティング予測

行列因数分解モデルは評価予測タスクで適切なパフォーマンスを発揮しますが、基本的には線形モデルです。したがって、このようなモデルでは、ユーザーの好みを予測する複雑な非線形および複雑な関係を捉えることができません。このセクションでは、非線形ニューラルネットワーク協調フィルタリングモデル AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015` を紹介します。協調フィルタリング (CF) をオートエンコーダアーキテクチャで識別し、明示的なフィードバックに基づいて非線形変換を CF に統合することを目的としています。ニューラルネットワークはあらゆる連続関数を近似できることが証明されており、行列因数分解の限界に対処し、行列因数分解の表現力を豊かにするのに適しています。 

AutoRec は、入力レイヤー、隠れレイヤー、再構成 (出力) レイヤーで構成されるオートエンコーダーと同じ構造です。オートエンコーダは、入力を隠れた (通常は低次元) 表現にコード化するために、入力を出力にコピーすることを学習するニューラルネットワークです。AutoRec では、ユーザー/アイテムを低次元空間に明示的に埋め込むのではなく、対話行列の列/行を入力として使用し、出力層で相互作用行列を再構築します。 

一方、AutoRec は従来のオートエンコーダとは異なります。AutoRec は、隠れた表現を学習するのではなく、出力レイヤの学習/再構築に重点を置いています。完成した評価行列の再構築を目的として、部分的に観測された交互作用行列を入力として使用します。その間、入力の欠落したエントリは、レコメンデーションの目的で再構成によって出力層に埋められます。  

AutoRec には、ユーザベースと項目ベースの 2 種類があります。簡潔にするために、ここでは項目ベースの AutoRec のみを紹介します。それに応じて、ユーザベースの AutoRec を派生させることができます。 

## モデル

$\mathbf{R}_{*i}$ は評価マトリックスの $i^\mathrm{th}$ 列を表し、未知の評価はデフォルトでゼロに設定されます。ニューラルアーキテクチャは次のように定義されます。 

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

$f(\cdot)$ と $g(\cdot)$ は活性化関数を表し、$\mathbf{W}$ と $\mathbf{V}$ は重み行列を表し、$\mu$ と $b$ はバイアスです。$h( \cdot )$ は AutoRec のネットワーク全体を表すとします。出力 $h(\mathbf{R}_{*i})$ は、評価マトリックスの $i^\mathrm{th}$ 列を再構成したものです。 

次の目的関数は、再構成誤差を最小化することを目的としています。 

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

$\| \cdot \|_{\mathcal{O}}$ は、観測された評価の寄与のみが考慮されることを意味します。つまり、観測された入力に関連する重みのみが逆伝播中に更新されます。

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## モデルの実装

一般的なオートエンコーダは、エンコーダとデコーダで構成されます。符号化器は入力を隠れ表現に投影し、復号化器は隠れ層を再構成層にマッピングします。このプラクティスに従い、密度の高い層をもつエンコーダとデコーダを作成します。エンコーダのアクティベーションはデフォルトで `sigmoid` に設定され、デコーダのアクティベーションは適用されません。過剰適合を減らすために、エンコード変換の後にドロップアウトが含まれます。観測されない入力の勾配はマスクされ、観測された評価のみがモデル学習プロセスに寄与することが保証されます。

```{.python .input  n=2}
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## エバリュエーターの再実装

入力と出力が変更されたため、RMSE を精度の尺度として使用しながら、評価関数を再実装する必要があります。

```{.python .input  n=3}
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## モデルのトレーニングと評価

それでは、MovieLens データセットで AutoRec をトレーニングして評価してみましょう。検定RMSEが行列分解モデルよりも低いことがはっきりと分かり、評価予測タスクにおけるニューラルネットワークの有効性が確認されました。

```{.python .input  n=4}
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## [概要

* 非線形層とドロップアウト正則化を積分しながら、行列分解アルゴリズムをオートエンコーダーで組み立てることができます。 
* MovieLens 100K データセットでの実験から、AutoRec は行列因数分解よりも優れたパフォーマンスを達成していることが示されています。

## 演習

* AutoRec の非表示寸法を変更して、モデルのパフォーマンスに与える影響を確認します。
* 非表示レイヤーをさらに追加してみます。モデルのパフォーマンスを向上させることは役に立ちますか？
* デコーダとエンコーダのアクティベーション機能のより優れた組み合わせを見つけられますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:

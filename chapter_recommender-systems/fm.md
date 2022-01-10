# 因数分解機

2010年にSteffen Rendleによって提案された因数分解マシン (FM) :cite:`Rendle.2010` は、分類、回帰、およびランク付けタスクに使用できる教師ありアルゴリズムです。それはすぐに注目され、予測と推奨を行うための一般的で影響力のある方法になりました。特に、線形回帰モデルと行列因数分解モデルの一般化です。さらに、多項式カーネルを持つサポートベクターマシンを思い起こさせます。線形回帰と行列因数分解に対する因数分解マシンの強みは次のとおりです。(1) $\chi$ は多項式の次数で、通常は 2 に設定される $\chi$ ウェイ変数交互作用をモデル化できる。(2) 因数分解マシンに関連付けられた高速最適化アルゴリズムは、多項式の計算時間を線形複雑度に変換し、特に高次元のスパース入力に対して非常に効率的です。これらの理由から、因数分解機は現代の広告や製品レコメンデーションに広く採用されています。技術的な詳細と実装については、以下で説明します。 

## 双方向因数分解機

正式には、$x \in \mathbb{R}^d$ は 1 つの標本の特徴ベクトルを示し、$y$ は対応するラベル (バイナリクラス「クリック/非クリック」など) の実数値のラベルまたはクラスラベルであるとします。次数が 2 の因数分解マシンのモデルは次のように定義されます。 

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

$\mathbf{w}_0 \in \mathbb{R}$ はグローバルバイアス、$\mathbf{w} \in \mathbb{R}^d$ は i 番目の変数の重み、$\mathbf{V} \in \mathbb{R}^{d\times k}$ は特徴の埋め込みを表し、$\mathbf{v}_i$ は $\mathbf{V}$ の $i^\mathrm{th}$ 行、$k$ は潜在因子の次元、$\langle\cdot, \cdot \rangle$ は 2 つのベクトルの内積を表します。$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ は交互作用をモデル化します。$i^\mathrm{th}$ と $j^\mathrm{th}$ の間の機能です。一部の機能の相互作用は理解しやすいため、専門家が設計できます。ただし、他のほとんどのフィーチャの相互作用はデータに隠れており、識別が困難です。そのため、フィーチャの相互作用を自動的にモデリングすると、フィーチャエンジニアリングの労力を大幅に削減できます。最初の 2 つの項が線形回帰モデルに対応し、最後の項が行列分解モデルの拡張であることは明らかです。フィーチャー $i$ がアイテムを表し、フィーチャー $j$ がユーザーを表す場合、3 番目の用語はユーザーとアイテムの埋め込みの間のドット積になります。FMは高次（次数> 2）に一般化することもできることは注目に値します。それでも、数値の安定性は汎化を弱める可能性があります。 

## 効率的な最適化基準

素因数分解機を単純な方法で最適化すると、すべてのペアワイズ交互作用を計算する必要があるため、複雑さは $\mathcal{O}(kd^2)$ になります。この非効率性の問題を解決するために、FM の 3 番目の項を再編成すると、計算コストが大幅に削減され、線形時間の複雑さにつながります ($\mathcal{O}(kd)$)。ペアワイズ交互作用項の再定式化は次のようになります。 

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

この再定式化により、モデルの複雑さが大幅に減少します。さらに、スパースフィーチャの場合、全体の複雑度が 0 以外のフィーチャの数に線形になるように、0 以外の要素のみを計算する必要があります。 

FMモデルを学習するために、回帰タスクにはMSE損失、分類タスクにはクロスエントロピー損失、ランク付けタスクにはBPR損失を使用できます。確率的勾配降下法や Adam などの標準オプティマイザーは、最適化に使用できます。

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## モデルの実装以下のコードは因数分解機械を実装します。FM が線形回帰ブロックと効率的な特徴相互作用ブロックで構成されていることは明らかです。CTR 予測を分類タスクとして扱うため、シグモイド関数を最終スコアに適用します。

```{.python .input  n=2}
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## 広告データセットの読み込みオンライン広告データセットを読み込むには、前のセクションの CTR データラッパーを使用します。

```{.python .input  n=3}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## モデルのトレーニングその後、モデルをトレーニングします。既定では、学習率は 0.02、埋め込みサイズは 20 に設定されています。`Adam` オプティマイザーと `SigmoidBinaryCrossEntropyLoss` 損失はモデルトレーニングに使用されます。

```{.python .input  n=5}
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [概要

* FMは、回帰、分類、ランク付けなどのさまざまなタスクに適用できる一般的なフレームワークです。
* 特徴量の相互作用/交差は予測タスクにとって重要であり、双方向相互作用は FM で効率的にモデル化できます。

## 演習

* Avazu、MovieLens、Criteoデータセットなど、他のデータセットでFMをテストできますか？
* 埋め込みサイズを変更して、パフォーマンスへの影響を確認します。行列分解と同様のパターンが見られますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:

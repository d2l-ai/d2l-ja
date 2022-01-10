# 深層因数分解機

クリックスルー率予測タスクを成功させるには、効果的な特徴量の組み合わせを学習することが不可欠です。因数分解マシンは、特徴の相互作用を線形パラダイム (共一次相互作用など) でモデル化します。これは多くの場合、固有のフィーチャ交差構造が非常に複雑で非線形な現実世界のデータには不十分です。さらに悪いことに、実際には因数分解マシンでは二次の特徴量の交互作用が一般的に使用されます。因数分解機械によるより高度な特徴量の組み合わせをモデル化することは理論的には可能ですが、数値が不安定で計算が複雑なため、通常は採用されません。 

効果的な解決策の 1 つは、ディープニューラルネットワークの使用です。ディープニューラルネットワークは、特徴表現の学習において強力であり、高度な特徴の相互作用を学習する可能性を秘めています。そのため、ディープニューラルネットワークを因数分解マシンに統合するのは自然なことです。非線形変換層を因数分解マシンに追加すると、低次の特徴の組み合わせと高次の特徴の組み合わせの両方をモデル化できるようになります。さらに、入力からの非線形の固有構造をディープニューラルネットワークで捉えることもできます。ここでは、FMとディープニューラルネットワークを組み合わせたディープファクタライズマシン (DeepFM) :cite:`Guo.Tang.Ye.ea.2017`という代表的なモデルを紹介します。 

## モデルアーキテクチャ

DeepFMは、FMコンポーネントとディープコンポーネントで構成され、並列構造に統合されています。FM コンポーネントは、低次の特徴の相互作用をモデル化するために使用される 2 元分解マシンと同じです。深層成分は、高次の特徴の相互作用と非線形性を捕捉するために使用される MLP です。これら 2 つのコンポーネントは同じ入力/埋め込みを共有し、その出力は最終予測として合計されます。DeepFMの精神は、暗記と一般化の両方を捉えることができるWide\ & Deepアーキテクチャの精神に似ていることを指摘する価値があります。Wide & Deepモデルに対するDeepFMの利点は、特徴の組み合わせを自動的に識別することで、手作業による特徴量エンジニアリングの労力を軽減できることです。 

簡略化のため FM コンポーネントの説明は省略し、出力は $\hat{y}^{(FM)}$ と表記しています。詳細については、最後のセクションを参照してください。$\mathbf{e}_i \in \mathbb{R}^{k}$ は $i^\mathrm{th}$ 場の潜在特徴ベクトルを表すとします。ディープコンポーネントの入力は、スパースなカテゴリカル特徴入力で検索されるすべてのフィールドの密な埋め込みを連結したもので、次のように表されます。 

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

$f$ はフィールドの数です。その後、次のニューラルネットワークに入力されます。 

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

$\alpha$ は活性化関数です。$\mathbf{W}_{l}$ と $\mathbf{b}_{l}$ は $l^\mathrm{th}$ 層の重みとバイアスです。$y_{DNN}$ を予測の出力としましょう。DeepFMの最終的な予測は、FMとDNNの両方からの出力の合計です。だから我々は持っている： 

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

$\sigma$ はシグモイド関数です。DeepFMのアーキテクチャを以下に示します。！[Illustration of the DeepFM model](../img/rec-deepfm.svg) 

ディープニューラルネットワークとFMを組み合わせる方法はDeepFMだけではないことは注目に値します。特徴の相互作用 :cite:`He.Chua.2017` に非線形層を追加することもできます。

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## DeepFMの実装 DeepFMの実装はFMの実装と似ています。FM 部分は変更せず、アクティベーション関数として `relu` を持つ MLP ブロックを使用します。ドロップアウトはモデルの正則化にも使用されます。MLP のニューロンの数は `mlp_dims` ハイパーパラメーターで調整できます。

```{.python .input  n=2}
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## モデルのトレーニングと評価データ読み込みのプロセスはFMと同様です。DeepFMのMLPコンポーネントをピラミッド構造 (30-20-10) の三層高密度ネットワークに設定します。他のすべてのハイパーパラメータは FM と同じままです。

```{.python .input  n=4}
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

FMと比較して、DeepFMはより速く収束し、より良いパフォーマンスを実現します。 

## [概要

* ニューラルネットワークをFMに統合することで、複雑で高次のインタラクションをモデル化できます。
* DeepFM は、広告データセットのオリジナルの FM よりも優れています。

## 演習

* MLP の構造を変化させて、MLP がモデルのパフォーマンスに与える影響をチェックします。
* データセットをCriteoに変更し、元のFMモデルと比較します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:

# レコメンダーシステムのパーソナライズされたランキング

前のセクションでは、明示的なフィードバックのみが考慮され、モデルは観測された評価に基づいてトレーニングおよびテストされました。このような方法には 2 つのデメリットがあります。1 つは、ほとんどのフィードバックが明示的ではなく現実世界のシナリオでは暗黙的であり、明示的なフィードバックは収集にコストがかかることです。第二に、ユーザーの関心を予測する可能性のある観測されていないユーザーとアイテムのペアは完全に無視されるため、これらの方法は評価がランダムに欠落するのではなく、ユーザーの好みのために欠落している場合には不適切です。観測されていないユーザーとアイテムのペアは、実際の否定的なフィードバック (ユーザーはアイテムに興味がない) と欠損値 (ユーザーが将来アイテムとやり取りする可能性がある) が混在しています。行列因数分解と AutoRec では、観測されないペアを単純に無視します。明らかに、これらのモデルは観測されたペアと観測されていないペアを区別することができず、通常はパーソナライズされたランキングタスクには適していません。 

この目的のために、暗黙的なフィードバックからランク付けされた推奨リストを生成することを目的としたレコメンデーションモデルのクラスが人気を博しています。一般に、パーソナライズされたランキングモデルは、ポイントワイズ、ペアワイズ、またはリストワイズのアプローチで最適化できます。ポイントワイズアプローチでは、一度に 1 つの交互作用を考慮し、分類器またはリグレッサーに学習をさせて個々の選好を予測します。行列因数分解と AutoRec は点ワイズ目的関数で最適化されます。ペアワイズアプローチでは、各ユーザーの項目のペアを考慮し、そのペアの最適な順序を近似することを目指します。相対的な順序を予測することはランキングの性質を思い起こさせるため、通常、ペアワイズアプローチはランク付けタスクに適しています。リストワイズアプローチは、正規化割引累積ゲイン ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)) などのランキング測度を直接最適化するなど、項目のリスト全体の順序を近似します。ただし、リストワイズアプローチは、ポイントワイズアプローチやペアワイズアプローチよりも複雑で計算負荷が高くなります。このセクションでは、ベイジアンパーソナライズランキング損失とヒンジ損失の2つのペアワイズ目標/損失と、それぞれの実装について紹介します。 

## ベイジアン・パーソナライズド・ランキング・ロスとその実装

ベイジアンパーソナライズランキング (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` は、最大事後推定量から導出されるペアワイズパーソナライズドランキング損失です。多くの既存のレコメンデーションモデルで広く使用されています。BPR の学習データは、正と負のペア (欠損値) の両方で構成されます。これは、ユーザーが他のすべての非観測項目よりも陽性項目を好むことを前提としています。 

正式には、トレーニングデータは $(u, i, j)$ の形式のタプルで構成されます。これは、ユーザー $u$ が項目 $j$ よりも項目 $i$ を優先することを表します。事後確率を最大化することを目的としたBPRのベイズ定式化を以下に示す。 

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

$\Theta$ は任意のレコメンデーションモデルのパラメーターを表し、$>_u$ はユーザー $u$ のすべての項目について希望するパーソナライズされた合計ランキングを表します。パーソナライズされたランキングタスクの一般的な最適化基準を導き出すために、最大事後推定量を定式化できます。 

$$
\begin{aligned}
\text{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$

$D := \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ はトレーニングセットで、$I^+_u$ はユーザーが気に入ったアイテム $u$、$I$ はすべてのアイテム、$I \backslash I^+_u$ はユーザーが気に入ったアイテムを除くその他すべてのアイテムを表します。$\hat{y}_{ui}$ と $\hat{y}_{uj}$ は、$i$ および $i$ および $u$ に対するユーザー $u$ の予測スコアです。20、それぞれ。先行の $p(\Theta)$ は、平均がゼロで分散共分散行列 $\Sigma_\Theta$ をもつ正規分布です。ここで $\Sigma_\Theta = \lambda_\Theta I$ にしましょう。 

![Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg) 基本クラス `mxnet.gluon.loss.Loss` を実装し、`forward` メソッドをオーバーライドして、ベイジアンパーソナライズランキング損失を構築します。まず、Loss クラスと np モジュールをインポートします。

```{.python .input  n=5}
from mxnet import gluon, np, npx
npx.set_np()
```

BPR損失の実施は以下の通りです。

```{.python .input  n=2}
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## ヒンジ損失とその実装

ランキングのヒンジ損失は、SVMなどの分類器でよく使われるグルーオンライブラリ内で提供されている [ヒンジ損失](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) とは形が異なります。レコメンダーシステムでのランキングに使用される損失は、次の形式になります。 

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

$m$ は安全マージンの大きさです。ネガティブなアイテムをポジティブなアイテムから遠ざけることを目的としています。BPR と同様に、絶対出力ではなく正と負のサンプル間の距離を最適化することを目的としており、レコメンダーシステムに適しています。

```{.python .input  n=3}
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

これらの2つの損失は、レコメンデーションのパーソナライズされたランキングと交換可能です。 

## [概要

- レコメンダーシステムのパーソナライズされたランキングタスクには、ポイントワイズ、ペアワイズ、リストワイズの3種類のランキング損失があります。
- 2つのペアワイズロス、ベイジアンパーソナライズランキングロスとヒンジロスは同じ意味で使用できます。

## 演習

- BPRとヒンジ損失のバリエーションはありますか？
- BPRまたはヒンジ損失を使用する推奨モデルはありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:

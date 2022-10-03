```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 基本分類モデル
:label:`sec_classification`

リグレッションの場合、ゼロからの実装とフレームワーク機能を使用した簡潔な実装がかなり似ていることに気づいたかもしれません。分類についても同じことが言えます。この本の非常に多くのモデルが分類を扱っているので、特にこの設定をサポートするいくつかの機能を追加する価値があります。このセクションでは、将来のコードを簡略化するための分類モデルの基本クラスを提供します。

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

## `Classifier`クラスは

以下に `Classifier` クラスを定義します。`validation_step`では、検証バッチの損失値と分類精度の両方を報告します。`num_val_batches`バッチごとに更新を描画します。これには、検証データ全体で平均化された損失と精度を生成するという利点があります。最後のバッチに含まれる例が少ない場合、これらの平均数は正確ではありませんが、コードを単純にするためにこの小さな違いを無視します。

```{.python .input  n=5}
%%tab all
class Classifier(d2l.Module):  #@save
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

デフォルトでは、線形回帰のコンテキストで行ったように、ミニバッチで動作する確率的勾配降下オプティマイザを使用します。

```{.python .input  n=6}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input  n=7}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input  n=8}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

## 精度

予測確率分布 `y_hat` を考えると、通常、ハード予測を出力する必要がある場合は常に、予測確率が最も高いクラスを選択します。実際、多くのアプリケーションでは選択が必要です。たとえば、Gmailはメールを「プライマリ」、「ソーシャル」、「アップデート」、「フォーラム」、「スパム」に分類する必要があります。内部で確率を推定するかもしれませんが、結局のところ、クラスの中から1つを選択する必要があります。 

予測がラベルクラス `y` と一致する場合、それらは正しいです。分類精度は、正しいすべての予測の比率です。精度を直接最適化するのは難しいかもしれませんが（微分できません）、私たちが最も重視するのはパフォーマンス指標です。多くの場合、これはベンチマークの「関連量」です。そのため、ほとんどの場合、分類器をトレーニングするときに報告します。 

精度は次のように計算されます。まず、`y_hat` が行列の場合、2 番目の次元には各クラスの予測スコアが格納されていると仮定します。`argmax` を使用して、各行の最大エントリのインデックスによって予測クラスを取得します。次に [**予測されたクラスとグラウンドトゥルース`y`を要素ごとに比較します。**] 等価演算子`==`はデータ型に敏感であるため、`y_hat`のデータ型を`y`のデータ型と一致するように変換します。結果は、0 (false) と 1 (true) のエントリを含むテンソルになります。合計を取ると、正しい予測の数が得られます。

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## まとめ

分類は十分に一般的な問題であり、それ自体の便利な機能を保証する。分類で最も重要なのは、分類器の「正確さ」です。私たちはしばしば主に正確さを重視しますが、統計的および計算上の理由から、他のさまざまな目的を最適化するように分類器をトレーニングすることに注意してください。ただし、学習中にどの損失関数が最小化されたかにかかわらず、分類器の精度を経験的に評価するための便利な方法があると便利です。  

## 演習

1. $L_v$で検証損失を表し、$L_v^q$をこのセクションの損失関数の平均化によって計算されたその迅速で汚い推定とします。最後に、最後のミニバッチの損失を$l_v^b$で表します。$L_v$ を $L_v^q$、$l_v^b$、およびサンプルとミニバッチのサイズで表します。
1. 迅速で汚い推定$L_v^q$が偏りがないことを示します。つまり、$E[L_v] = E[L_v^q]$を見せてください。なぜ代わりに$L_v$を使いたいのですか？
1. $y$ を見ると $y'$ を推定した場合のペナルティが $l(y,y')$ で示され、確率 $p(y \mid x)$ が与えられるマルチクラス分類損失を考えると、$y'$ の最適選択のルールを定式化します。ヒント:$l$と$p(y \mid x)$を使用して、予想される損失を表現します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6810)
:end_tab:

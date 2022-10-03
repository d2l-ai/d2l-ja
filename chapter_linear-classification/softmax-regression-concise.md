```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# ソフトマックス回帰の簡潔な実装
:label:`sec_softmax_concise`

高レベルのディープラーニングフレームワークによって線形回帰の実装が容易になったように (:numref:`sec_linear_concise` を参照)、ここでも同様に便利です。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## モデルを定義する

:numref:`sec_linear_concise` と同様に、組み込みのレイヤーを使用して完全接続レイヤーを構築します。組み込みの `__call__` メソッドは、ネットワークを何らかの入力に適用する必要があるときはいつでも `forward` を呼び出します。

:begin_tab:`mxnet`
入力 `X` が 4 次テンソルであっても、組み込みの `Dense` 層は、第 1 軸に沿った次元を変更せずに維持することにより、自動的に `X` を 2 次テンソルに変換します。
:end_tab:

:begin_tab:`pytorch`
`Flatten` 層を使用して、第 1 軸に沿った次元を変更せずに第 4 次テンソル `X` を 2 次に変換します。
:end_tab:

:begin_tab:`tensorflow`
`Flatten` レイヤーを使用して、1 番目の軸に沿った次元を変更せずに保持して 4 次テンソル `X` を変換します。
:end_tab:

```{.python .input}
%%tab all
class SoftmaxRegression(d2l.Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(nn.Flatten(),
                                     nn.LazyLinear(num_outputs))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

## Softmax 再訪
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch`では、モデルの出力を計算し、クロスエントロピー損失を適用しました。これは数学的には完全に合理的ですが、べき乗の数値アンダーフローとオーバーフローのため、計算上は危険です。 

softmax 関数は $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ を介して確率を計算することを思い出してください。$o_k$の一部が非常に大きい、つまり非常に正の場合、$\exp(o_k)$は特定のデータ型で取得できる最大数よりも大きくなる可能性があります。これは*オーバーフロー* と呼ばれます。同様に、すべての引数が非常に負の場合、*underflow* になります。たとえば、単精度浮動小数点数は、おおよそ$10^{-38}$から$10^{38}$の範囲をカバーします。そのため、$\mathbf{o}$の最大の項が区間$[-90, 90]$の外にある場合、結果は安定しません。この問題の解決策は、すべてのエントリから $\bar{o} \stackrel{\mathrm{def}}{=} \max_k o_k$ を引くことです。 

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

構造上、$o_j - \bar{o} \leq 0$がすべての$j$であることを知っています。そのため、$q$ クラスの分類問題では、分母は区間 $[1, q]$ に含まれます。さらに、分子が$1$を超えることはないため、数値のオーバーフローを防ぎます。数値アンダーフローは、$\exp(o_j - \bar{o})$ が数値的に $0$ と評価された場合にのみ発生します。それでも、$\log \hat{y}_j$を$\log 0$として計算したいとき、道を数歩進んだときに問題が発生する可能性があります。特に、バックプロパゲーションでは、恐ろしい`NaN`（Not a Number）の結果の一部に直面する可能性があります。 

幸いなことに、指数関数を計算しているにもかかわらず、最終的には（クロスエントロピー損失を計算するときに）対数を取るつもりであるという事実によって救われます。ソフトマックスとクロスエントロピーを組み合わせることで、数値安定性の問題を完全に回避できます。私たちには次のものがあります。 

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

これにより、オーバーフローとアンダーフローの両方が回避されます。モデルによって出力確率を評価したい場合に備えて、従来のソフトマックス関数を手元に置いておきたいと思います。しかし、ソフトマックス確率を新しい損失関数に渡す代わりに、["LogsumExp trick"]（https://en.wikipedia.org/wiki/LogSumExp）のようなスマートなことをするクロスエントロピー損失関数内で、[**ロジットを渡してソフトマックスとその対数を一度に計算する**] だけです。

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

## トレーニング

次に、モデルをトレーニングします。前と同じように、784次元の特徴ベクトルに平坦化されたFashion-MNIST画像を使用します。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

以前と同様に、このアルゴリズムは、今回は以前よりも少ないコード行数ではありますが、まともな精度を達成するソリューションに収束します。 

## まとめ

高レベル API は、数値の安定性など、潜在的に危険な側面をユーザーから隠すのに非常に便利です。さらに、ユーザーはごくわずかなコード行でモデルを簡潔に設計できます。これは祝福と呪いの両方です。明らかな利点は、人生で単一のクラスの統計をとったことがないエンジニアにとっても、物事にアクセスしやすくなることです（実際、これは本の対象読者の1人です）。しかし、鋭いエッジを隠すことには代償が伴います。新しいコンポーネントや異なるコンポーネントを自分で追加することは、それを行うための筋肉の記憶がほとんどないため、阻害要因になります。さらに、フレームワークの保護パッドがすべてのコーナーケースを完全に覆うことができない場合は、物事を*修正*することがより困難になります。繰り返しますが、これは親しみやすさの欠如によるものです。 

そのため、後続する多くの実装の最低限のバージョンとエレガントなバージョンの両方を確認することを強くお勧めします。私たちは理解のしやすさを強調していますが、それでも実装は通常かなりパフォーマンスが良いです（ここでは畳み込みが大きな例外です）。私たちの意図は、フレームワークでは得られない新しいものを発明するときに、これらに基づいて構築できるようにすることです。 

## 演習

1. ディープラーニングは、FP64倍精度（ごくまれにしか使用されない）など、さまざまな数値形式を使用します。
FP32 単精度、BFLOAT16 (圧縮表現に最適)、FP16 (非常に不安定な)、TF32 (NVIDIA からの新しいフォーマット)、および INT8。結果が数値的なアンダーフローまたはオーバーフローを引き起こさない指数関数の最小および最大の引数を計算します。
1. INT8は、$1$から$255$までのゼロ以外の数字を含む非常に限定された形式です。より多くのビットを使用せずにダイナミックレンジを拡張するにはどうすればよいでしょうか？標準の乗算と加算はまだ機能しますか?
1. トレーニングのエポック数を増やします。しばらくすると検証精度が低下するのはなぜですか？どうやってこれを直せる？
1. 学習率を上げるとどうなりますか？いくつかの学習率の損失曲線を比較します。どちらがうまくいきますか？いつ？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:

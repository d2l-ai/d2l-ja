```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 体重減衰
:label:`sec_weight_decay`

オーバーフィットの問題を特徴づけたところで、最初の*正則化*手法を紹介します。より多くのトレーニングデータを収集することで、過適合をいつでも軽減できることを思い出してください。しかし、それはコストがかかり、時間がかかり、または完全に私たちの制御不能になる可能性があり、短期的には不可能になります。今のところ、私たちはリソースが許す限り多くの高品質のデータをすでに持っていると仮定し、データセットが与えられたものとして取られたとしても、自由に使えるツールに集中することができます。 

多項式回帰の例 (:numref:`subsec_polynomial-curve-fitting`) では、近似した多項式の次数を微調整することでモデルの容量を制限できることを思い出してください。実際、特徴の数を制限することは、過適合を緩和するための一般的な手法です。しかし、単に機能を捨てるだけでは、楽器が鈍すぎる可能性があります。多項式回帰の例に固執し、高次元の入力で何が起こるかを考えてみましょう。多変量データへの多項式の自然な拡張は*単項式*と呼ばれ、単に変数のべき乗の積です。単項式の次数は、べき乗の合計です。たとえば、$x_1^2 x_2$ と $x_3 x_5^2$ は、どちらも次数 3 の単項式です。 

$d$ の次数を持つ項の数は、$d$ が大きくなるにつれて急速に増加することに注意してください。$k$の変数が与えられた場合、$d$の次数の単項式（つまり、$k$のマルチチョイス$d$）は${k - 1 + d} \choose {k - 1}$になります。$2$から$3$への小さな次数の変化でも、モデルの複雑さは劇的に増大します。そのため、関数の複雑さを調整するために、よりきめ細かなツールが必要になることがよくあります。 

## 規範と体重減少

(**パラメータの数を直接操作するのではなく、
*重量の減衰*、値を制限することで動作します 
パラメータが使用できること。**) ディープラーニングサークルの外ではより一般的には $\ell_2$ 正則化と呼ばれ、ミニバッチの確率的勾配降下法によって最適化される場合、重み減衰は、パラメトリック機械学習モデルを正則化するために最も広く使用されている手法である可能性があります。この手法は、すべての関数$f$の中で、関数$f = 0$（すべての入力に値$0$を割り当てる）が何らかの形で*最も単純*であり、ゼロからのパラメーターの距離によって関数の複雑さを測定できるという基本的な直感によって動機付けられています。しかし、関数とゼロの間の距離をどれくらい正確に測定すべきでしょうか？正解は1つもありません。実際、関数解析の一部やバナッハ空間の理論を含む数学の分野全体が、そのような問題に取り組むことに専念しています。 

簡単な解釈の1つは、線形関数$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$の複雑さをその重みベクトルのあるノルム、たとえば$\| \mathbf{w} \|^2$によって測定することです。$\ell_2$ノルムと$\ell_1$ノルムを導入したことを思い出してください。これらは、:numref:`subsec_lin-algebra-norms`のより一般的な$\ell_p$ノルムの特別なケースです。小さい重みベクトルを保証する最も一般的な方法は、損失を最小にする問題に、そのノルムをペナルティ項として追加することです。したがって、私たちは当初の目標を置き換え、
*トレーニングラベルの予測損失を最小限に抑える*、
新しい目的で、
*予測損失とペナルティタームの合計を最小化する*。
ここで、重みベクトルが大きくなりすぎると、学習アルゴリズムは重みノルム $\| \mathbf{w} \|^2$ の最小化と学習エラーの最小化に焦点を当てる可能性があります。それがまさに私たちが望んでいることです。コードで説明するために、線形回帰の:numref:`sec_linear_regression`の前の例を復活させます。そこで、私たちの損失は 

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$ がフィーチャ、$y^{(i)}$ が任意のデータ例のラベル $i$、$(\mathbf{w}, b)$ がそれぞれ重みとバイアスのパラメーターであることを思い出してください。重みベクトルの大きさにペナルティを課すには、何らかの形で$\| \mathbf{w} \|^2$を損失関数に追加する必要がありますが、モデルはこの新しい加算ペナルティに対して標準損失をどのようにトレードオフする必要がありますか？実際には、検証データを使用して近似する非負のハイパーパラメータである*正則化定数* $\lambda$を使用してこのトレードオフを特徴付けます。 

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$

$\lambda = 0$ では、元の損失関数を回復します。$\lambda > 0$ については、$\| \mathbf{w} \|$ のサイズを制限しています。慣例により $2$ で割ります。二次関数の微分を取るとき、$2$ と $1/2$ は相殺され、更新の式が美しくシンプルに見えるようにします。鋭い読者は、なぜ標準ノルム（ユークリッド距離）ではなく二乗ノルムを扱うのか疑問に思うかもしれません。これは、計算の便宜のために行います。$\ell_2$ ノルムを二乗することにより、平方根を削除し、重みベクトルの各成分の二乗和を残します。これにより、ペナルティの微分を計算しやすくなります。導関数の合計は合計の微分と等しくなります。 

さらに、そもそもなぜ私たちが$\ell_2$ノルムを使用し、たとえば$\ell_1$ノルムを使用しないのかと尋ねるかもしれません。実際、他の選択肢は統計全体で有効で人気があります。$\ell_2$ 正則化線形モデルは古典的な *リッジ回帰* アルゴリズムを構成しますが、$\ell_1$ 正則化線形回帰は、統計における同様に基本的な方法であり、一般に*投げ縄回帰* として知られています。$\ell_2$ ノルムを使用する理由の 1 つは、重みベクトルの大きな成分に大きすぎるペナルティを課すことです。これにより、学習アルゴリズムは、より多くの特徴に均等に重みを配分するモデルに偏ります。実際には、これにより、単一変数の測定誤差に対してよりロバストになる可能性があります。対照的に、$\ell_1$ のペナルティは、他のウェイトをゼロにクリアすることにより、一部のフィーチャに重みを集中させるモデルにつながります。これにより、*フィーチャ選択*の効果的な方法が得られますが、これは他の理由で望ましい場合があります。たとえば、モデルが少数のフィーチャのみに依存している場合、他の (ドロップされた) フィーチャのデータを収集、保存、または送信する必要がない場合があります。  

:eqref:`eq_linreg_batch_update` で同じ表記法を使用して、$\ell_2$ 正則化回帰のミニバッチ確率勾配降下法の更新は次のとおりです。 

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

前と同様に、推定値が観測値と異なる量に基づいて$\mathbf{w}$を更新します。ただし、$\mathbf{w}$ のサイズもゼロに向かって縮小します。そのため、この方法は「体重減衰」と呼ばれることもあります。ペナルティ項のみを考慮すると、最適化アルゴリズムはトレーニングの各ステップで体重を*減衰*します。特徴の選択とは対照的に、重量減衰は機能の複雑さを調整するための連続的なメカニズムを提供します。$\lambda$ の値が小さいほど制約の少ない $\mathbf{w}$ に対応し、$\lambda$ の値が大きいほど $\mathbf{w}$ の制約が大きくなります。対応するバイアスペナルティ$b^2$を含めるかどうかは、実装によって異なり、ニューラルネットワークのレイヤーによって異なる場合があります。多くの場合、バイアス項を正則化しません。また、$\ell_2$ の正則化は、他の最適化アルゴリズムの重み減衰と同等ではないかもしれませんが、重みのサイズを縮小して正則化するという考え方は依然として当てはまります。 

## 高次元線形回帰

簡単な合成例を通して、体重減衰の利点を説明できます。

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

まず、[**前と同じようにデータを生成する**]: 

(**$$y = 0.05 +\ sum_ {i = 1} ^d 0.01 x_i +\ イプシロン\ テキスト {どこ}\ イプシロン\ sim\ mathcal {N} (0, 0.01^2) .$$**) 

この合成データセットでは、ラベルは入力の基礎となる線形関数によって与えられ、ゼロ平均、標準偏差 0.01 のガウスノイズによって破損しています。説明のために、問題の次元を$d = 200$に増やし、20例しかない小さなトレーニングセットで作業することで、オーバーフィットの影響を顕著にすることができます。

```{.python .input  n=5}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## ゼロからの実装

それでは、体重減衰をゼロから実装してみましょう。ミニバッチの確率的勾配降下法はオプティマイザなので、元の損失関数に二乗した$\ell_2$ペナルティを追加するだけで済みます。 

### (**$\ell_2$ ノルムペナルティの定義**)

おそらく、このペナルティを実装する最も便利な方法は、すべての項を二乗して合計することです。

```{.python .input  n=6}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### モデルを定義する

最終的なモデルでは、線形回帰と二乗損失は :numref:`sec_linear_scratch` 以降変化していないため、`d2l.LinearRegressionScratch` のサブクラスを定義します。ここでの唯一の変更点は、損失にペナルティ期間が含まれるようになったことです。

```{.python .input  n=7}
%%tab all
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.lambd * l2_penalty(self.w)
```

次のコードは、20個の例を含むトレーニングセットのモデルを適合させ、100個の例を含む検証セットで評価します。

```{.python .input  n=8}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
```

### [**正規化なしのトレーニング**]

このコードを `lambd = 0` で実行し、重量の減衰を無効にします。オーバーフィットがひどく、学習エラーは減少しますが、検証エラーは減少しないことに注意してください。これは教科書ではオーバーフィットのケースです。

```{.python .input  n=9}
%%tab all
train_scratch(0)
```

### [**重量減衰を使用する**]

以下では、かなりの重量減衰で走ります。学習誤差は増加するが、検証誤差は減少することに注意してください。これは正則化から期待される効果です。

```{.python .input  n=10}
%%tab all
train_scratch(3)
```

## [**簡潔な実装**]

重み減衰はニューラルネットワークの最適化に遍在するため、ディープラーニングフレームワークは特に便利で、重み減衰を最適化アルゴリズム自体に統合して、損失関数と組み合わせて簡単に使用できます。さらに、この統合は計算上の利点をもたらし、追加の計算オーバーヘッドなしに実装トリックがアルゴリズムに重み付けを加えることを可能にします。更新の重み減衰部分は各パラメーターの現在の値にのみ依存するため、オプティマイザーはいずれにせよ各パラメーターに一度タッチする必要があります。

:begin_tab:`mxnet`
次のコードでは、`Trainer`をインスタンス化するときに、`wd`を介して直接重み減衰ハイパーパラメータを指定します。既定では、Gluon は重みとバイアスの両方を同時に減衰させます。モデルパラメーターを更新すると、ハイパーパラメーター `wd` に `wd_mult` が乗算されることに注意してください。したがって、`wd_mult`をゼロに設定すると、バイアスパラメータ$b$は減衰しません。
:end_tab:

:begin_tab:`pytorch`
次のコードでは、オプティマイザをインスタンス化するときに `weight_decay` を介して直接重み減衰ハイパーパラメータを指定します。デフォルトでは、PyTorch はウェイトとバイアスの両方を同時に減衰させます。ここでは、重みに `weight_decay` を設定しただけなので、バイアスパラメータ $b$ は減衰しません。
:end_tab:

:begin_tab:`tensorflow`
次のコードでは、重み減衰ハイパーパラメーター `wd` を使用して $\ell_2$ 正則化器を作成し、`kernel_regularizer` 引数によって層の重みに適用します。
:end_tab:

```{.python .input  n=11}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input  n=12}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), 
                               lr=self.lr, weight_decay=self.wd)
```

```{.python .input  n=13}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

[**このプロットは、ゼロからの重量減衰を実装したときと似ています**]。しかし、このバージョンはより速く実行され、実装が簡単です。より大きな問題に対処し、この作業がより日常的になるにつれて、利点はより顕著になります。

```{.python .input  n=14}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

これまでは、単純な一次関数を構成するものの概念について触れただけです。さらに、単純な非線形関数を構成するものは、さらに複雑な問題になる可能性があります。たとえば、[カーネルヒルベルト空間 (RKHS) の再現](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) を使用すると、非線形コンテキストで線形関数に導入されたツールを適用できます。残念ながら、RKHSベースのアルゴリズムは、大規模で高次元のデータにはあまりスケーリングできない傾向があります。この本では、重みの減衰が深いネットワークのすべての層に適用されるという共通のヒューリスティックをしばしば採用します。 

## まとめ

* 正則化は、過適合に対処するための一般的な方法です。従来の正則化手法では、学習したモデルの複雑さを軽減するために (学習時に) 損失関数にペナルティ項を追加します。
* モデルをシンプルに保つための特別な選択肢の 1 つは、$\ell_2$ ペナルティを使用することです。これにより、ミニバッチ確率的勾配降下アルゴリズムの更新ステップで重みが減衰します。
* 重み減衰機能は、ディープラーニングフレームワークのオプティマイザーで提供されます。
* パラメーターのセットが異なれば、同じトレーニングループ内で異なる更新動作を持つことができます。

## 演習

1. このセクションの推定問題で $\lambda$ の値を試します。学習と検証の精度を$\lambda$の関数としてプロットします。あなたは何を観察していますか？
1. 検証セットを使用して、$\lambda$ の最適値を見つけます。本当に最適値なのですか？これは問題なの？
1. $\|\mathbf{w}\|^2$の代わりに$\sum_i |w_i|$を選択したペナルティ（$\ell_1$正則化）として使用した場合、更新方程式はどのようになりますか？
1. 私たちは$\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$を知っています。同様の行列方程式が見つかりますか (:numref:`subsec_lin-algebra-norms`のフロベニウスノルムを参照)。
1. 学習誤差と汎化誤差の関係を確認します。体重減少、トレーニングの増加、適切な複雑さのモデルの使用に加えて、過適合に対処するために他にどのような方法を考えられますか？
1. ベイズ統計では、$P(w \mid x) \propto P(x \mid w) P(w)$を介して事後に到達する前の確率と尤度の積を使用します。$P(w)$を正規化でどのように識別できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:

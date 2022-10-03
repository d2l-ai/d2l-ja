```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# ゼロからの線形回帰の実装
:label:`sec_linear_scratch`

これで、完全に機能する線形回帰の実装に取り組む準備が整いました。このセクションでは、(** (i) モデル、(ii) 損失関数、(iii) ミニバッチ確率的勾配降下オプティマイザ、(iv) これらすべてをまとめるトレーニング関数を含む、メソッド全体をゼロから実装します。**) 最後に、以下から合成データジェネレータを実行します。:numref:`sec_synthetic-regression-data`、結果のデータセットにモデルを適用します。最新のディープラーニングフレームワークはこの作業のほとんどすべてを自動化できますが、何をしているのかを確実に把握するには、ゼロから実装することが唯一の方法です。さらに、モデルをカスタマイズしたり、独自のレイヤーや損失関数を定義したりするときには、内部で物事がどのように機能するかを理解することが役立ちます。このセクションでは、テンソルと自動微分のみを使用します。後ほど、以下の構造を維持しながら、ディープラーニングフレームワークの機能を活用して、より簡潔な実装を紹介します。

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

## モデルを定義する

[**モデルのパラメーターの最適化を始める前に**] minibatch SGD (**そもそもいくつかのパラメーターが必要です。**) 以下では、平均 0、標準偏差 0.01 の正規分布から乱数を抽出し、重みを初期化します。マジックナンバー0.01は実際にはうまく機能することが多いですが、引数`sigma`で別の値を指定できます。さらに、バイアスを0に設定します。オブジェクト指向設計では、`d2l.Module` (:numref:`oo-design-models` で導入) のサブクラスの `__init__` メソッドにコードを追加することに注意してください。

```{.python .input  n=5}
%%tab all
class LinearRegressionScratch(d2l.Module):  #@save
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

次に、[**入力とパラメーターを出力に関連付けてモデルを定義します**]。線形モデルでは、入力フィーチャ $\mathbf{X}$ とモデルの重み $\mathbf{w}$ の行列ベクトル積を取得し、オフセット $b$ を各例に追加します。$\mathbf{Xw}$ はベクトル、$b$ はスカラーです。ブロードキャストメカニズム (:numref:`subsec_broadcasting` を参照) により、ベクトルとスカラーを追加すると、スカラーはベクトルの各コンポーネントに追加されます。結果の `forward` 関数は、`add_to_class` (:numref:`oo-design-utilities` で導入) を介して `LinearRegressionScratch` クラスのメソッドとして登録されます。

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    """The linear regression model."""
    return d2l.matmul(X, self.w) + self.b
```

## 損失関数の定義

[**モデルを更新するには損失関数の勾配を取る必要があるため、**](**損失関数を最初に定義します**) ここでは :eqref:`eq_mse` の二乗損失関数を使用します。実装では、真の値`y`を予測値の形状`y_hat`に変換する必要があります。次の関数によって返される結果も、`y_hat` と同じ形状になります。また、ミニバッチのすべての例の平均損失値を返します。

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## 最適化アルゴリズムの定義

:numref:`sec_linear_regression`で説明したように、線形回帰には閉形式の解があります。ただし、ここでの目標は、より一般的なニューラルネットワークを学習させる方法を説明することであり、そのためには、ミニバッチ SGD の使い方を教える必要があります。そこで、この機会にSGDの最初の実例を紹介します。各ステップで、データセットからランダムに抽出されたミニバッチを使用して、パラメータに対する損失の勾配を推定します。次に、損失を減らす可能性のある方向にパラメータを更新します。 

次のコードは、一連のパラメーター、学習率 `lr` を指定して、更新を適用します。損失はミニバッチの平均として計算されるため、バッチサイズに対して学習率を調整する必要はありません。後の章では、分散型大規模学習で発生する非常に大きなミニバッチの学習率をどのように調整すべきかを調査します。今のところ、この依存関係は無視できます。

:begin_tab:`mxnet`
`d2l.HyperParameters` (:numref:`oo-design-utilities` で導入) のサブクラスである `SGD` クラスを、組み込みの SGD オプティマイザと同様の API を持つように定義します。`step` メソッドのパラメーターを更新します。無視できる `batch_size` 引数を受け入れます。
:end_tab:

:begin_tab:`pytorch`
`d2l.HyperParameters` (:numref:`oo-design-utilities` で導入) のサブクラスである `SGD` クラスを、組み込みの SGD オプティマイザと同様の API を持つように定義します。`step` メソッドのパラメーターを更新します。`zero_grad` メソッドは、すべてのグラデーションを 0 に設定します。これは、バックプロパゲーションステップの前に実行する必要があります。
:end_tab:

:begin_tab:`tensorflow`
`SGD` クラスは `d2l.HyperParameters` (:numref:`oo-design-utilities` で導入) のサブクラスであり、組み込みの SGD オプティマイザと同様の API を持つように定義しています。`apply_gradients` メソッドのパラメーターを更新します。パラメータとグラデーションのペアのリストを受け入れます。
:end_tab:

```{.python .input  n=8}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad
    
    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=9}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()
    
    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

次に、`SGD` クラスのインスタンスを返す `configure_optimizers` メソッドを定義します。

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow'):
        return SGD(self.lr)
```

## トレーニング

これで、すべての部分 (パラメーター、損失関数、モデル、オプティマイザー) が揃ったので、[**メイントレーニングループを実装する**] 準備ができました。この本で取り上げている他のすべてのディープラーニングモデルにも同様のトレーニングループを使用するため、このコードをよく理解することが重要です。各*epoch* では、トレーニングデータセット全体を反復処理し、すべての例を 1 回通過します (例の数がバッチサイズで割り切れると仮定)。各反復で、トレーニング例のミニバッチを取得し、モデルの`training_step`メソッドを使用してその損失を計算します。次に、各パラメータに関する勾配を計算します。最後に、最適化アルゴリズムを呼び出してモデルパラメーターを更新します。要約すると、次のループを実行します。 

* パラメータを初期化する $(\mathbf{w}, b)$
* 完了するまで繰り返します
    * グラデーションの計算 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新パラメータ $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

: numref: ``sec_synthetic-regression-data``で生成した合成回帰データセットは検証データセットを提供しないことを思い出してください。ただし、ほとんどの場合、検証データセットを使用してモデルの品質を測定します。ここでは、モデルのパフォーマンスを測定するために、各エポックで検証データローダーを 1 回渡します。オブジェクト指向設計に従って、`prepare_batch`および`fit_epoch`関数は、`d2l.Trainer`クラス（:numref:`oo-design-training`で導入された）のメソッドとして登録されます。

```{.python .input  n=11}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=13}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=14}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

モデルをトレーニングする準備はほぼできていますが、まずトレーニングするデータが必要です。ここでは `SyntheticRegressionData` クラスを使用し、いくつかのグラウンドトゥルースパラメータを渡します。次に、学習率 `lr=0.03` でモデルをトレーニングし、`max_epochs=3` を設定します。一般に、エポック数と学習率の両方がハイパーパラメータであることに注意してください。一般に、ハイパーパラメータの設定は難しく、通常、3方向スプリットを使用します。1つはトレーニング用、もう1つはハイパーパラメータ選択用、3つ目は最終評価用です。これらの詳細は今のところ省略しますが、後で修正します。

```{.python .input  n=15}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

私たちはデータセットを自分で合成したので、真のパラメータが何であるかを正確に知っています。したがって、トレーニングループを通じて [**真のパラメータと学習したパラメータを比較することにより、トレーニングの成功を評価する**] ことができます。確かに、彼らはお互いに非常に近いことが分かります。

```{.python .input  n=16}
%%tab all
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

グラウンドトゥルースパラメータを正確に回復する能力を当然のことと考えてはいけません。一般に、ディープモデルでは、パラメータに対する独自のソリューションは存在せず、線形モデルであっても、他のフィーチャに線形に依存するフィーチャがない場合にのみパラメータを正確に回復できます。しかし、機械学習では、真の基礎となるパラメーターを回復することにはあまり関心がなく、高精度の予測につながるパラメーターに関心があることがよくあります。:cite:`Vapnik.1992`。幸いなことに、困難な最適化問題であっても、確率的勾配降下法は多くの場合、非常に優れた解を見つけることができます。これは、深いネットワークでは、高精度の予測につながるパラメーターの構成が多数存在するためです。 

## まとめ

このセクションでは、完全に機能するニューラルネットワークモデルとトレーニングループを実装することにより、ディープラーニングシステムの設計に向けて重要な一歩を踏み出しました。このプロセスでは、データローダー、モデル、損失関数、最適化手順、および視覚化および監視ツールを構築しました。これは、モデルのトレーニングに関連するすべてのコンポーネントを含む Python オブジェクトを作成することで実現しました。これはまだプロ級の実装ではありませんが、完全に機能しており、このようなコードはすでに小さな問題を迅速に解決するのに役立ちます。次のセクションでは、これを*より簡潔に*（定型コードを避ける）と*より効率的に*（GPUを最大限に活用する）の両方を行う方法について説明します。 

## 演習

1. 重みをゼロに初期化するとどうなるでしょうか。アルゴリズムはまだ機能しますか？$0.01$ではなく分散$1,000$でパラメータを初期化した場合はどうなりますか？
1. [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)が、電圧と電流を関連付ける抵抗器のモデルを考え出そうとしているとします。自動微分を使用してモデルのパラメーターを学習できますか？
1. [プランクの法則](https://en.wikipedia.org/wiki/Planck%27s_law) を使用して、スペクトルエネルギー密度を使用して物体の温度を決定できますか？参考までに、黒体から放射される放射線のスペクトル密度$B$は$B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$です。ここで、$\lambda$は波長、$T$は温度、$c$は光の速度、$h$はプランクの量子、$k$はボルツマン定数です。さまざまな波長 $\lambda$ のエネルギーを測定し、スペクトル密度曲線をプランクの法則に適合させる必要があります。
1. 損失の二次導関数を計算する場合に遭遇する可能性のある問題は何ですか？どうやって直すの？
1. `loss` 関数に `reshape` メソッドが必要なのはなぜですか?
1. さまざまな学習率を使用して実験し、損失関数の値がどれだけ早く低下するかを調べます。トレーニングのエポック数を増やすことでエラーを減らすことはできますか？
1. 例の数をバッチサイズで割ることができない場合、エポックの終わりに`data_iter`はどうなりますか？
1. 絶対値損失 `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()` など、別の損失関数を実装してみてください。
    1. 通常のデータに何が起こるかを確認します。
    1. $y_5 = 10,000$ など、$\mathbf{y}$ の一部のエントリをアクティブに摂動させる場合は、動作に違いがあるかどうかを確認します。
    1. 二乗損失と絶対値損失の最良の側面を組み合わせる安価なソリューションを考えられますか？ヒント:どうしたら本当に大きなグラデーション値を避けることができますか?
1. データセットを再シャッフルする必要があるのはなぜですか？そうでなければ、悪意のあるデータセットが最適化アルゴリズムを破るケースを設計できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

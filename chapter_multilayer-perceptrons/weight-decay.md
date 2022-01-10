# 体重減衰
:label:`sec_weight_decay`

オーバーフィットの問題を特徴づけたので、モデルを正則化するための標準的な手法をいくつか紹介します。外出してより多くのトレーニングデータを収集することで、常に過適合を緩和できることを思い出してください。これはコストがかかり、時間がかかったり、完全に制御不能になったりする可能性があり、短期的には不可能です。今のところ、リソースが許す限り高品質のデータがすでに存在し、正則化手法に焦点を当てていると想定できます。 

多項式回帰の例 (:numref:`sec_model_selection`) では、近似多項式の次数を微調整するだけでモデルの容量を制限できることを思い出してください。実際、特徴量の数を制限することは、過剰適合を緩和するための一般的な手法です。しかし、単に機能を捨てるだけでは、仕事にはあまりにも鈍い楽器になる可能性があります。多項式回帰の例にこだわり、高次元の入力で何が起こるかを考えてみましょう。多項式の多変量データへの自然拡張は*単項式* と呼ばれ、単に変数のべき乗の積です。単項式の次数は累乗の和です。たとえば、$x_1^2 x_2$ と $x_3 x_5^2$ はどちらも次数 3 の単項式です。 

$d$ の次数を持つ項の数は $d$ が大きくなるにつれて急激に増加することに注意してください。$k$ 個の変数が与えられた場合、$d$ (つまり $k$ マルチチョース $d$) の単項式の数は ${k - 1 + d} \choose {k - 1}$ になります。$2$ から $3$ へのわずかな次数の変化でも、モデルの複雑さが大幅に増します。したがって、関数の複雑さを調整するために、よりきめ細かいツールが必要になることがよくあります。 

## 規範と体重減少

:numref:`subsec_lin-algebra-norms` のより一般的な $L_p$ ノルムの特殊なケースである $L_2$ ノルムと $L_1$ ノルムの両方について説明しました。(***Weight decay* (一般に $L_2$ 正則化と呼ばれる) は、パラメトリック機械学習モデルの正則化に最も広く使用されている手法です。**) この手法は、すべての関数の中で $f$ 関数 $f = 0$ (すべての入力に値 $0$ を代入する) という基本的な直感に基づいています。ある意味では*最も単純*であり、ゼロからの距離で関数の複雑さを測定できるということです。しかし、関数とゼロの間の距離をどの程度正確に測定すべきでしょうか？正解は一つもありません。実際、関数解析の一部やバナッハ空間の理論など、数学の全分野がこの問題に答えることに専念しています。 

単純な解釈の 1 つとして、線形関数 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ の実数/複素数を、その重みベクトルのノルム ($\| \mathbf{w} \|^2$ など) で測定すると考えられます。重みベクトルを小さくするための最も一般的な方法は、そのノルムをペナルティ項として損失を最小化する問題に加えることです。したがって、私たちは当初の目標を置き換え、
*トレーニングラベルの予測損失を最小化*、
新しい目標をもって、
*予測損失とペナルティ項*の合計を最小化する。
ここで、重みベクトルが大きくなりすぎると、学習アルゴリズムは重みノルム $\| \mathbf{w} \|^2$ の最小化と学習誤差の最小化に重点を置く可能性があります。それがまさに私たちの望みです。コードで説明するために、前の例の :numref:`sec_linear_regression` の線形回帰を復活させましょう。そこで、私たちの損失はによって与えられました 

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$ はフィーチャ、$y^{(i)}$ はすべてのデータ例のラベル $i$、$(\mathbf{w}, b)$ はそれぞれ重みとバイアスのパラメーターであることを思い出してください。重みベクトルのサイズにペナルティを課すには、$\| \mathbf{w} \|^2$ を何らかの形で損失関数に加算しなければなりませんが、モデルはこの新しい加法ペナルティに対して標準損失とどのようにトレードオフすべきでしょうか？実際には、検証データを使用して近似する非負のハイパーパラメーターである*正則化定数* $\lambda$ を使用して、このトレードオフを特徴付けます。 

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

$\lambda = 0$ では、元の損失関数を回復します。$\lambda > 0$ では、$\| \mathbf{w} \|$ のサイズを制限しています。慣例により $2$ で割ります。二次関数の導関数を取るとき、$2$ と $1/2$ は相殺され、更新の式は見栄えがよくシンプルになります。賢明な読者は、なぜ標準ノルム（ユークリッド距離）ではなく二乗ノルムを使って作業するのか疑問に思うかもしれません。これは計算上の利便性のために行います。$L_2$ ノルムを二乗することで、重みベクトルの各成分の二乗和を残して、平方根を削除します。これにより、ペナルティの微分を計算しやすくなります。微分の和は和の導関数と等しくなります。 

さらに、そもそもなぜ$L_2$ノルムを使用し、たとえば$L_1$ノルムを使用しないのかと尋ねるかもしれません。実際、他の選択肢は統計全体で有効で人気があります。$L_2$ 正則化線形モデルは従来の*リッジ回帰* アルゴリズムを構成しますが、$L_1$ 正化線形回帰は統計学でも同様に基本的なモデルであり、一般に*LASSO 回帰* として知られています。 

$L_2$ ノルムを使用する理由の 1 つは、重みベクトルの大きな成分にアウトサイズペナルティが課されるためです。これにより、学習アルゴリズムは、より多くの特徴量にわたって重みを均等に分散するモデルに偏ります。実際には、これによって 1 つの変数の測定誤差に対してロバスト性が高まる可能性があります。一方、$L_1$ のペナルティでは、他の重みをゼロにすることで、モデルの重みを小さな特徴量に集中させることになります。これは*特徴選択* と呼ばれ、他の理由から望ましい場合もあります。 

:eqref:`eq_linreg_batch_update` で同じ表記法を使用すると、$L_2$ 正則化回帰のミニバッチ確率的勾配降下法の更新は次のようになります。 

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

以前と同様に、推定値が観測値と異なる量に基づいて $\mathbf{w}$ を更新します。ただし、$\mathbf{w}$ のサイズもゼロに縮小します。そのため、この方法は「ウェイト減衰」と呼ばれることもあります。ペナルティ項のみを考えると、最適化アルゴリズムはトレーニングの各ステップでウェイトを*減衰* します。特徴量の選択とは対照的に、重みの減衰は関数の複雑さを調整するための連続的なメカニズムを提供します。$\lambda$ の値が小さいほど制約が少ない $\mathbf{w}$ に対応し、$\lambda$ の値が大きいほど $\mathbf{w}$ の制約が大きくなります。 

対応するバイアスペナルティ $b^2$ を含めるかどうかは、実装によって異なり、ニューラルネットワークのレイヤーによって異なる場合があります。多くの場合、ネットワークの出力層のバイアス項は正則化されません。 

## 高次元線形回帰

簡単な合成例を通して、体重減少の利点を説明できます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

まず、[**以前と同じようにデータを生成する**] 

(**$$y = 0.05 +\ sum_ {i = 1} ^d 0.01 x_i +\ イプシロン\ text {どこ}\ イプシロン\ sim\ mathcal {N} (0, 0.01^2) .$$**) 

ラベルを入力の線形関数として選択し、平均がゼロで標準偏差が 0.01 のガウスノイズによって破損します。過適合の影響を顕著にするために、問題の次元を $d = 200$ に増やし、20 個の例のみを含む小さなトレーニングセットで作業できます。

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## ゼロからの実装

以下では、$L_2$ の二乗ペナルティを元のターゲット関数に追加するだけで、重みの減衰をゼロから実装します。 

### [**モデルパラメーターの初期化**]

まず、モデルパラメーターをランダムに初期化する関数を定義します。

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### (** $L_2$ ノルムペナルティの定義**)

おそらく、このペナルティを実装する最も便利な方法は、すべての項を二乗して合計することです。

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### [**トレーニングループの定義**]

次のコードは、モデルをトレーニングセットにあてはめ、テストセットで評価します。線形ネットワークと二乗損失は :numref:`chap_linear` 以降変更されていないため、`d2l.linreg` と `d2l.squared_loss` を使用してインポートします。ここでの唯一の変更点は、損失にペナルティ期間が含まれるようになったことです。

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### [**正規化なしのトレーニング**]

このコードを `lambd = 0` で実行し、重みの減衰を無効にします。過適合がひどく、学習誤差は減少するが、テスト誤差（過適合の教科書の場合）は減少しないことに注意してください。

```{.python .input}
#@tab all
train(lambd=0)
```

### [**ウェイトディケイの使用**]

以下では、かなりの体重減少を伴って走ります。学習誤差は増加するが、検定誤差は減少することに注意してください。これは正則化から期待される効果です。

```{.python .input}
#@tab all
train(lambd=3)
```

## [**簡潔な実装**]

重み減衰はニューラルネットワークの最適化では至る所に存在するため、ディープラーニングフレームワークは特に便利で、重み減衰を最適化アルゴリズム自体に統合して、あらゆる損失関数と組み合わせて簡単に使用できます。さらに、この積分は計算上の利点をもたらし、追加の計算オーバーヘッドなしにアルゴリズムに重みの減衰を追加するための実装トリックが可能になります。更新のウェイト減衰部分は各パラメーターの現在の値にのみ依存するため、オプティマイザーは各パラメーターに 1 回タッチする必要があります。

:begin_tab:`mxnet`
次のコードでは、`Trainer` をインスタンス化するときに `wd` で直接ウェイト減衰ハイパーパラメータを指定します。デフォルトでは、Gluon はウェイトとバイアスの両方を同時に減衰させます。モデルパラメーターの更新時に、ハイパーパラメーター `wd` に `wd_mult` が乗算されることに注意してください。したがって、`wd_mult` を 0 に設定すると、バイアスパラメータ $b$ は減衰しません。
:end_tab:

:begin_tab:`pytorch`
次のコードでは、オプティマイザーをインスタンス化するときに `weight_decay` を使用して weight decay ハイパーパラメーターを直接指定します。デフォルトでは、PyTorch はウェイトとバイアスの両方を同時に減衰させます。ここではウェイトに `weight_decay` のみを設定しているので、バイアスパラメータ $b$ は減衰しません。
:end_tab:

:begin_tab:`tensorflow`
次のコードでは、重み減衰ハイパーパラメーター `wd` をもつ $L_2$ 正則化器を作成し、`kernel_regularizer` 引数によって層に適用します。
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

[**プロットは、ゼロからの重量減衰を実装したときのプロットと同じように見えます**]ただし、実行速度がかなり速く、実装が簡単なため、大きな問題ではより顕著になります。

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

ここまでは、単純な一次関数を構成するものの概念を1つだけ取り上げました。さらに、単純な非線形関数を構成するものは、さらに複雑な問題になる可能性があります。例えば [カーネルヒルベルト空間 (RKHS) を再現](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) を使うと、線形関数に導入されたツールを非線形の文脈で適用することができます。残念ながら、RKHS ベースのアルゴリズムは、大規模で高次元のデータにはあまりスケーリングされない傾向があります。本書では、ディープネットワークのすべてのレイヤにウェイト減衰を適用するという単純なヒューリスティックをデフォルトとします。 

## [概要

* 正則化は、過適合に対処するための一般的な方法です。学習セットの損失関数にペナルティ項を追加して、学習したモデルの複雑さを軽減します。
* モデルを単純に保つための特別な選択肢の 1 つは、$L_2$ ペナルティを使用した重量減衰です。これにより、学習アルゴリズムの更新ステップで重みが減衰します。
* 重み減衰機能は、ディープラーニングフレームワークのオプティマイザーで提供されます。
* 同じトレーニングループ内で、パラメーターのセットが異なると、更新動作が異なる場合があります。

## 演習

1. このセクションの推定問題で $\lambda$ の値を試してみてください。学習とテストの精度を $\lambda$ の関数としてプロットします。あなたは何を観察していますか？
1. 検証セットを使用して $\lambda$ の最適値を求めます。本当に最適値なのでしょうか？これは問題なの？
1. $\|\mathbf{w}\|^2$ の代わりに $\sum_i |w_i|$ をペナルティ ($L_1$ 正則化) として使用した場合、更新方程式はどのようになるでしょうか。
1. 私たちは$\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ということを知っています。行列についても同様の方程式を見つけることができますか (:numref:`subsec_lin-algebra-norms` のフロベニウスノルムを参照)。
1. 学習誤差と汎化誤差の関係を確認します。体重減少、トレーニングの増加、適切な複雑さのモデルの使用に加えて、オーバーフィットに対処するために他にどのような方法が考えられますか？
1. ベイズ統計では、$P(w \mid x) \propto P(x \mid w) P(w)$ を介して事後に到達する事前確率と尤度の積を使用します。正則化で $P(w)$ をどのように識別できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:

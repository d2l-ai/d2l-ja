# シーケンスモデル
:label:`sec_sequence`

Netflixで映画を見ていると想像してみてください。優れたNetflixユーザーとして、各映画を宗教的に評価することにしました。結局のところ、良い映画は良い映画であり、あなたはそれらの映画をもっと見たいですよね？結局のところ、物事はそれほど単純ではありません。映画に対する人々の意見は、時間の経過とともにかなり大きく変化する可能性があります。実際、心理学者はいくつかの効果の名前さえ持っています。 

* 他人の意見に基づいて、*アンカー*があります。たとえば、オスカー賞の後、同じ映画であるにもかかわらず、対応する映画の評価が上がります。この効果は、賞が忘れられるまで数か月間持続します。この効果により、レーティングが半ポイント以上上昇することが示されています。
:cite:`Wu.Ahmed.Beutel.ea.2017`。
* *快楽的適応*があります。これは、人間が改善された、または悪化した状況をニューノーマルとして受け入れるために迅速に適応するものです。たとえば、良い映画をたくさん見た後、次の映画が同等かそれ以上であるという期待は高くなります。したがって、平均的な映画でさえ、多くの素晴らしい映画が視聴された後は悪いと見なされる可能性があります。
* *季節性*があります。8月にサンタクロースの映画を見るのが好きな視聴者はほとんどいません。
* 場合によっては、制作中の監督や俳優の不正行為のために映画が不人気になります。
* 彼らはほとんどコミカルに悪かったので、いくつかの映画はカルト映画になります。*宇宙*のプラン9と*トロール2*は、この理由で高い評価を得た。

要するに、映画の評価は静止しているわけではありません。したがって、時系列ダイナミクスを使用することで、映画の推奨がより正確になりました :cite:`Koren.2009`。もちろん、シーケンスデータは映画のレーティングだけではありません。以下に、さらに図を示します。 

* 多くのユーザーは、アプリを開くときに非常に特殊な行動をとります。たとえば、ソーシャルメディアアプリは、放課後、学生にはるかに人気があります。株式市場の取引アプリは、市場が開いているときにより一般的に使用されます。
* 明日の株価を予測するのは、昨日逃した株価の空白を埋めるよりもずっと難しい。どちらもただ一つの数字を見積もるだけなのに。結局のところ、先見の明は後知恵よりもはるかに困難です。統計学では、前者 (既知の観測値を超える予測) は*外挿* と呼ばれ、後者 (既存の観測値間の推定) は*補間* と呼ばれます。
* 音楽、音声、テキスト、ビデオはすべてシーケンシャルです。それらを順列させるとしたら、ほとんど意味がありません。見出し*犬が人を噛む*は、単語が同じであっても、*man bites dog*よりもはるかに驚くことではありません。
* 地震は強く相関しています。つまり、大地震の後、いくつかの小さな余震が発生する可能性が非常に高く、強い地震がない場合よりもはるかに多くなります。実際、地震は時空間的に相関しています。つまり、余震は通常、短期間で近接して発生します。
* ツイッターの戦い、ダンスのパターン、討論に見られるように、人間は連続した性質で互いに相互作用します。

## 統計ツール

シーケンスデータを扱うには、統計ツールと新しいディープニューラルネットワークアーキテクチャが必要です。単純化するために、:numref:`fig_ftse100`に示した株価（FTSE100指数）を例に挙げています。 

![FTSE 100 index over about 30 years.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

価格を$x_t$で表してみましょう。つまり、*タイムステップ* $t \in \mathbb{Z}^+$で価格$x_t$が観測されます。このテキストのシーケンスでは、$t$ は通常は離散で、整数またはそのサブセットによって変化することに注意してください。$t$日に株式市場で好調にやりたいトレーダーが、$x_t$を次のように予測したとします。 

$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1).$$

### 自己回帰モデル

これを実現するために、トレーダーは :numref:`sec_linear_concise` でトレーニングしたような回帰モデルを使用できます。大きな問題が 1 つだけあります。$t$ によって入力の数 $x_{t-1}, \ldots, x_1$ が変わるということです。つまり、遭遇するデータ量に応じて数が増え、これを計算上扱いやすくするためには近似が必要になります。この章で後述する内容の多くは、$P(x_t \mid x_{t-1}, \ldots, x_1)$ を効率的に推定する方法を中心に展開します。一言で言えば、それは次の2つの戦略に要約されます。 

まず、潜在的にかなり長いシーケンス $x_{t-1}, \ldots, x_1$ は実際には必要ないと仮定します。この場合、長さが $\tau$ のタイムスパンで満足し、$x_{t-1}, \ldots, x_{t-\tau}$ の観測値のみを使用する可能性があります。直接的なメリットは、少なくとも $t > \tau$ では引数の数が常に同じになることです。これにより、上に示したようにディープネットワークに学習させることができます。このようなモデルは、文字通り自分自身で回帰を実行するため、*自己回帰モデル*と呼ばれます。 

:numref:`fig_sequence-model` に示されている 2 つ目の戦略は、過去の観測値のサマリー $h_t$ を保持すると同時に、予測 $\hat{x}_t$ に加えて $h_t$ を更新することです。これにより、$\hat{x}_t = P(x_t \mid h_{t})$ で $x_t$ を推定するモデルが作成され、さらに $h_t = g(h_{t-1}, x_{t-1})$ という形式が更新されます。$h_t$ は観測されないため、これらのモデルは*潜在自己回帰モデル* とも呼ばれます。 

![A latent autoregressive model.](../img/sequence-model.svg)
:label:`fig_sequence-model`

どちらの場合も、トレーニングデータをどのように生成するかという明白な疑問が生じます。通常、履歴観測値を使用して、現在までの観測値から次の観測値を予測します。明らかに、私たちは時間が止まるとは思わない。ただし、$x_t$ の特定の値は変化する可能性がありますが、少なくともシーケンス自体のダイナミクスは変化しないというのが一般的な前提です。これは合理的です。なぜなら、新しいダイナミクスはまさに斬新であり、したがってこれまでに得たデータを使用して予測することはできないからです。統計学者は変化しないダイナミクスを*定常*と呼ぶ。したがって、何をするかにかかわらず、シーケンス全体の推定値は次のようにして得られます。 

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

連続した数ではなく単語などの離散的なオブジェクトを扱う場合は、上記の考慮事項が成り立ちます。唯一の違いは、このような状況では $P(x_t \mid  x_{t-1}, \ldots, x_1)$ を推定するために回帰モデルではなく分類器を使用する必要があることです。 

### マルコフモデル

自己回帰モデルでは $x_t$ を推定するために $x_{t-1}, \ldots, x_1$ ではなく $x_{t-1}, \ldots, x_{t-\tau}$ のみを使用するという近似を思い出してください。この近似が正確であれば、シーケンスは*マルコフ条件*を満たすと言います。特に $\tau = 1$ の場合、*一次マルコフモデル* があり、$P(x)$ は次の式で与えられます。 

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ where } P(x_1 \mid x_0) = P(x_1).$$

このようなモデルは、$x_t$ が離散値のみを仮定する場合に特に便利です。この場合、動的計画法を使用してチェーンに沿って値を正確に計算できるためです。たとえば、$P(x_{t+1} \mid x_{t-1})$ を効率的に計算できます。 

$$\begin{aligned}
P(x_{t+1} \mid x_{t-1})
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}
$$

過去の観測の非常に短い履歴を考慮する必要があるという事実を利用することで、$P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$。動的計画法の詳細は、この節では扱いません。制御および強化学習アルゴリズムは、このようなツールを幅広く使用しています。 

### 因果関係

原則として、$P(x_1, \ldots, x_T)$を逆の順序で展開しても問題はありません。結局のところ、条件付けによっていつでもそれを書くことができます 

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

実際、マルコフモデルがあれば、逆の条件付き確率分布も得られます。しかし、多くの場合、データには自然な方向性、つまり時間の経過とともに進む方向があります。将来の出来事が過去に影響を与えないことは明らかです。したがって、$x_t$ を変更すると、$x_{t+1}$ の今後の動作に影響を与えることはできますが、その逆には影響しない可能性があります。つまり、$x_t$ を変更しても、過去のイベントの分布は変わりません。したがって、$P(x_t \mid x_{t+1})$ よりも $P(x_{t+1} \mid x_t)$ を説明するほうが簡単になるはずです。たとえば、加法性ノイズ $\epsilon$ に対して $x_{t+1} = f(x_t) + \epsilon$ が見つかる場合もありますが、その逆は真ではないことが示されています :cite:`Hoyer.Janzing.Mooij.ea.2009`。これは素晴らしいニュースです。なぜなら、これは通常、私たちが見積もりたい前進方向だからです。Petersらの本は、このトピック:cite:`Peters.Janzing.Scholkopf.2017`についてさらに説明しています。私たちはその表面をかろうじて傷つけています。 

## 訓練

たくさんの統計ツールを見直した後、実際に試してみましょう。まず、データを生成します。単純にするために、(**タイムステップ $1, 2, \ldots, 1000$.の加法性ノイズをもつ正弦関数を使用してシーケンスデータを生成する**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
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

```{.python .input}
#@tab mxnet, pytorch
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

```{.python .input}
#@tab tensorflow
T = 1000  # Generate a total of 1000 points
time = d2l.arange(1, T + 1, dtype=d2l.float32)
x = d2l.sin(0.01 * time) + d2l.normal([T], 0, 0.2)
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

次に、このようなシーケンスを、モデルがトレーニングできるフィーチャとラベルに変換する必要があります。埋め込みディメンション $\tau$ に基づいて [**データを $y_t = x_t$ と $\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$ のペアにマッピングします] 賢明な読者は、最初の $\tau$ の履歴が十分でないため、$\tau$ のデータ例が少なくなることに気付いたかもしれません。特にシーケンスが長い場合の簡単な修正は、これらのいくつかの項を破棄することです。あるいは、シーケンスをゼロで埋めることもできます。ここでは、最初の 600 個のフィーチャとラベルのペアのみを学習に使用します。

```{.python .input}
#@tab mxnet, pytorch
tau = 4
features = d2l.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab tensorflow
tau = 4
features = tf.Variable(d2l.zeros((T - tau, tau)))
for i in range(tau):
    features[:, i].assign(x[i: T - tau + i])
labels = d2l.reshape(x[tau:], (-1, 1))
```

```{.python .input}
#@tab all
batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

ここでは、2 つの完全接続レイヤー、ReLU アクティベーション、二乗損失を備えた [**アーキテクチャをかなりシンプルに保つ:MLP**]。

```{.python .input}
# A simple MLP
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Square loss
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Note: `MSELoss` computes squared error without the 1/2 factor
loss = nn.MSELoss(reduction='none')
```

```{.python .input}
#@tab tensorflow
# Vanilla MLP architecture
def get_net():
    net = tf.keras.Sequential([tf.keras.layers.Dense(10, activation='relu'),
                              tf.keras.layers.Dense(1)])
    return net

# Note: `MeanSquaredError` computes squared error without the 1/2 factor
loss = tf.keras.losses.MeanSquaredError()
```

これで [**モデルのトレーニング**] の準備が整いました。以下のコードは、:numref:`sec_linear_concise` など、前のセクションのトレーニングループと基本的に同じです。したがって、詳細については詳しく説明しません。

```{.python .input}
def train(net, train_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

```{.python .input}
#@tab tensorflow
def train(net, train_iter, loss, epochs, lr):
    trainer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        for X, y in train_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            trainer.apply_gradients(zip(grads, params))
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

## 予測

トレーニングロスは小さいので、このモデルはうまく機能すると予想されます。これが実際に何を意味するのか見てみましょう。最初に確認するのは、モデルがどの程度良好に [**次のタイムステップで何が起こるかを予測する**]、つまり*1 ステップ先の予測* です。

```{.python .input}
#@tab all
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [d2l.numpy(x), d2l.numpy(onestep_preds)], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

ワンステップ先の予測は、予想どおりに見栄えがします。604 (`n_train + tau`) を超える観測値であっても、予測は依然として信頼できるように見えます。ただし、これには小さな問題が 1 つだけあります。時間ステップ 604 までのシーケンスデータのみを観測すると、将来の 1 ステップ先の予測の入力をすべて受け取ることは期待できません。その代わり、一歩ずつ前進していく必要があります。 

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

一般に、$x_t$ までの観測された系列では、タイムステップ $t+k$ での予測出力 $\hat{x}_{t+k}$ は $k$*-stepahead 予測* と呼ばれます。$x_{604}$ まで観測されているので、その $k$ ステップ先の予測は $\hat{x}_{604+k}$ です。つまり、[**独自の予測を使って多段階先の予測を立てる**] 必要があります。これがどれほどうまくいくか見てみましょう。

```{.python .input}
#@tab mxnet, pytorch
multistep_preds = d2l.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1)))
```

```{.python .input}
#@tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(T))
multistep_preds[:n_train + tau].assign(x[:n_train + tau])
for i in range(n_train + tau, T):
    multistep_preds[i].assign(d2l.reshape(net(
        d2l.reshape(multistep_preds[i - tau: i], (1, -1))), ()))
```

```{.python .input}
#@tab all
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [d2l.numpy(x), d2l.numpy(onestep_preds),
          d2l.numpy(multistep_preds[n_train + tau:])], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

上の例が示すように、これは目を見張るような失敗です。予測は、いくつかの予測ステップの後、かなり速く一定に減衰します。アルゴリズムがそれほどうまく機能しなかったのはなぜですか？これは最終的に、エラーが蓄積するためです。ステップ1の後に、エラー$\epsilon_1 = \bar\epsilon$が発生したとしましょう。これで、ステップ 2 の*入力* が $\epsilon_1$ によって摂動されます。したがって、定数 $c$ では $\epsilon_2 = \bar\epsilon + c \epsilon_1$ のオーダで何らかのエラーが発生します。誤差は、実際の観測値からかなり急速に逸脱する可能性があります。これはよくある現象です。たとえば、次の24時間の天気予報はかなり正確になる傾向がありますが、それを超えると精度は急速に低下します。この章以降では、これを改善する方法について説明します。 

$k = 1, 4, 16, 64$ の系列全体に対する予測を計算して、[**$k$ ステップ先予測の難しさを詳しく見てみよう]。

```{.python .input}
#@tab all
max_steps = 64
```

```{.python .input}
#@tab mxnet, pytorch
features = d2l.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = d2l.reshape(net(features[:, i - tau: i]), -1)
```

```{.python .input}
#@tab tensorflow
features = tf.Variable(d2l.zeros((T - tau - max_steps + 1, tau + max_steps)))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i].assign(x[i: i + T - tau - max_steps + 1].numpy())

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i].assign(d2l.reshape(net((features[:, i - tau: i])), -1))
```

```{.python .input}
#@tab all
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [d2l.numpy(features[:, tau + i - 1]) for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

これは、将来予測を進めようとするにつれて、予測の質がどのように変化するかを明確に示しています。4ステップ先の予測は依然として良好に見えますが、それを超えるものはほとんど役に立ちません。 

## [概要

* 補間と外挿では難易度が大きく異なります。したがって、シーケンスがある場合は、トレーニング時には常にデータの時間的順序を尊重します。つまり、将来のデータではトレーニングを行わないでください。
* シーケンスモデルには、推定のための特殊な統計ツールが必要です。2 つの一般的な選択肢は、自己回帰モデルと潜在変数自己回帰モデルです。
* 因果モデル (先に進む時間など) では、通常、順方向の推定は逆方向よりもずっと簡単です。
* タイムステップ $t$ までの観測されたシーケンスでは、タイムステップ $t+k$ での予測出力は $k$*-step Ahead 予測* です。$k$ を大きくしてさらに予測すると、誤差が累積され、予測の質が大きく低下します。

## 演習

1. このセクションの実験でモデルを改良します。
    1. 過去4件以上の観測を取り入れますか？本当に何個必要ですか？
    1. 騒音がなかったら、過去に何回観測する必要がありますか？ヒント:$\sin$ と $\cos$ を微分方程式として書くことができます。
    1. 特徴量の総数を一定に保ちながら、古い観測値を組み込むことはできますか？これにより精度は向上しますか？なぜ？
    1. ニューラルネットワークのアーキテクチャを変更し、パフォーマンスを評価します。
1. 投資家は、購入するのに適した証券を見つけたいと思っています。彼は過去のリターンを調べて、どちらがうまくいくかを判断します。この戦略で何がうまくいかない可能性がありますか？
1. 因果関係はテキストにも当てはまりますか？どの程度？
1. データのダイナミクスを捉えるために潜在的自己回帰モデルが必要になる場合について例を挙げてください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:

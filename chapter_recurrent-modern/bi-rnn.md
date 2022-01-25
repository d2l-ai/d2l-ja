# 双方向リカレントニューラルネットワーク
:label:`sec_bi_rnn`

シーケンス学習では、これまでの目標は、時系列のコンテキストや言語モデルのコンテキストなど、これまで見てきたことを考慮して次の出力をモデル化することであると想定していました。これは典型的なシナリオですが、遭遇する可能性があるのはそれだけではありません。この問題を説明するために、空白をテキストシーケンスで埋める次の 3 つのタスクについて考えてみます。 

* 私は`___`です。
* お腹が空いてる `___`
* お腹が空いてる `___`、豚を半分食べることができます。

入手可能な情報の量によっては、「幸せ」、「ない」、「非常に」など、非常に異なる単語を空白に埋める場合があります。フレーズの終わり（ある場合）は、どの単語を選ぶべきかについての重要な情報を伝えていることは明らかです。これを利用できないシーケンスモデルは、関連するタスクではパフォーマンスが低下します。例えば、名前付きエンティティの認識 (例えば、「緑」が「Mr. Green」を意味するのか、それとも色を指しているのかを認識する) において、より長い範囲のコンテキストをうまく処理することも同様に重要です。この問題に対処するためのインスピレーションを得るために、確率的グラフィカルモデルに回り道してみましょう。 

## 隠れマルコフモデルにおける動的計画法

このサブセクションでは、動的計画法の問題について説明します。特定の技術的な詳細は、ディープラーニングモデルの理解には関係ありませんが、ディープラーニングを使用する理由や、特定のアーキテクチャを選択する理由の動機付けに役立ちます。 

確率的グラフィカルモデルを使用して問題を解きたい場合、たとえば潜在変数モデルを次のように設計できます。任意のタイムステップ $t$ で、$P(x_t \mid h_t)$ を介して観測された放出 $x_t$ を支配する潜在変数 $h_t$ が存在すると仮定します。さらに、遷移$h_t \to h_{t+1}$は何らかの状態遷移確率$P(h_{t+1} \mid h_{t})$によって与えられる。この確率的グラフィカルモデルは :numref:`fig_hmm` のように *隠れマルコフモデル* になります。 

![A hidden Markov model.](../img/hmm.svg)
:label:`fig_hmm`

したがって、一連の $T$ 個の観測値では、観測された状態と隠れた状態について次の結合確率分布が得られます。 

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ where } P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

ここで、$x_j$ を除くすべての $x_i$ が観測され、$P(x_j \mid x_{-j})$ を計算することが目標であると仮定します。ここで $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$ です。$P(x_j \mid x_{-j})$ には潜在変数がないため、$h_1, \ldots, h_T$ の選択肢の組み合わせをすべて合計することを検討します。$h_i$ のいずれかが $k$ 個の異なる値 (有限個の状態) をとることができる場合、これは $k^T$ 項を合計する必要があることを意味し、通常はミッションインポッシブルです。幸いなことに、これには*動的プログラミング*という洗練された解決策があります。 

その仕組みを確認するには、潜在変数 $h_1, \ldots, h_T$ を順番に合計することを検討してください。:eqref:`eq_hmm_jointP` によると、この結果は次のようになります。 

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

一般に、*前方再帰*は次のようになります。 

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

再帰は $\pi_1(h_1) = P(h_1)$ として初期化されます。抽象的には $\pi_{t+1} = f(\pi_t, x_t)$ と書くことができます。$f$ は学習可能な関数です。これは、RNNの文脈でこれまで議論した潜在変数モデルの更新方程式と非常によく似ています！  

前方再帰と全く同じように、同じ潜在変数セットを逆再帰で合計することもできます。これにより、次の結果が得られます。 

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

したがって、*逆方向再帰*を次のように書くことができます。 

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

初期化は$\rho_T(h_T) = 1$で行います。順方向再帰と逆方向再帰の両方により、$(h_1, \ldots, h_T)$ のすべての値に対して $\mathcal{O}(kT)$ (線形) 時間で $T$ を超える潜在変数を指数関数時間ではなく合計することができます。これは、グラフィカルモデルによる確率的推論の大きな利点の 1 つです。これは一般的なメッセージパッシングアルゴリズム :cite:`Aji.McEliece.2000` の非常に特殊なインスタンスでもあります。前方再帰と後方再帰の両方を組み合わせることで、 

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

抽象用語では、後方再帰は $\rho_{t-1} = g(\rho_t, x_t)$ と書けることに注意してください。$g$ は学習可能な関数です。繰り返しますが、これは更新方程式に非常によく似ており、RNNでこれまで見てきたものとは異なり、逆方向に実行されています。実際、隠れマルコフモデルは、将来のデータが利用可能になったときにそれを知ることから利益を得ます。信号処理の科学者は、将来の観測値を知っている場合と知らない場合の 2 つのケースを、内挿と外挿として区別します。:cite:`Doucet.De-Freitas.Gordon.2001` の詳細については、逐次モンテカルロアルゴリズムに関する本の入門章を参照してください。 

## 双方向モデル

隠れマルコフモデルと同等の先読み能力を提供するメカニズムをRNNに持ちたいのであれば、これまで見てきたRNN設計を修正する必要があります。幸いなことに、これは概念的には簡単です。RNN を最初のトークンからフォワードモードでのみ実行するのではなく、最後のトークンから後ろから前に走る別のトークンから別のトークンを開始します。 
*双方向 RNN* は、情報を逆方向に渡す隠れ層を追加して、そのような情報をより柔軟に処理します。:numref:`fig_birnn` は、単一の隠れ層をもつ双方向 RNN のアーキテクチャを示しています。

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

実際、これは隠れマルコフモデルの動的計画法における順方向再帰や逆方向再帰とあまり変わらない。主な違いは、前のケースでは、これらの方程式が特定の統計的意味を持っていたことです。今、彼らはそのような簡単にアクセスできる解釈を欠いており、私たちはそれらを一般的で学習可能な関数として扱うことができます。この移行は、最新の深層ネットワークの設計を導く多くの原則を象徴しています。まず、古典的な統計モデルの関数的依存性のタイプを使用し、次にそれらを一般的な形式でパラメーター化します。 

### 定義

双方向 RNN は :cite:`Schuster.Paliwal.1997` によって導入されました。各種のアーキテクチャについての詳しい説明は :cite:`Graves.Schmidhuber.2005` の文書も参照してください。このようなネットワークの詳細を見てみましょう。 

任意のタイムステップ $t$ で、ミニバッチ入力 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (例の数:$n$、各例の入力数:$d$) を指定し、隠れ層の活性化関数を $\phi$ とします。双方向アーキテクチャでは、このタイムステップの順方向および逆方向の隠れ状態はそれぞれ $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ と $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ であると仮定します。$h$ は隠れユニットの数です。前方および後方非表示ステートの更新は以下のとおりです。 

$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

重み $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$ とバイアス $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ はすべてモデルパラメーターです。 

次に、順方向および逆方向の隠れ状態 $\overrightarrow{\mathbf{H}}_t$ と $\overleftarrow{\mathbf{H}}_t$ を連結して、出力層に供給される隠れ状態 $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ を取得します。複数の隠れ層をもつディープ双方向RNNでは、そのような情報は*input* として次の双方向層に渡されます。最後に、出力層は出力 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (出力数:$q$) を計算します。 

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

ここで、重み行列 $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ とバイアス $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ は出力層のモデルパラメーターです。実際には、2 つの方向には異なる数の隠れたユニットがあります。 

### 計算コストと応用

双方向 RNN の重要な特徴の 1 つは、シーケンスの両端からの情報が出力の推定に使用されることです。つまり、将来と過去の観測の両方からの情報を使用して、現在の観測を予測します。次のトークン予測の場合、これは私たちが望むものではありません。結局のところ、次のトークンを予測するときに、次のトークンを知るという贅沢はありません。したがって、双方向RNNを単純に使用した場合、あまり精度は得られません。トレーニング中、現在を推定するための過去と未来のデータがあります。テスト時間中は過去のデータしかないため、精度が悪くなります。これについては、以下の実験で説明します。 

傷害に侮辱を加えるために、双方向のRNNも非常に遅い。この主な理由は、順伝播では双方向レイヤーで順方向再帰と逆方向再帰の両方が必要であり、逆伝播は順伝播の結果に依存しているためです。したがって、グラデーションは非常に長い依存関係チェーンを持つことになります。 

実際には、双方向レイヤーは、欠落している単語の記入、トークンの注釈 (名前付きエンティティの認識など)、シーケンス処理パイプラインのステップとしてのシーケンスの卸売り (機械翻訳など) など、ごくわずかなアプリケーションでのみ使用されます。:numref:`sec_bert` と :numref:`sec_sentiment_rnn` では、双方向 RNN を使用してテキストシーケンスをエンコードする方法を紹介します。 

## (**誤ったアプリケーションに対する双方向 RNN のトレーニング**)

双方向RNNが過去と未来のデータを使用し、それを単に言語モデルに適用するという事実に関するアドバイスをすべて無視すれば、許容できるパープレキシティで推定値が得られます。それにもかかわらず、以下の実験が示すように、将来のトークンを予測するモデルの能力は著しく損なわれています。妥当な混乱にもかかわらず、何度も繰り返した後でも意味不明な表現しか発生しません。以下のコードを、間違ったコンテキストで使用することに対する注意例として含めます。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

上記の理由により、出力は明らかに不十分です。双方向 RNN のより効果的な使用方法については、:numref:`sec_sentiment_rnn` のセンチメント分析アプリケーションを参照してください。 

## [概要

* 双方向 RNN では、各タイムステップの隠れ状態は、現在のタイムステップの前後のデータによって同時に決定されます。
* 双方向 RNN は、確率的グラフィカルモデルにおけるフォワードバックワードアルゴリズムと非常によく似ています。
* 双方向RNNは、シーケンスの符号化や、双方向のコンテキストを考慮した観測値の推定に主に有用です。
* 双方向 RNN は、勾配チェーンが長いため、トレーニングに非常にコストがかかります。

## 演習

1. 方向によって隠れ単位の数が異なる場合、$\mathbf{H}_t$ の形状はどのように変化しますか。
1. 複数の隠れ層をもつ双方向 RNN を設計します。
1. 多義性は自然言語では一般的です。たとえば、「銀行」という言葉は、「現金を預けるために銀行に行った」と「座るために銀行に行った」という文脈で意味が異なります。コンテキストシーケンスと単語が与えられると、コンテキスト内の単語のベクトル表現が返されるようなニューラルネットワークモデルをどのように設計できるでしょうか。多義性を扱うには、どのタイプのニューラルアーキテクチャが好まれますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:

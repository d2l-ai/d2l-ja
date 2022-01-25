# 自然言語推論:アテンションを使う
:label:`sec_natural-language-inference-attention`

:numref:`sec_natural-language-inference-and-dataset` では、自然言語推論タスクと SNLI データセットを導入しました。Parikhらは、複雑で深いアーキテクチャに基づく多くのモデルを考慮して、注意メカニズムを用いて自然言語推論に取り組むことを提案し、それを「分解可能な注意モデル」と呼んだ。:cite:`Parikh.Tackstrom.Das.ea.2016`。この結果、再帰層や畳み込み層のないモデルが生成され、SNLI データセットではパラメーターがはるかに少なくて済み、その時点で最良の結果が得られます。このセクションでは、:numref:`fig_nlp-map-nli-attention` に示すように、自然言語推論のためのこの注意ベースの方法 (MLP を使用) について説明し、実装します。 

![This section feeds pretrained GloVe to an architecture based on attention and MLPs for natural language inference.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`

## ザ・モデル

構内や仮説内のトークンの順序を維持するよりも簡単です。あるテキストシーケンスのトークンを他のテキストシーケンスのすべてのトークンに揃え、その逆も可能です。次に、そのような情報を比較および集約して、前提と仮説の論理的関係を予測できます。機械翻訳における原文と訳文間のトークンのアライメントと同様に、前提と仮説の間のトークンのアライメントは、アテンションメカニズムによってきちんと達成できます。 

![Natural language inference using attention mechanisms.](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` は、注意メカニズムを用いた自然言語推論法を示しています。大まかに言うと、参加する、比較する、集約という 3 つの共同トレーニングされたステップで構成されます。以下では、それらを段階的に説明します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### 出席する

最初のステップは、あるテキストシーケンスのトークンを他のシーケンスの各トークンに揃えることです。前提が「寝る必要がある」で、仮説が「疲れた」と仮定します。意味的な類似性から、仮説の「i」を前提に「i」を揃え、仮説の「疲れた」を前提に「睡眠」を揃えたいと思うかもしれません。同様に、前提の「i」を仮説の「i」に合わせ、前提の「必要」と「睡眠」を仮説の「疲れ」に合わせるといいかもしれません。このようなアラインメントは加重平均を使用して*ソフト* であることに注意してください。アライメントされるトークンには、理想的に大きなウェイトが関連付けられます。デモンストレーションを容易にするために、:numref:`fig_nli_attention` はこのようなアライメントを*難しい* 方法で示しています。 

ここで、アテンションメカニズムを使用したソフトアライメントについて詳しく説明します。$\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$ と $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ によって、トークンの数がそれぞれ $m$ と $n$ である前提と仮説を示します。$\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) は $d$ 次元の単語ベクトルです。ソフトアライメントでは、アテンションウェイト $e_{ij} \in \mathbb{R}$ を次のように計算します。 

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

ここで、関数 $f$ は、次の `mlp` 関数で定義されている MLP です。$f$ の出力次元は `mlp` という引数 `num_hiddens` によって指定されます。

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

:eqref:`eq_nli_e` では $f$ は入力 $\mathbf{a}_i$ と $\mathbf{b}_j$ のペアを入力として取るのではなく、別々に受け取ることを強調しておく必要があります。この*分解* トリックは $mn$ アプリケーション (二次複雑度) ではなく $f$ の $m + n$ アプリケーション (線形複雑度) のみになります。 

:eqref:`eq_nli_e` で注意の重みを正規化し、仮説に含まれるすべてのトークンベクトルの加重平均を計算して、前提の $i$ によってインデックス付けされたトークンとソフトに整合する仮説の表現を取得します。 

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

同様に、仮説で $j$ でインデックス付けされた各トークンについて、前提トークンのソフトアラインメントを計算します。 

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

以下では `Attend` クラスを定義して、仮説 (`beta`) と入力仮説 `A` のソフトアラインメント、入力仮説 `B` を使用した前提のソフトアラインメント (`alpha`) を計算します。

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        # no. of tokens in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        # `embed_size`), where sequence B is softly aligned with each token
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        # `embed_size`), where sequence A is softly aligned with each token
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### 比較する

次のステップでは、あるシーケンスのトークンを、そのトークンとソフトアラインされた別のシーケンスと比較します。ソフトアライメントでは、あるシーケンスのすべてのトークンが、おそらくアテンションウェイトは異なるものの、もう一方のシーケンスのトークンと比較されることに注意してください。簡単に説明できるように、:numref:`fig_nli_attention` は、トークンとアライメントされたトークンをハードな方法でペアリングします。たとえば、前提の「必要」と「睡眠」が仮説で「疲れた」と合致していると出席するステップが判断したとすると、「疲れている-睡眠が必要」のペアが比較されます。 

比較ステップでは、あるシーケンスのトークンの連結 (演算子 $[\cdot, \cdot]$) と、もう一方のシーケンスのアライメントされたトークンを関数 $g$ (MLP) に送ります。 

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab` 

:eqref:`eq_nli_v_ab` では、$\mathbf{v}_{A,i}$ は、前提内のトークン $i$ と、トークン $i$ にソフトアライメントされたすべての仮説トークンの比較です。$\mathbf{v}_{B,j}$ は、仮説のトークン $j$ と、トークン $j$ にソフトアライメントされたすべての前提トークンの比較です。以下の `Compare` クラスでは、比較ステップなどを定義しています。

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### アグリゲーティング

2 組の比較ベクトル $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) と $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) を用意して、最後のステップでこのような情報を集約して論理的な関係を推測します。まず、両方のセットを合計します。 

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

次に、両方の要約結果を関数 $h$ (MLP) に連結して、論理関係の分類結果を取得します。 

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

集約ステップは、次の `Aggregate` クラスで定義されています。

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### すべてのものをまとめる

出席ステップ、比較ステップ、集約ステップをまとめることで、分解可能な注意モデルを定義し、これら 3 つのステップを共同でトレーニングします。

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## モデルのトレーニングと評価

次に、SNLI データセットで定義した分解可能アテンションモデルをトレーニングし、評価します。まず、データセットを読み取ります。 

### データセットの読み込み

:numref:`sec_natural-language-inference-and-dataset` で定義されている関数を使用して SNLI データセットをダウンロードして読み込みます。バッチサイズとシーケンス長はそれぞれ $256$ と $50$ に設定されます。

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### モデルを作成する

入力トークンを表すには、事前学習済みの 100 次元 GLOVE 埋め込みを使用します。したがって、:eqref:`eq_nli_e` のベクトル $\mathbf{a}_i$ と $\mathbf{b}_j$ の次元を 100 として事前定義しています。:eqref:`eq_nli_e` の関数 $f$ と :eqref:`eq_nli_v_ab` の関数 $g$ の出力次元は 200 に設定されます。次に、モデルインスタンスを作成し、パラメーターを初期化し、GLOVE 埋め込みをロードして入力トークンのベクトルを初期化します。

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### モデルのトレーニングと評価

:numref:`sec_multi_gpu` の `split_batch` 関数では、テキストシーケンス (または画像) などの 1 つの入力を受け取るのに対し、前提や仮説などの複数の入力をミニバッチで受け取る `split_batch_multi_inputs` 関数を定義します。

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

これで、SNLI データセットでモデルをトレーニングし、評価することができます。

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### モデルを使う

最後に、前提と仮説の対の論理関係を出力する予測関数を定義します。

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

学習済みモデルを使用して、サンプル文ペアの自然言語推論結果を得ることができます。

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## [概要

* 分解可能な注意モデルは、前提と仮説の間の論理的関係を予測するための 3 つのステップ (出席する、比較する、集約する) で構成されます。
* アテンション・メカニズムにより、あるテキスト・シーケンスのトークンを他のテキスト・シーケンスのすべてのトークンと整列させたり、その逆を行ったりすることができます。このようなアライメントは加重平均を使用するとソフトになり、アライメントされるトークンには理想的に大きなウェイトが関連付けられます。
* アテンションウェイトを計算する場合、分解のトリックは 2 次複雑度よりも望ましい線形複雑度になります。
* 事前学習済みの単語ベクトルを、自然言語推論などの下流の自然言語処理タスクの入力表現として使用できます。

## 演習

1. ハイパーパラメーターの他の組み合わせを使用してモデルに学習をさせます。テストセットの精度は向上しますか？
1. 自然言語推論のための分解可能アテンションモデルの大きな欠点は何ですか？
1. 任意の文のペアについて、意味的類似性のレベル (たとえば、0 と 1 の間の連続値) を取得するとします。データセットの収集とラベル付けはどのように行えばよいのでしょうか？アテンション機構のあるモデルを設計できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:

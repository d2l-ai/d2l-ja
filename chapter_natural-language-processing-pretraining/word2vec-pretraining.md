# プレトレーニングword2vec
:label:`sec_word2vec_pretraining`

:numref:`sec_word2vec` で定義されているスキップグラムモデルを実装します。次に、PTB データセットで負のサンプリングを使用して word2vec を事前学習します。まず、:numref:`sec_word2vec_data` で説明した `d2l.load_data_ptb` 関数を呼び出して、このデータセットのデータイテレータとボキャブラリを取得しましょう。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## スキップ・グラム・モデル

埋め込み層とバッチ行列乗算を使用してスキップグラムモデルを実装します。まず、埋め込みレイヤーの仕組みを確認しましょう。 

### 埋め込みレイヤー

:numref:`sec_seq2seq` で説明したように、埋め込み層はトークンのインデックスをその特徴ベクトルにマッピングします。この層の重みは、行数がディクショナリのサイズ (`input_dim`) に等しく、列数が各トークンのベクトル次元 (`output_dim`) に等しい行列です。単語埋め込みモデルのトレーニングが完了したら、この重みが必要となります。

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

埋め込みレイヤーの入力はトークン (単語) のインデックスです。どのトークンインデックス $i$ でも、そのベクトル表現は、埋め込み層の重み行列の $i^\mathrm{th}$ 行から取得できます。ベクトルの次元 (`output_dim`) が 4 に設定されているため、埋め込み層は shape (2, 3) を持つトークンインデックスのミニバッチに対して、形状 (2, 3, 4) のベクトルを返します。

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### フォワード伝播の定義

順伝播では、スキップグラムモデルの入力には、形状 (バッチサイズ、1) の中央ワードインデックス `center` と、シェイプの連結されたコンテキストおよびノイズワードインデックス `contexts_and_negatives` (バッチサイズ、`max_len`) が含まれます。`max_len` は :numref:`subsec_word2vec-minibatch-loading` で定義されています。これらの 2 つの変数は、まず埋め込み層を介してトークンインデックスからベクトルに変換され、次に、それらのバッチ行列乗算 (:numref:`subsec_batch_dot` で説明) が shape (バッチサイズ、1, `max_len`) の出力を返します。出力の各要素は、センターワードベクトルとコンテキストワードベクトルまたはノイズワードベクトルの内積です。

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

いくつかの入力例について、この `skip_gram` 関数の出力形状を出力してみましょう。

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 訓練

負のサンプリングでスキップグラムモデルに学習をさせる前に、まず損失関数を定義しましょう。 

### バイナリクロスエントロピー損失

:numref:`subsec_negative-sampling` の負のサンプリングに対する損失関数の定義に従って、バイナリクロスエントロピー損失を使用します。

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

:numref:`subsec_word2vec-minibatch-loading` のマスク変数とラベル変数についての説明を思い出してください。次は、与えられた変数のバイナリクロスエントロピー損失を計算します。

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

以下に、バイナリクロスエントロピー損失のシグモイド活性化関数を使用して、上記の結果を (効率の悪い方法で) 計算する方法を示します。2 つの出力は、マスクされていない予測で平均化された 2 つの正規化された損失と見なすことができます。

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### モデルパラメーターの初期化

ボキャブラリ内のすべての単語がセンターワードとコンテキストワードとして使用される場合、それぞれ2つの埋め込みレイヤーを定義します。ワードベクトルの次元 `embed_size` は 100 に設定されます。

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### トレーニングループの定義

トレーニングループの定義を以下に示します。パディングが存在するため、損失関数の計算は以前の学習関数とわずかに異なります。

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

これで、負のサンプリングを使用してスキップグラムモデルに学習をさせることができます。

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 単語の埋め込みを適用する
:label:`subsec_apply-word-embed`

word2vec モデルに学習させた後、学習済みモデルの単語ベクトルの余弦類似度を使用して、辞書から入力単語に最も意味的に類似する単語を見つけることができます。

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## [概要

* 埋め込み層とバイナリ交差エントロピー損失を使用して、負のサンプリングでスキップグラムモデルを学習させることができます。
* 単語埋め込みのアプリケーションには、単語ベクトルの余弦類似度に基づいて、特定の単語について意味的に類似する単語を見つけることが含まれます。

## 演習

1. 学習済みモデルを使用して、他の入力単語に対して意味的に類似する単語を検出します。ハイパーパラメーターをチューニングして結果を改善できるか
1. トレーニングコーパスが巨大な場合、*モデルパラメーターの更新時* で、現在のミニバッチで中心となる単語のコンテキストワードとノイズワードをサンプリングすることがよくあります。言い換えると、同じセンターワードでも、異なるトレーニングエポックで異なるコンテキストワードまたはノイズワードが含まれる場合があります。この方法の利点は何ですか？このトレーニング方法を実装してみてください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:

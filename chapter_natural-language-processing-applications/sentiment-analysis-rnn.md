# センチメント分析:リカレントニューラルネットワークの使用
:label:`sec_sentiment_rnn`

単語の類似性や類推のタスクと同様に、事前学習済みの単語ベクトルをセンチメント分析に適用することもできます。:numref:`sec_sentiment` の IMDb レビューデータセットはそれほど大きくないため、大規模コーパスで事前学習されたテキスト表現を使用すると、モデルの過適合が減少する可能性があります。:numref:`fig_nlp-map-sa-rnn` に示す具体的な例として、事前学習済みの GLOVE モデルを使用して各トークンを表し、これらのトークン表現を多層双方向 RNN にフィードしてテキストシーケンス表現を取得します。テキストシーケンス表現は、センチメント分析出力 :cite:`Maas.Daly.Pham.ea.2011` に変換されます。同じダウンストリームアプリケーションについては、後で別のアーキテクチャを選択することを検討します。 

![This section feeds pretrained GloVe to an RNN-based architecture for sentiment analysis.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN による単一テキストの表現

センチメント分析などのテキスト分類タスクでは、可変長のテキストシーケンスが固定長のカテゴリに変換されます。次の `BiRNN` クラスでは、テキストシーケンスの各トークンは埋め込み層 (`self.embedding`) を介して個別の事前学習済みの GLOVE 表現を取得しますが、シーケンス全体が双方向 RNN (`self.encoder`) によってエンコードされます。具体的には、初期タイムステップと最終タイムステップの両方における双方向 LSTM の隠れ状態 (最後のレイヤー) は、テキストシーケンスの表現として連結されます。この単一のテキスト表現は、2 つの出力 (「正」と「負」) を持つ完全結合レイヤー (`self.decoder`) によって出力カテゴリに変換されます。

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        outs = self.decoder(encoding)
        return outs
```

センチメント分析用の 1 つのテキストを表す 2 つの隠れ層をもつ双方向 RNN を構築してみましょう。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

## 事前学習済みの単語ベクトルの読み込み

以下では、ボキャブラリ内のトークンの事前学習済みの 100 次元 (`embed_size` との整合性が必要) GLOVE 埋め込みをロードします。

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

語彙に含まれるすべてのトークンのベクトルの形状を出力します。

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

これらの事前学習済みの単語ベクトルは、レビュー内のトークンを表すために使用され、学習中にこれらのベクトルは更新されません。

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## モデルのトレーニングと評価

これで、センチメント分析用に双方向 RNN をトレーニングできます。

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

学習済みモデル `net` を使用してテキストシーケンスのセンチメントを予測するために、次の関数を定義します。

```{.python .input}
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

最後に、学習済みのモデルを使用して、2 つの単純な文のセンチメントを予測します。

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## [概要

* 事前学習済みの単語ベクトルは、テキストシーケンス内の個々のトークンを表すことができます。
* 双方向 RNN は、初期タイムステップと最終タイムステップで隠れ状態を連結するなどして、テキストシーケンスを表すことができます。この単一のテキスト表現は、完全に接続されたレイヤーを使用してカテゴリに変換できます。

## 演習

1. エポック数を増やす。トレーニングとテストの精度を向上させることはできますか？他のハイパーパラメーターのチューニングはどうですか？
1. 300 次元の GLOVE 埋め込みなど、より大きい事前学習済みの単語ベクトルを使用します。分類の精度は向上しますか？
1. spaCy トークン化を使用して分類の精度を向上させることはできますか？spaCy (`pip install spacy`) をインストールし、英語版パッケージ (`python -m spacy download en`) をインストールする必要があります。コードでは、まず spaCY (`import spacy`) をインポートします。次に、spaCy 英語パッケージ (`spacy_en = spacy.load('en')`) をロードします。最後に、関数 `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` を定義し、元の `tokenizer` 関数を置き換えます。GLOVE と spaCy ではフレーズトークンの形式が異なることに注意してください。たとえば、「new york」という語句は、GLOVE では「new-york」の形をとり、spaCy のトークン化後に「ニューヨーク」の形式をとります。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:

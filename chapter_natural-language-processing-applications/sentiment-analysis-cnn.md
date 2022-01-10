# センチメント分析:畳み込みニューラルネットワークの使用 
:label:`sec_sentiment_cnn`

:numref:`chap_cnn` では、隣接画素などの局所特徴に適用された 2 次元 CNN をもつ二次元画像データを処理するメカニズムについて検討した。CNN はもともとコンピュータービジョン用に設計されましたが、自然言語処理にも広く使用されています。簡単に言えば、テキストシーケンスを一次元のイメージと考えてください。このようにして、1 次元 CNN はテキスト内の $n$ グラムなどのローカルフィーチャを処理できます。 

このセクションでは、*TextCNN* モデルを使用して、単一テキスト :cite:`Kim.2014` を表現するための CNN アーキテクチャの設計方法を示します。RNN アーキテクチャと GloVE 事前トレーニングをセンチメント分析に使用する :numref:`fig_nlp-map-sa-rnn` と比較すると、:numref:`fig_nlp-map-sa-cnn` の唯一の違いはアーキテクチャの選択にあります。 

![This section feeds pretrained GloVe to a CNN-based architecture for sentiment analysis.](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
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

## 一次元たたみ込み

モデルを紹介する前に、一次元の畳み込みがどのように機能するかを見てみましょう。これは、相互相関演算に基づく 2 次元畳み込みの特殊なケースにすぎないことに注意してください。 

![One-dimensional cross-correlation operation. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`

:numref:`fig_conv1d` に示すように、1 次元の場合、畳み込みウィンドウは入力テンソルを横切って左から右にスライドします。スライディング中、ある位置の畳み込みウィンドウに含まれる入力サブテンソル (:numref:`fig_conv1d` では $0$ と $1$) とカーネルテンソル (:numref:`fig_conv1d` の $1$ と $2$) は要素単位で乗算されます。これらの乗算の和は、出力テンソルの対応する位置に 1 つのスカラー値 (:numref:`fig_conv1d` の $0\times1+1\times2=2$ など) を与えます。 

次の `corr1d` 関数では、一次元の相互相関を実装しています。入力テンソル `X` とカーネルテンソル `K` を指定すると、出力テンソル `Y` が返されます。

```{.python .input}
#@tab all
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

:numref:`fig_conv1d` から入力テンソル `X` とカーネルテンソル `K` を構築して、上記の 1 次元相互相関実装の出力を検証できます。

```{.python .input}
#@tab all
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

複数のチャネルを持つ 1 次元入力の場合、畳み込みカーネルは同じ数の入力チャネルをもつ必要があります。次に、各チャネルについて、入力の 1 次元テンソルと畳み込みカーネルの 1 次元テンソルに対して相互相関演算を実行し、すべてのチャネルにわたって結果を合計して 1 次元の出力テンソルを生成します。:numref:`fig_conv1d_channel` に、1 次元の相互相関演算を示します。3つの入力チャンネル付き。 

![One-dimensional cross-correlation operation with 3 input channels. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`

複数の入力チャネルに対して 1 次元の相互相関演算を実装し、:numref:`fig_conv1d_channel` で結果を検証できます。

```{.python .input}
#@tab all
def corr1d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

複数入力チャネルの 1 次元相互相関は、単一入力チャネルの 2 次元相互相関に相当することに注意してください。説明すると、:numref:`fig_conv1d_channel` における複数入力チャネルの 1 次元相互相関の等価形式は :numref:`fig_conv1d_2d` の単一入力チャネルの 2 次元相互相関であり、畳み込みカーネルの高さは入力テンソルの高さと同じでなければなりません。 

![Two-dimensional cross-correlation operation with a single input channel. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`

:numref:`fig_conv1d` と :numref:`fig_conv1d_channel` の両出力にはチャンネルが 1 つしかありません。:numref:`subsec_multi-output-channels` で説明した複数の出力チャネルを持つ 2 次元畳み込みと同様に、1 次元の畳み込みに対して複数の出力チャネルを指定することもできます。 

## Max Over-Time プーリング

同様に、プーリングを使用して、時間ステップ全体で最も重要な特徴としてシーケンス表現から最大値を抽出できます。textCNN で使用される*max-overtime プーリング* は、1 次元のグローバル最大プーリング :cite:`Collobert.Weston.Bottou.ea.2011` のように機能します。各チャンネルが異なるタイムステップで値を保存するマルチチャンネル入力の場合、各チャンネルの出力はそのチャンネルの最大値になります。max-overtime プーリングでは、チャネルごとに異なるタイムステップ数が許容されることに注意してください。 

## TextCNN モデル

TextCNN モデルは、1 次元の畳み込みと max-over time プーリングを使用して、事前学習済みのトークン表現を個別に入力として受け取り、下流アプリケーションのシーケンス表現を取得して変換します。 

$d$ 次元のベクトルで表される $n$ トークンを持つ 1 つのテキストシーケンスの場合、入力テンソルの幅、高さ、チャネル数はそれぞれ $n$、$1$、および $d$ です。TextCNN モデルは次のように入力を出力に変換します。 

1. 複数の 1 次元畳み込みカーネルを定義し、入力に対して個別に畳み込み演算を実行します。幅の異なる畳み込みカーネルは、隣接するトークンの数が異なるローカル特徴を捕捉する可能性があります。
1. すべての出力チャネルで max-overtime プーリングを実行し、すべてのスカラープーリング出力をベクトルとして連結します。
1. 全結合層を使用して、連結されたベクトルを出力カテゴリに変換します。ドロップアウトはオーバーフィットを減らすために使用できます。

![The model architecture of textCNN.](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

:numref:`fig_conv1d_textcnn` は、TextCNN のモデルアーキテクチャを具体的な例とともに示しています。入力は 11 個のトークンを持つ文で、各トークンは 6 次元のベクトルで表されます。そこで、幅11の6チャンネル入力があります。幅 2 と 4 の 2 つの 1 次元畳み込みカーネルを定義し、それぞれ 4 つと 5 つの出力チャネルを使用します。これらは、幅 $11-2+1=10$ の 4 つの出力チャンネルと、幅 $11-4+1=8$ の 5 つの出力チャンネルを生成します。これら 9 つのチャネルの幅は異なりますが、max-over time プーリングでは連結された 9 次元ベクトルが得られ、最終的にバイナリセンチメント予測用に 2 次元の出力ベクトルに変換されます。 

### モデルを定義する

TextCNN モデルを以下のクラスに実装します。:numref:`sec_sentiment_rnn` の双方向 RNN モデルと比較して、リカレント層を畳み込み層に置き換える以外に、2 つの埋め込み層も使用します。1 つはトレーニング可能な重みをもつ、もう 1 つは固定された重みをもつ。

```{.python .input}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

```{.python .input}
#@tab pytorch
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer not to be trained
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # The max-over-time pooling layer has no parameters, so this instance
        # can be shared
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate two embedding layer outputs with shape (batch size, no.
        # of tokens, token vector dimension) along vectors
        embeddings = torch.cat((
            self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Per the input format of one-dimensional convolutional layers,
        # rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, a tensor of shape (batch size, no. of channels, 1) is
        # obtained. Remove the last dimension and concatenate along channels
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

TextCNN インスタンスを作成してみましょう。カーネル幅が 3、4、5 の 3 つの畳み込み層があり、すべて 100 個の出力チャネルがあります。

```{.python .input}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights);
```

### 事前学習済みの単語ベクトルの読み込み

:numref:`sec_sentiment_rnn` と同様に、事前学習済みの 100 次元 GLOVE 埋め込みを初期化されたトークン表現としてロードします。これらのトークン表現 (重みの埋め込み) は `embedding` でトレーニングされ、`constant_embedding` で修正されます。

```{.python .input}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

### モデルのトレーニングと評価

これで、TextCNN モデルをセンチメント分析用にトレーニングできます。

```{.python .input}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

以下では、学習済みのモデルを使用して、2 つの単純な文のセンチメントを予測します。

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## [概要

* 1 次元 CNN は、テキスト内の $n$ グラムなどのローカルフィーチャを処理できます。
* 多入力チャネルの 1 次元相互相関は、単一入力チャネルの 2 次元相互相関に相当します。
* max-overtime プーリングでは、チャネルごとに異なるタイムステップ数が許容されます。
* TextCNN モデルでは、1 次元の畳み込み層と max-over time プーリング層を使用して、個々のトークン表現を下流のアプリケーション出力に変換します。

## 演習

1. 分類の精度や計算効率など、:numref:`sec_sentiment_rnn` とこのセクションのセンチメント分析のために、ハイパーパラメーターを調整し、2 つのアーキテクチャを比較します。
1. :numref:`sec_sentiment_rnn` の演習で紹介した方法を使用して、モデルの分類精度をさらに向上させることができますか。
1. 入力リプレゼンテーションに位置エンコーディングを追加します。分類の精度は向上しますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/393)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1425)
:end_tab:

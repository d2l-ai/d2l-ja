# 単語埋め込みを事前学習するためのデータセット
:label:`sec_word2vec_data`

word2vec モデルの技術的な詳細と近似的なトレーニング方法がわかったところで、それらの実装を見ていきましょう。具体的には、:numref:`sec_word2vec` ではスキップグラムモデル、:numref:`sec_approx_train` ではネガティブサンプリングを例にとります。このセクションでは、埋め込みモデルという単語を事前トレーニングするためのデータセットから始めます。データの元の形式は、トレーニング中に反復処理できるミニバッチに変換されます。

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

## データセットの読み取り

ここで使用するデータセットは [ペンツリーバンク (PTB)](https://catalog.ldc.upenn.edu/LDC99T42) です。このコーパスは、Wall Street Journal の記事からサンプリングされ、トレーニング、検証、およびテストセットに分割されています。元の形式では、テキストファイルの各行はスペースで区切られた単語の文を表します。ここでは、各単語をトークンとして扱います。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

トレーニングセットを読んだ後、コーパスのボキャブラリを作成します。このボキャブラリでは、10 回未満出現する単語は "<unk>" トークンに置き換えられます。元のデータセットには、<unk>まれな (未知の) 単語を表す "" トークンも含まれていることに注意してください。

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## サブサンプリング

テキストデータには通常、「the」、「a」、「in」などの頻度の高い単語が含まれており、非常に大きなコーパスでは数十億回出現することもあります。ただし、これらの単語はコンテキストウィンドウで多くの異なる単語と共存することが多く、有用なシグナルはほとんどありません。たとえば、コンテキストウィンドウで「チップ」という単語を考えてみましょう。直感的には、低頻度の単語「intel」との共起は、高頻度の単語「a」との共起よりも学習に有用です。さらに、大量の（頻度の高い）単語を使ったトレーニングは遅いです。したがって、単語埋め込みモデルの学習時に、高頻度の単語を*サブサンプリング* :cite:`Mikolov.Sutskever.Chen.ea.2013` にすることができます。具体的には、データセット内の索引付けされた単語 $w_i$ は確率で破棄されます。 

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

$f(w_i)$ はデータセット内の総単語数に対する単語数 $w_i$ の比率で、定数 $t$ はハイパーパラメータ (実験では $10^{-4}$) です。相対頻度 $f(w_i) > t$ が (高頻度の) 単語 $w_i$ を破棄できる場合にのみ、単語の相対頻度が高いほど、破棄される確率が高くなることがわかります。

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

次のコードスニペットは、サブサンプリングの前後の 1 文あたりのトークン数のヒストグラムをプロットします。予想通り、サブサンプリングは頻度の高い単語を落とすことで文章を大幅に短縮し、学習のスピードアップにつながります。

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

個々のトークンの場合、高頻度の単語「the」のサンプリングレートは 1/20 未満です。

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

一方、低頻度の単語「join」は完全に保持されます。

```{.python .input}
#@tab all
compare_counts('join')
```

サブサンプリング後、トークンをコーパスのインデックスにマッピングします。

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## センターワードとコンテキストワードの抽出

次の `get_centers_and_contexts` 関数は `corpus` からすべてのセンターワードとそのコンテキストワードを抽出します。コンテキストウィンドウサイズとして 1 ～ `max_window_size` の整数をランダムに一様にサンプリングします。どのセンターワードでも、そこから距離がサンプリングされたコンテキストウィンドウサイズを超えない単語は、そのコンテキストワードです。

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

次に、それぞれ 7 語と 3 語の文を 2 つ含む人工データセットを作成します。コンテキストウィンドウの最大サイズを 2 とし、すべてのセンターワードとそのコンテキストワードを出力します。

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB データセットで学習する場合、コンテキストウィンドウの最大サイズを 5 に設定します。次の例では、データセット内のすべての中心語とその文脈語が抽出されます。

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 負サンプリング

近似トレーニングには負のサンプリングを使用します。定義済みの分布に従ってノイズワードをサンプリングするために、次の `RandomGenerator` クラスを定義します。このクラスでは、(正規化されていない可能性もある) サンプリング分布が引数 `sampling_weights` を介して渡されます。

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

たとえば、標本確率 $P(X=1)=2/9, P(X=2)=3/9$ と $P(X=3)=4/9$ をもつインデックス 1、2、3 のうち、次のように 10 個の確率変数 $X$ を描画できます。

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

センターワードとコンテキストワードのペアに対して、`K` (実験では 5 個) のノイズワードをランダムにサンプリングします。word2vec 論文の提案によると、ノイズワード $w$ のサンプリング確率 $P(w)$ は、辞書の相対周波数を 0.75 :cite:`Mikolov.Sutskever.Chen.ea.2013` の累乗値に設定しています。

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## トレーニング例をミニバッチで読み込む
:label:`subsec_word2vec-minibatch-loading`

すべてのセンターワードとそのコンテキストワードおよびサンプリングされたノイズワードが抽出されると、それらはサンプルのミニバッチに変換され、トレーニング中に繰り返しロードできます。 

ミニバッチでは、$i^\mathrm{th}$ の例にはセンターワードとその $n_i$ コンテキストワード、および $m_i$ ノイズワードが含まれています。コンテキストウィンドウのサイズが異なるため、$n_i+m_i$ は $i$ によって異なります。したがって、例ごとに `contexts_negatives` 変数にコンテキストワードとノイズワードを連結し、連結長が $\max_i n_i+m_i$ (`max_len`) に達するまでゼロをパディングします。損失の計算でパディングを除外するために、マスク変数 `masks` を定義します。`masks` の要素と `contexts_negatives` の要素の間には 1 対 1 の対応関係があります。`masks` のゼロ (そうでなければ 1) は `contexts_negatives` のパディングに対応します。 

ポジティブな例とネガティブな例を区別するために、`contexts_negatives` のコンテキストワードとノイズワードを `labels` 変数で分けます。`masks` と同様に、`labels` の要素と `contexts_negatives` の要素の間には 1 対 1 の対応関係があります。`labels` の 1 (そうでなければゼロ) は `contexts_negatives` の文脈語 (正の例) に対応します。 

上記の考え方は、次の `batchify` 関数に実装されています。入力 `data` は、長さがバッチサイズに等しいリストです。各要素は、センターワード `center`、コンテキストワード `context`、ノイズワード `negative` で構成される例です。この関数は、mask 変数を含めるなど、学習中の計算のために読み込むことができるミニバッチを返します。

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

2 つの例のミニバッチを使用してこの関数をテストしてみましょう。

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## すべてのものをまとめる

最後に、PTB データセットを読み取り、データイテレータとボキャブラリを返す `load_data_ptb` 関数を定義します。

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

データイテレータの最初のミニバッチを出力してみましょう。

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## [概要

* 頻度の高い単語はトレーニングにあまり役に立たないかもしれません。トレーニングを高速化するためにサブサンプリングできます。
* 計算効率を上げるため、サンプルをミニバッチでロードします。他の変数を定義して、パディングと非パディングを区別し、ポジティブな例とネガティブな例を区別することができます。

## 演習

1. サブサンプリングを使用しない場合、このセクションのコードの実行時間はどのように変化しますか。
1. `RandomGenerator` クラスは `k` のランダムサンプリング結果をキャッシュします。`k` を他の値に設定して、データの読み込み速度にどのような影響があるかを確認します。
1. この節のコードで他にどのようなハイパーパラメータがデータ読み込み速度に影響を与えますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:

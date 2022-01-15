# 言語モデルとデータセット
:label:`sec_language_model`

:numref:`sec_text_preprocessing` では、テキストデータをトークンにマッピングする方法を説明しています。トークンは、単語や文字などの一連の離散観測として見ることができます。長さが $T$ のテキストシーケンスのトークンが $x_1, x_2, \ldots, x_T$ であると仮定します。テキストシーケンスでは、$x_t$ ($1 \leq t \leq T$) をタイムステップ $t$ の観測値またはラベルと見なすことができます。このようなテキストシーケンスを考えると、*言語モデル*の目標は、シーケンスの結合確率を推定することです。 

$$P(x_1, x_2, \ldots, x_T).$$

言語モデルは非常に便利です。たとえば、理想的な言語モデルでは、一度に 1 つのトークンを描画するだけで、自然テキストを単独で生成できます $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$。タイプライターを使う猿と全く違って、そのようなモデルから出てくるすべてのテキストは自然言語、例えば英語のテキストとして渡されます。さらに、前のダイアログフラグメントでテキストをコンディショニングするだけで、意味のあるダイアログを生成するのに十分です。明らかに、文法的にわかりやすいコンテンツを生成するのではなく、テキストを「理解」する必要があるため、このようなシステムの設計にはまだほど遠いです。 

とはいえ、言語モデルは限られた形でも非常に役に立ちます。たとえば、「スピーチを認識する」と「素敵なビーチを破壊する」というフレーズは非常によく似ています。これは音声認識にあいまいさを引き起こす可能性があり、2 番目の翻訳を異様なものとして拒否する言語モデルによって簡単に解決できます。同様に、文書要約アルゴリズムでは、「犬が犬を噛む」よりも「犬が男に噛む」の方がはるかに頻繁であること、または「おばあちゃんを食べたい」というのはかなり不穏な発言であることを知っておく価値があります。「食べたい、おばあちゃん」ははるかに良性です。 

## 言語モデルの学習

明白な疑問は、ドキュメント、あるいは一連のトークンをどのようにモデル化すべきかということです。テキストデータを単語レベルでトークン化するとします。:numref:`sec_sequence` で配列モデルに適用した解析に頼ることができます。基本的な確率ルールを適用することから始めましょう。 

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

たとえば、4 つの単語を含むテキストシーケンスの確率は次のようになります。 

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

言語モデルを計算するには、単語の確率と、前の数単語から与えられた単語の条件付き確率を計算する必要があります。このような確率は本質的に言語モデルのパラメータです。 

ここでは、トレーニングデータセットは、Wikipedia のすべてのエントリ [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg)、Web に投稿されたすべてのテキストなど、大きなテキストコーパスであると仮定します。単語の確率は、トレーニングデータセット内の特定の単語の相対的な単語頻度から計算できます。たとえば、推定値 $\hat{P}(\text{deep})$ は、単語「深い」で始まる文の確率として計算できます。少し精度の低いアプローチは、「deep」という単語をすべてカウントし、それをコーパス内の総単語数で割るというものです。これは、特に頻繁な単語ではかなりうまく機能します。先に進むと、見積もりを試みることができます 

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

$n(x)$ と $n(x, x')$ は、それぞれシングルトンと連続する単語ペアの出現回数です。残念ながら、単語ペアの確率を推定することは、「ディープラーニング」の発生頻度がはるかに低いため、やや困難です。特に、一部の珍しい単語の組み合わせでは、正確な推定値を得るのに十分な出現箇所を見つけるのが難しい場合があります。3単語の組み合わせ以降では、事態は悪化の一途をたどります。私たちのデータセットには見られない、もっともらしい3語の組み合わせがたくさんあります。このような単語の組み合わせをゼロ以外のカウントに代入するソリューションを提供しない限り、言語モデルでは使用できません。データセットが小さい場合や、単語が非常にまれな場合は、その中の単語が1つでも見つからない可能性があります。 

一般的な方法は、何らかの形の*ラプラス平滑化*を実行することです。解決策は、すべての計数に小さな定数を加えることです。学習セットに含まれる単語の総数を $n$ で表し、固有な単語の数を $m$ で表します。このソリューションは、シングルトン、例えば 

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

ここで $\epsilon_1,\epsilon_2$ と $\epsilon_3$ はハイパーパラメータです。$\epsilon_1$ を例にとると、$\epsilon_1 = 0$ では平滑化は適用されず、$\epsilon_1$ が正の無限大に近づくと $\hat{P}(x)$ は一様確率 $1/m$ に近づきます。上記は、:cite:`Wood.Gasthaus.Archambeau.ea.2011` を他の手法で実現できることのかなり原始的な変形です。 

残念ながら、このようなモデルは次の理由からかなり早く扱いにくくなります。まず、すべてのカウントを保存する必要があります。第二に、これは単語の意味を完全に無視します。たとえば、「cat」と「cat」は関連するコンテキストで発生する必要があります。このようなモデルを他のコンテキストに合わせて調整することは非常に困難ですが、ディープラーニングベースの言語モデルはこれを考慮に入れるのに適しています。最後に、長い単語シーケンスはほぼ確実に新規であるため、以前に見た単語シーケンスの頻度を単純にカウントするモデルは、そこではうまく機能しません。 

## マルコフモデルと$n$グラム

ディープラーニングに関連するソリューションについて説明する前に、もう少し用語と概念が必要です。:numref:`sec_sequence` でのマルコフモデルについての議論を思い出してください。これを言語モデリングに当てはめてみましょう。$P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$ の場合、系列にまたがる分布は 1 次のマルコフ特性を満たします。次数が高いほど、依存関係が長くなります。これにより、シーケンスのモデル化に適用できる近似がいくつか得られます。 

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

1 つ、2 つ、および 3 つの変数を含む確率式は、通常、それぞれ*unigram*、*bigram*、*trigram* モデルと呼ばれます。以下では、より良いモデルを設計する方法を学びます。 

## 自然言語統計学

これが実際のデータでどのように機能するか見てみましょう。:numref:`sec_text_preprocessing` で導入されたタイムマシンデータセットに基づいて語彙を構築し、最も頻出する上位10個の単語を出力します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines 
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

ご覧のとおり、（**最も人気のある単語は**）実際には見るのはかなり退屈です。それらはしばしば (***ストップワード***) と呼ばれ、除外されます。それにもかかわらず、それらにはまだ意味があり、私たちはまだそれらを使用します。その上、頻度という言葉がかなり急速に減衰することは明らかです。$10^{\mathrm{th}}$ の最頻単語は、最も一般的な単語と同じくらい $1/5$ 未満です。より良いアイデアを得るために、[**単語の頻度の数字をプロットする**]。

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

ここでは、非常に基本的なことに取り組んでいます。頻度という言葉は、明確に定義された方法で急速に減衰します。最初のいくつかの単語を例外として扱った後、残りの単語はすべて対数対数プロットでほぼ直線に従います。これは、単語が*Zipfの法則*を満たすことを意味し、$i^\mathrm{th}$の頻度$n_i$が最も多い単語は次のようになります。 

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

これは次のものと同等です 

$$\log n_i = -\alpha \log i + c,$$

$\alpha$ は分布を表す指数で、$c$ は定数です。統計を数えて平滑化することで単語をモデル化したい場合、これはすでに一時停止しているはずです。結局のところ、まれな単語としても知られている尾の頻度を大幅に過大評価します。しかし、[**バイグラム、トリグラムなどの他の単語の組み合わせについてはどうですか？バイグラム周波数がユニグラム周波数と同じように動作するかどうかを見てみましょう。**]

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

ここで注目すべきことが1つあります。最も頻繁に使用される10の単語のペアのうち、9つは両方のストップワードで構成され、実際の本に関連するのは「時間」だけです。さらに、トライグラム周波数が同じように動作するかどうかを見てみましょう。

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

最後に、ユニグラム、バイグラム、トライグラムの3つのモデルの中で [**トークンの頻度を可視化**] してみましょう。

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

この数字は、いくつかの理由で非常にエキサイティングです。まず、ユニグラム単語以外では、単語のシーケンスもZipfの法則に従っているように見えますが、:eqref:`eq_zipf_law`では指数 $\alpha$ はシーケンスの長さに応じて小さくなります。第二に、明確な$n$グラムの数はそれほど多くありません。これは、言語にかなりの構造があるという希望を与えてくれます。第3に、$n$グラムの多くがごくまれにしか発生しないため、ラプラス平滑化は言語モデリングには適していません。代わりに、ディープラーニングベースのモデルを使用します。 

## ロングシーケンスデータの読み込み

シーケンスデータは本質的にシーケンシャルなので、処理の問題に対処する必要があります。:numref:`sec_sequence` では、かなりアドホックな方法でこれを行いました。シーケンスが長すぎてモデルで一度に処理できない場合は、そのようなシーケンスを分割して読み込みたいと思うかもしれません。ここで、一般的な戦略について説明しましょう。モデルを導入する前に、ニューラルネットワークを使用して言語モデルをトレーニングすると仮定します。この場合、ネットワークは定義済みの長さ (たとえば $n$ タイムステップ) を持つシーケンスのミニバッチを一度に処理します。ここで問題となるのは、[**フィーチャとラベルのミニバッチをランダムに読み取る**] 

まず、テキストシーケンスは*The Time Machine* ブック全体のように任意に長くなる可能性があるため、このような長いシーケンスを同じタイムステップ数のサブシーケンスに分割できます。ニューラルネットワークをトレーニングすると、そのようなサブシーケンスのミニバッチがモデルに入力されます。ネットワークが $n$ のタイムステップのサブシーケンスを一度に処理するとします。:numref:`fig_timemachine_5gram` は、元のテキストシーケンスから部分シーケンスを取得するさまざまな方法をすべて示しています。ここで、$n=5$ と各タイムステップでのトークンは 1 文字に対応します。初期位置を示す任意のオフセットを選択できるため、ある程度の自由度があることに注意してください。 

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

したがって、:numref:`fig_timemachine_5gram`からどれを選ぶべきですか？実際、それらはすべて同じように優れています。ただし、オフセットを 1 つだけ選択すると、ネットワークの学習に使用できるすべてのサブシーケンスのカバレッジが制限されます。したがって、ランダムオフセットから始めてシーケンスを分割し、*coverage* と*randomness* の両方を得ることができます。以下では、両方でこれを実現する方法を説明します。
*ランダムサンプリング* および*シーケンシャルパーティショニング* ストラテジー

### ランダムサンプリング

(**ランダムサンプリングでは、各例は元の長いシーケンスに任意に捕捉された部分列です**) 反復中の2つの隣接したランダムミニバッチの部分列は、必ずしも元の配列に隣接しているとは限りません。言語モデリングの目標は、これまで見てきたトークンに基づいて次のトークンを予測することです。したがって、ラベルは 1 トークン分シフトされた元のシーケンスです。 

次のコードは、その都度、データからミニバッチをランダムに生成します。ここで、引数 `batch_size` は各ミニバッチ内のサブシーケンスの例数を指定し、`num_steps` は各サブシーケンスの定義済みタイムステップ数です。

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

[**0 から 34.までのシーケンスを手動で生成する**] バッチサイズとタイムステップ数はそれぞれ 2 と 5 であると仮定します。つまり、$\lfloor (35 - 1) / 5 \rfloor= 6$ フィーチャとラベルのサブシーケンスのペアを生成できます。ミニバッチサイズが 2 の場合、ミニバッチは 3 つしか得られません。

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### シーケンシャル・パーティショニング

元のシーケンスのランダムサンプリングに加えて、[**反復中に隣接する 2 つのミニバッチのサブシーケンスが元のシーケンスに隣接していることを確認することもできます。**] このストラテジーは、ミニバッチを反復するときに分割されたサブシーケンスの順序を保持するため、シーケンシャルと呼ばれます。パーティショニング。

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

同じ設定を使って、シーケンシャル・パーティショニングによって読み込まれたサブシーケンスの [**機能 `X` とラベル `Y` をミニバッチごとに印刷**] してみましょう。反復処理中の隣接する 2 つのミニバッチの部分列は、実際には元のシーケンス上で隣接していることに注意してください。

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

ここで、上記の 2 つのサンプリング関数をクラスにラップして、後でデータイテレータとして使用できるようにします。

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

[**最後に、データイテレータとボキャブラリの両方を返す関数 `load_data_time_machine` を定義します**]、:numref:`sec_fashion_mnist` で定義されている `d2l.load_data_fashion_mnist` のように `load_data` の接頭辞を持つ他の関数と同様に使用できます。

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## [概要

* 言語モデルは自然言語処理の鍵となります。
* $n$-gram は、依存関係を切り捨てることで長いシーケンスを扱うのに便利なモデルです。
* 長いシーケンスは、非常にまれにしか発生しない、またはまったく発生しないという問題に悩まされます。
* ジップの法則は、ユニグラムだけでなく他の$n$グラムの単語分布にも適用されます。
* 構造はたくさんありますが、頻度の低い単語の組み合わせをラプラス平滑化によって効率的に処理するには頻度が不十分です。
* 長いシーケンスを読み取るための主な選択肢は、ランダム・サンプリングとシーケンシャル・パーティショニングです。後者は、反復処理中に隣接する 2 つのミニバッチからのサブシーケンスが元のシーケンスで隣接していることを保証できます。

## 演習

1. トレーニングデータセットに $100,000$ 語があるとします。4グラムはどのくらいのワード周波数とマルチワード隣接周波数を格納する必要がありますか？
1. 対話をどのようにモデル化しますか？
1. ユニグラム、バイグラム、およびトライグラムの Zipf の法則の指数を推定します。
1. 長いシーケンスのデータを読み取るには、他にどのような方法が考えられますか？
1. 長いシーケンスの読み込みに使用するランダムオフセットについて考えてみましょう。
    1. ランダムなオフセットを設定するのが良いのはなぜですか？
    1. ドキュメント上のシーケンスにわたって完全に一様な分布が得られるのでしょうか？
    1. 物事をより均一にするためには何をしなければなりませんか？
1. シーケンスの例を完全な文にしたいのであれば、ミニバッチサンプリングではどのような問題が生じますか？この問題を解決するにはどうすればいいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

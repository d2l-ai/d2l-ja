# テキストの前処理
:label:`sec_text_preprocessing`

シーケンスデータの統計ツールと予測の課題を見直し、評価しました。このようなデータにはさまざまな形式があります。特に、本書の多くの章で取り上げるように、テキストはシーケンスデータの最も一般的な例の 1 つです。たとえば、記事は単純に単語のシーケンス、または一連の文字として表示できます。シーケンスデータを使用した今後の実験を容易にするために、このセクションではテキストの一般的な前処理手順について説明します。通常、これらの手順は次のとおりです。 

1. テキストを文字列としてメモリに読み込みます。
1. 文字列をトークン (単語や文字など) に分割します。
1. 分割されたトークンを数値インデックスにマッピングする語彙の表を作成します。
1. テキストを数値インデックスのシーケンスに変換して、モデルで簡単に操作できるようにします。

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## データセットの読み取り

はじめに、H.G. Wellsの [*The Time Machine*](http://www.gutenberg.org/ebooks/35) からテキストをロードします。これは30000語強のかなり小さなコーパスですが、説明したいのはこれで問題ありません。より現実的なドキュメントコレクションには、何十億もの単語が含まれています。次の関数 (**データセットをテキスト行のリストに読み込む**) で、各行は文字列です。簡単にするために、ここでは句読点と大文字の使用は無視します。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## トークン化

次の `tokenize` 関数はリスト (`lines`) を入力として受け取り、各要素はテキストシーケンス (テキスト行など) です。[**各テキストシーケンスはトークンのリストに分割されます**]。*token* はテキストの基本単位です。最後に、トークン・リストのリストが返され、各トークンは文字列になります。

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## ボキャブラリー

トークンの文字列型は、数値入力を受け取るモデルでは不便です。ここで [**辞書を作って、文字列トークンを 0** から始まる数値インデックスにマップするために、*vocabulary* と呼ばれることもあります]。そのためには、まずトレーニングセットのすべてのドキュメント、つまり*corpus* で一意のトークンをカウントし、その頻度に応じて一意の各トークンに数値インデックスを割り当てます。まれにしか出現しないトークンは、複雑さを軽減するために削除されることがよくあります。コーパスに存在しないトークン、または削除されたトークンは、未知の特殊なトークン「<unk>」にマッピングされます。オプションで、<pad>パディングを表す「」、<bos>シーケンスの先頭を示す「」、<eos>シーケンスの最後を表す「」など、予約されたトークンのリストを追加します。

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

タイムマシンデータセットをコーパスとして [**語彙を構築**] します。次に、最初の数個の頻出トークンとそのインデックスを出力します。

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

これで (**各テキスト行を数値インデックスのリストに変換する**) ことができます。

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## すべてのものをまとめる

上記の関数を使って [**すべてを`load_corpus_time_machine` 関数にパッケージ化**] すると、トークンインデックスのリストである `corpus` と、タイムマシンコーパスの語彙である `vocab` が返されます。ここで行った変更は次のとおりです。(i) 後のセクションで学習を簡略化するために、テキストを単語ではなく文字にトークン化する。(ii) `corpus` は、Time Machine データセットの各テキスト行が必ずしも文や段落ではないため、トークンリストのリストではなく、単一のリストである。

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## [概要

* テキストはシーケンスデータの重要な形式です。
* テキストを前処理するために、通常はテキストをトークンに分割し、トークン文字列を数値インデックスにマッピングするボキャブラリを構築し、テキストデータをトークンインデックスに変換してモデルが操作できるようにします。

## 演習

1. トークン化は重要な前処理ステップです。言語によって異なります。テキストをトークン化するためによく使われる別の3つの方法を探してみます。
1. このセクションの実験では、テキストを単語にトークン化し、`Vocab` インスタンスの `min_freq` 引数を変化させます。これは語彙の大きさにどのように影響しますか？

[Discussions](https://discuss.d2l.ai/t/115)

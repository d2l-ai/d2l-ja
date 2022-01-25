# 自然言語推論とデータセット
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment` では、センチメント分析の問題について議論しました。このタスクの目的は、1 つのテキストシーケンスを、一連のセンチメント極性などの定義済みカテゴリに分類することです。ただし、ある文を別の文から推論できるかどうかを判断する必要がある場合や、意味的に同等の文を特定して冗長性を排除する必要がある場合は、テキストシーケンスの分類方法を知るだけでは不十分です。代わりに、テキストシーケンスのペアを推論できる必要があります。 

## 自然言語推論

*自然言語推論*は*仮説*であるかどうかを研究する
どちらもテキストシーケンスである*前提*から推測できます。言い換えれば、自然言語推論は、一対のテキストシーケンス間の論理的な関係を決定するということです。このような関係は、通常、次の3つのタイプに分類されます。 

* *含意*：仮説は前提から推測できます。
* *矛盾*：仮説の否定は前提から推測できます。
* *ニュートラル*: その他すべてのケース。

自然言語推論は、認識するテキスト包含タスクとしても知られています。たとえば、仮説における「愛情を示す」は、前提の「抱き合う」ことから推測できるため、次のペアは*含意*とラベル付けされます。 

> 前提：2人の女性が抱き合っている。 

> 仮説：2人の女性が愛情を示している。 

以下は、「コーディング例を実行する」は「スリープ」ではなく「スリープしていない」ことを示す「矛盾」の例です。 

> 前提:ある男性がDive into Deep Learningのコーディング例を実行しています。 

> 仮説：男は眠っている。 

3つ目の例は*中立性*の関係を示しています。「有名」も「有名でない」のどちらも「私たちのために演じている」という事実から推測できないからです。  

> Premise: ミュージシャンが私たちのために演奏してくれます。 

> 仮説：ミュージシャンは有名だ。 

自然言語推論は、自然言語を理解する上で中心的なトピックとなっています。情報検索からオープンドメインの質問応答まで、幅広い用途に利用できます。この問題を研究するために、まず一般的な自然言語推論ベンチマークデータセットを調査することから始めます。 

## スタンフォード大学自然言語推論 (SNLI) データセット

スタンフォード自然言語推論 (SNLI) コーパスは、ラベル付けされた 500000 以上の英文のペアを集めたものです :cite:`Bowman.Angeli.Potts.ea.2015`。抽出された SNLI データセットをダウンロードし、パス `../data/snli_1.0` に格納します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### データセットの読み取り

元のSNLIデータセットには、実験で本当に必要な情報よりもはるかに豊富な情報が含まれています。そこで、データセットの一部のみを抽出し、前提、仮説、およびそのラベルのリストを返す関数 `read_snli` を定義します。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

ここで、前提と仮説の最初の3組とそのラベル (「0」、「1」、「2」はそれぞれ「含意」、「矛盾」、「中立」に対応) を印刷します。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

トレーニングセットには約550000ペア、テストセットには約10000ペアがあります。以下に、「含意」、「矛盾」、「中立」の 3 つのラベルが、学習セットとテストセットの両方でバランスが取れていることを示しています。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### データセットをロードするクラスの定義

以下では、Gluon の `Dataset` クラスから継承して SNLI データセットをロードするためのクラスを定義します。クラスコンストラクターの引数 `num_steps` は、シーケンスの各ミニバッチが同じ形状になるように、テキストシーケンスの長さを指定します。言い換えると、最初の `num_steps` <pad>より後のトークンは切り捨てられ、特殊トークン「」はその長さが `num_steps` になるまで短いシーケンスに追加されます。`__getitem__` 関数を実装することで、インデックス `idx` の前提、仮説、ラベルに任意にアクセスすることができます。

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### すべてのものをまとめる

これで、関数 `read_snli` と `SNLIDataset` クラスを呼び出して SNLI データセットをダウンロードし、トレーニングセットとテストセットの両方の `DataLoader` インスタンスを、トレーニングセットのボキャブラリとともに返すことができます。トレーニングセットから構築された語彙をテストセットの語彙として使用しなければならないことは注目に値します。その結果、テストセットからの新しいトークンは、トレーニングセットでトレーニングされたモデルには不明になります。

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

ここでは、バッチサイズを 128、シーケンス長を 50 に設定し、`load_data_snli` 関数を呼び出してデータイテレータとボキャブラリを取得します。次に、語彙サイズを印刷します。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

今度は、最初のミニバッチの形状を印刷します。センチメント分析とは対照的に、前提と仮説のペアを表す 2 つの入力 `X[0]` と `X[1]` があります。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## [概要

* 自然言語推論は、仮説が前提から推論できるかどうかを研究します。仮説は両方ともテキストシーケンスです。
* 自然言語推論では、前提と仮説の関係には、含意、矛盾、中立が含まれます。
* スタンフォード自然言語推論 (SNLI) コーパスは、自然言語推論の一般的なベンチマークデータセットです。

## 演習

1. 機械翻訳は、出力翻訳とグラウンドトゥルース翻訳の間の表面的な $n$ グラム一致に基づいて長い間評価されてきました。自然言語推論を用いて機械翻訳の結果を評価する手段をデザインできますか？
1. ハイパーパラメータを変更して語彙数を減らすにはどうすればいいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:

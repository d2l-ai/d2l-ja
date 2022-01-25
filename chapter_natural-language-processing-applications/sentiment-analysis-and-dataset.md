# センチメント分析とデータセット
:label:`sec_sentiment`

オンラインのソーシャルメディアとレビュープラットフォームの急増に伴い、大量の意見のあるデータが記録され、意思決定プロセスをサポートする大きな可能性を秘めています。
*センチメント分析*
製品レビュー、ブログコメント、フォーラムディスカッションなど、生成したテキストに含まれる人々の感情を調査します。政治（政策に対する世論の分析など）、財務（市場のセンチメント分析など）、マーケティング（商品調査やブランドマネジメントなど）など、多岐にわたる分野に幅広く応用されています。 

センチメントは個別の極性またはスケール (ポジティブとネガティブなど) に分類できるため、センチメント分析は、さまざまな長さのテキストシーケンスを固定長のテキストカテゴリに変換するテキスト分類タスクと見なすことができます。この章では、スタンフォード大学の [大規模映画レビューデータセット](https://ai.stanford.edu/~amaas/data/sentiment/) をセンチメント分析に使用します。トレーニングセットとテストセットで構成され、IMDbからダウンロードされた25000件の映画レビューが含まれています。どちらのデータセットにも、「ポジティブ」と「ネガティブ」のラベルが同数あり、センチメントの極性が異なることを示しています。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  データセットの読み取り

まず、この IMDb レビューデータセットをダウンロードし、パス `../data/aclImdb` に展開します。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

次に、トレーニングデータセットとテストデータセットを読みます。各例はレビューとそのラベルです。1 は「正」、0 は「負」を表します。

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

## データセットの前処理

各単語をトークンとして扱い、5 回未満出現する単語を除外して、トレーニングデータセットから語彙を作成します。

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

トークン化後、レビューの長さのヒストグラムをトークン単位でプロットしてみましょう。

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

予想通り、レビューの長さはさまざまです。このようなレビューのミニバッチを毎回処理するために、各レビューの長さを切り捨てと埋め込み付きで 500 に設定しました。これは :numref:`sec_machine_translation` の機械翻訳データセットの前処理ステップと同様です。

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## データイテレータの作成

これでデータイテレータを作成できるようになりました。各反復で、サンプルのミニバッチが返されます。

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## すべてのものをまとめる

最後に、上記の手順を `load_data_imdb` 関数にまとめます。トレーニングデータとテストデータイテレータ、および IMDb レビューデータセットのボキャブラリを返します。

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## [概要

* センチメント分析は、生成したテキストに含まれる人々のセンチメントを調査します。これは、さまざまな長さのテキストシーケンスを変換するテキスト分類問題と見なされています。
固定長のテキストカテゴリに変換します。
* 前処理後、スタンフォード大学の大規模な映画レビューデータセット (IMDb レビューデータセット) をボキャブラリ付きのデータイテレーターに読み込むことができます。

## 演習

1. トレーニングセンチメント分析モデルを高速化するために、このセクションのどのハイパーパラメータを変更できますか？
1. センチメント分析のために [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) のデータセットをデータイテレーターとラベルに読み込む関数を実装できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:

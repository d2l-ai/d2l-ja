# 単語の類似性と類推
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining` では、小さなデータセットで word2vec モデルをトレーニングし、入力された単語に対して意味的に類似する単語を見つけるためにそれを適用しました。実際には、大規模なコーパスで事前学習された単語ベクトルは、:numref:`chap_nlp_app` で後述するダウンストリームの自然言語処理タスクに適用できます。大規模コーパスからの事前学習済みの単語ベクトルの意味論をわかりやすく説明するために、単語の類似性と類推のタスクに適用してみましょう。

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

## 事前学習済みの単語ベクトルの読み込み

次元 50、100、300 の事前学習済みの GLOVE 埋め込みを以下に示します。これらは [GloVe website](https://nlp.stanford.edu/projects/glove/) からダウンロードできます。事前学習済みの FastText 埋め込みは、複数の言語で利用できます。ここでは [fastText website](https://fasttext.cc/) からダウンロードできる英語版 (300 次元の「wiki.en」) を考えてみます。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

これらの事前学習済みの GLOVE および FastText 埋め込みを読み込むために、次の `TokenEmbedding` クラスを定義します。

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

以下では、50 次元 GLOVE 埋め込み (Wikipedia のサブセットで事前学習済み) を読み込みます。`TokenEmbedding` インスタンスを作成するときに、指定された埋め込みファイルがまだダウンロードされていない場合はダウンロードする必要があります。

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

ボキャブラリサイズを出力します。語彙には40万語（トークン）と特別な未知のトークンが含まれています。

```{.python .input}
#@tab all
len(glove_6b50d)
```

ボキャブラリ内の単語のインデックスを取得でき、その逆も可能です。

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 事前学習済みの単語ベクトルの適用

読み込まれた GLOVE ベクトルを使用して、次の単語の類似性と類推のタスクに適用することで、そのセマンティクスを示します。 

### 単語の類似性

:numref:`subsec_apply-word-embed` と同様に、単語ベクトル間の余弦類似度に基づいて入力単語の意味的に類似する単語を見つけるために、次の `knn` ($k$-最近傍) 関数を実装します。

```{.python .input}
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

次に、`TokenEmbedding` インスタンス `embed` の事前学習済みの単語ベクトルを使用して、類似する単語を検索します。

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

`glove_6b50d` の事前学習済みの単語ベクトルの語彙には 400000 語と特別な不明なトークンが含まれています。入力された単語と未知のトークンを除いて、この語彙の中で、単語「チップ」に最も意味的に類似した3つの単語を見つけよう。

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

以下は、「赤ちゃん」と「美しい」に似た単語を出力します。

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 単語アナロジー

類似する単語を見つける以外に、単語ベクトルを単語類推タスクに適用することもできます。たとえば、「男」:「女性」::「息子」:「娘」は単語の類推の形です。「男」は「女」、「息子」は「娘」です。具体的には、アナロジー完了タスクという単語は次のように定義できます。単語のアナロジー $a : b :: c : d$ の場合、最初の 3 つの単語 $a$、$b$、$c$ を指定すると $d$ が見つかります。$\text{vec}(w)$ で単語 $w$ のベクトルを表します。類推を完了するために、$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$の結果に最も類似したベクトルを持つ単語を見つけます。

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

読み込まれた単語ベクトルを使って、「男女」の類推を検証してみましょう。

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

「首都-国」の例えを以下に示します。「北京」：「中国」::「東京」:「日本」。これは、事前学習済みの単語ベクトルのセマンティクスを示しています。

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

「bad」:「worst」::「big」:「biggest」などの「形容詞と最上級の形容詞」のアナロジーでは、事前学習された単語ベクトルが構文情報を取得できることがわかります。

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

事前学習済みの単語ベクトルで捕捉された過去形の概念を示すために、「現在形-過去形」アナロジー「do」:「did」::「go」:「went」を使用して構文をテストできます。

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## [概要

* 実際には、大規模なコーパスで事前学習された単語ベクトルは、下流の自然言語処理タスクに適用できます。
* 事前学習済みの単語ベクトルは、単語の類似性タスクと類推タスクに適用できます。

## 演習

1. `TokenEmbedding('wiki.en')` を使用して FastText の結果をテストします。
1. 語彙が極端に大きい場合、類似する単語を見つけたり、単語のアナロジーをより早く完成させるにはどうすればよいでしょうか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:

# BERT を事前トレーニングするためのデータセット
:label:`sec_bert-dataset`

:numref:`sec_bert` で実装された BERT モデルを事前学習させるには、マスク言語モデリングと次の文の予測という 2 つの事前学習タスクを容易にする理想的な形式でデータセットを生成する必要があります。一方では、元の BERT モデルは 2 つの巨大なコーパス bookCorpus と英語版ウィキペディア (:numref:`subsec_bert_pretraining_tasks` 参照) を連結した上で事前にトレーニングされており、この本を読むほとんどの読者にとって実行が難しくなっています。一方、市販の事前学習済みの BERT モデルは、医療などの特定の領域のアプリケーションには適合しない場合があります。そのため、カスタマイズされたデータセットで BERT を事前トレーニングすることが一般的になっています。BERT 事前トレーニングのデモンストレーションを容易にするために、より小さなコーパス WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016` を使用します。 

:numref:`sec_word2vec_data` で word2vec の事前学習に使用された PTB データセットと比較すると、WikiText-2 (i) 元の句読点を保持し、次の文の予測に適している。(ii) 元の大文字と小文字と数字を保持している。(iii) 倍以上大きい。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

WikiText-2 データセットでは、各行は句読点とその前のトークンの間にスペースが挿入された段落を表します。文が 2 つ以上ある段落は保持されます。文を分割するには、わかりやすくするためにピリオドのみを区切り文字として使用します。より複雑な文の分割方法については、このセクションの最後にある演習に残しておきます。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 事前学習タスク用のヘルパー関数の定義

以下では、次の文の予測とマスクされた言語モデリングという 2 つの BERT 事前学習タスクのためのヘルパー関数を実装することから始めます。これらのヘルパー関数は、生のテキストコーパスを理想的な形式のデータセットに変換して BERT を事前にトレーニングするときに呼び出されます。 

### 次文予測タスクの生成

:numref:`subsec_nsp` の説明によると、関数 `_get_next_sentence` はバイナリ分類タスクの学習例を生成します。

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

次の関数は、関数 `_get_next_sentence` を呼び出して、入力 `paragraph` から次の文を予測するためのトレーニング例を生成します。ここで `paragraph` は文のリストで、各文はトークンのリストです。引数 `max_len` は、事前学習中の BERT 入力シーケンスの最大長を指定します。

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### マスク言語モデリングタスクの生成
:label:`subsec_prepare_mlm_data`

BERT 入力シーケンスからマスク言語モデリングタスクのトレーニング例を生成するために、次の `_replace_mlm_tokens` 関数を定義します。その入力では、`tokens` は BERT 入力シーケンスを表すトークンのリスト、`candidate_pred_positions` は特殊トークン (マスク言語モデリングタスクでは特殊トークンは予測されない) を除く BERT 入力シーケンスのトークンインデックスのリスト、`num_mlm_preds` は予測数 (リコール 15%)予測するランダムなトークン）。:numref:`subsec_mlm` のマスク言語モデリングタスクの定義に従い、各予測位置で、入力は特別な「<mask>」トークンまたはランダムトークンに置き換えられるか、変更されないままになります。最後に、この関数は可能な置換後の入力トークン、予測が行われるトークンのインデックス、および予測のラベルを返します。

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

前述の `_replace_mlm_tokens` 関数を呼び出すと、次の関数は BERT 入力シーケンス (`tokens`) を入力として受け取り、入力トークンのインデックス (:numref:`subsec_mlm` で説明したトークンの置換後)、予測が行われるトークンインデックス、およびこれらのラベルインデックスのインデックスを返します。予測。

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## テキストを事前学習データセットに変換する

これで、BERT の事前トレーニング用に `Dataset` クラスをカスタマイズする準備がほぼ整いました。その前に、特殊な「<mask>」トークンを入力に追加するヘルパー関数 `_pad_bert_inputs` を定義する必要があります。引数 `examples` には、2 つの事前学習タスクに対する補助関数 `_get_nsp_data_from_paragraph` と `_get_mlm_data_from_tokens` からの出力が含まれます。

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

2 つの事前トレーニングタスクのトレーニング例を生成するためのヘルパー関数と、入力をパディングするためのヘルパー関数を組み合わせて、次の `_WikiTextDataset` クラスを BERT 事前トレーニング用の WikiText-2 データセットとしてカスタマイズします。`__getitem__ `関数を実装することで、WikiText-2 コーパスの一対の文から生成された事前訓練 (仮面言語モデリングと次文予測) の例に任意にアクセスすることができます。 

元の BERT モデルでは、ボキャブラリのサイズが 30000 :cite:`Wu.Schuster.Chen.ea.2016` のワードピース埋め込みが使用されています。WordPiece のトークン化方法は、:numref:`subsec_Byte_Pair_Encoding` の元のバイトペアエンコーディングアルゴリズムを少し変更したものです。わかりやすくするために、トークン化には `d2l.tokenize` 関数を使用します。5 回未満出現する頻度の低いトークンは除外されます。

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

`_read_wiki` 関数と `_WikiTextDataset` クラスを使って、以下の `load_data_wiki` を定義し、WikiText-2 データセットをダウンロードし、そこから事前トレーニングの例を生成します。

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

バッチサイズを 512、BERT 入力シーケンスの最大長を 64 に設定して、BERT 事前学習例のミニバッチの形状を出力します。各 BERT 入力シーケンスでは、マスク言語モデリングタスクで $10$ ($64 \times 0.15$) の位置が予測されることに注意してください。

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

最後に、語彙の大きさを見てみましょう。頻度の低いトークンを除外した後でも、PTB データセットの 2 倍以上も大きくなっています。

```{.python .input}
#@tab all
len(vocab)
```

## [概要

* PTB データセットと比較すると、WikiText-2 の日付セットは元の句読点、大文字小文字、数字を保持し、2 倍以上大きくなります。
* WikiText-2 コーパスの一対の文から生成された事前訓練 (仮面言語モデリングと次文予測) の例に任意にアクセスすることができます。

## 演習

1. 簡単にするために、ピリオドは文を分割するための唯一の区切り文字として使用されます。spaCY や NLTK など、他の文分割テクニックを試してみてください。NLTKを例に挙げてみましょう。最初に NLTK をインストールする必要があります:`pip install nltk`。このコードでは、最初に `import nltk` です。次に、Punkt センテンストークナイザー `nltk.download('punkt')` をダウンロードします。`sentences = 'のような文を分割するなんてすごい！なんでだめなの？`, invoking `nltk.tokenize.sent_tokenize (センテンス) ` will return a list of two sentence strings: ` ['これは素晴らしい！'、'どうしてですか？']`。
1. 頻度の低いトークンを除外しない場合、語彙のサイズはどれくらいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:

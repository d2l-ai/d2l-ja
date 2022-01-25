# 自然言語推論:BERT の微調整
:label:`sec_natural-language-inference-bert`

この章の前のセクションでは、SNLI データセット (:numref:`sec_natural-language-inference-and-dataset` で説明) に対する自然言語推論タスクのための注意ベースのアーキテクチャ (:numref:`sec_natural-language-inference-attention`) を設計しました。ここで、BERT を微調整して、このタスクを再検討します。:numref:`sec_finetuning-bert` で説明したように、自然言語推論はシーケンスレベルのテキストペア分類問題であり、:numref:`fig_nlp-map-nli-bert` に示すように、BERT の微調整には MLP ベースのアーキテクチャを追加するだけで済みます。 

![This section feeds pretrained BERT to an MLP-based architecture for natural language inference.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

このセクションでは、事前学習済みの Small バージョンの BERT をダウンロードし、SNLI データセットの自然言語推論用に微調整します。

```{.python .input}
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## 事前学習済みの BERT

:numref:`sec_bert-dataset` と :numref:`sec_bert-pretraining` で WikiText-2 データセットで BERT を事前トレーニングする方法を説明しました (元の BERT モデルはもっと大きなコーパスで事前トレーニングされています)。:numref:`sec_bert-pretraining` で説明したように、元の BERT モデルには数億個のパラメータがあります。以下では、事前学習済みの BERT の 2 つのバージョンを提供しています。「bert.base」は、微調整に多くの計算リソースを必要とする元の BERT ベースモデルとほぼ同じ大きさで、「bert.small」はデモンストレーションを容易にする小さなバージョンです。

```{.python .input}
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

いずれの事前学習済みの BERT モデルにも、語彙セットを定義する「vocab.json」ファイルと、事前学習済みパラメーターの「pretrained.params」ファイルが含まれます。事前学習済みの BERT パラメーターをロードするために、次の `load_pretrained_model` 関数を実装します。

```{.python .input}
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_layers, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

ほとんどのマシンでのデモンストレーションを容易にするために、このセクションでは事前学習済みの BERT の small バージョン (「bert.small」) をロードして微調整します。この演習では、より大きな「bert.base」を微調整してテストの精度を大幅に向上させる方法を示します。

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

## BERT を微調整するためのデータセット

SNLI データセットに対するダウンストリームタスクの自然言語推論のために、カスタマイズされたデータセットクラス `SNLIBERTDataset` を定義します。各例で、前提と仮説はテキストシーケンスのペアを形成し、:numref:`fig_bert-two-seqs` に示すように 1 つの BERT 入力シーケンスにパックされます。:numref:`subsec_bert_input_rep` を思い出してください。セグメント ID は、BERT 入力シーケンスの前提と仮説を区別するために使用されます。定義済みの BERT 入力シーケンスの最大長 (`max_len`) では、入力テキストペアのうち長い方の最後のトークンは `max_len` に達するまで削除され続けます。BERT の微調整のための SNLI データセットの生成を高速化するために、4 つのワーカープロセスを使用してトレーニングまたはテストの例を並行して生成します。

```{.python .input}
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

SNLI データセットをダウンロードした後、`SNLIBERTDataset` クラスをインスタンス化して、トレーニングとテストの例を生成します。このような例は、自然言語推論のトレーニングとテスト中にミニバッチで読み込まれます。

```{.python .input}
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## BERT の微調整

:numref:`fig_bert-two-seqs` が示すように、自然言語推論のために BERT を微調整するには、2 つの完全に接続されたレイヤーで構成される追加の MLP のみが必要です (次の `BERTClassifier` クラスの `self.hidden` と `self.output` を参照)。この MLP は、<cls>前提と仮説の両方の情報をエンコードする特別な「」トークンの BERT 表現を、自然言語推論の 3 つのアウトプット (含意、矛盾、中立) に変換します。

```{.python .input}
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

以下では、事前学習済みの BERT モデル `bert` がダウンストリームアプリケーション用の `BERTClassifier` インスタンス `net` に供給されます。BERT 微調整の一般的な実装では、追加 MLP (`net.output`) の出力レイヤのパラメータのみがゼロから学習されます。事前学習済みの BERT エンコーダ (`net.encoder`) および追加 MLP (`net.hidden`) の隠れ層 (`net.hidden`) のすべてのパラメーターが微調整されます。

```{.python .input}
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

:numref:`sec_bert` では、`MaskLM` クラスと `NextSentencePred` クラスの両方に、使用される MLP にパラメータが含まれていることを思い出してください。これらのパラメーターは、事前学習済みの BERT モデル `bert` に含まれるパラメーターの一部であり、`net` のパラメーターの一部です。ただし、このようなパラメータは、事前学習中のマスク言語モデリング損失と次の文予測損失を計算するためだけのものです。これら 2 つの損失関数はダウンストリームアプリケーションの微調整とは無関係であるため、`MaskLM` および `NextSentencePred` で使用されている MLP のパラメータは、BERT が微調整されても更新 (失敗) されません。 

古いグラデーションを持つパラメータを許可するには、`d2l.train_batch_ch13` の `step` 関数に `ignore_stale_grad=True` というフラグを設定します。この関数を使用して、SNLI のトレーニングセット (`train_iter`) とテストセット (`test_iter`) を使用して、モデル `net` をトレーニングおよび評価します。計算リソースが限られているため、トレーニングとテストの精度をさらに向上させることができます。その議論は演習に残します。

```{.python .input}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [概要

* SNLI データセットでの自然言語推論など、ダウンストリームアプリケーション向けに事前学習済みの BERT モデルを微調整できます。
* 微調整中、BERT モデルはダウンストリームアプリケーションのモデルの一部になります。事前学習損失にのみ関連するパラメーターは、微調整中に更新されません。 

## 演習

1. 計算リソースが許せば、元の BERT ベースモデルとほぼ同じ大きさの事前学習済みの BERT モデルを微調整します。`load_pretrained_model` 関数の引数を次のように設定します。「bert.small」を「bert.base」に置き換え、`num_hiddens=256`、`ffn_num_hiddens=512`、`num_heads=4`、`num_layers=2` の値を 768、3072、12、12 に増やします。エポックを微調整する (そして場合によっては他のハイパーパラメータを調整する) ことで、0.86 より高いテスト精度を得ることができますか?
1. 長さの比に応じて一対のシーケンスを切り捨てる方法は？このペア切り捨て方法と `SNLIBERTDataset` クラスで使用されているものを比較してください。彼らの長所と短所は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1526)
:end_tab:

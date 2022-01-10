# トランスフォーマー (BERT) からの双方向エンコーダ表現
:label:`sec_bert`

自然言語理解のための単語埋め込みモデルをいくつか紹介しました。事前学習後の出力は、各行が定義済みの語彙の単語を表すベクトルである行列と考えることができます。実際、これらの単語埋め込みモデルはすべて*文脈に依存しない* です。このプロパティを説明することから始めましょう。 

## コンテキスト非依存からコンテキストセンシティブへ

:numref:`sec_word2vec_pretraining` と :numref:`sec_synonyms` の実験を思い出してください。たとえば、word2vec と GLOVE はどちらも、単語のコンテキスト (存在する場合) に関係なく、同じ事前学習済みベクトルを同じ単語に割り当てます。正式には、コンテキストに依存しないトークン $x$ の表現は $x$ だけを入力として取る関数 $f(x)$ である。自然言語における多義性と複雑な意味論の豊富さを考えると、文脈に依存しない表現には明らかな限界がある。例えば、「クレーンが飛んでいる」と「クレーンの運転手が来た」という文脈における「クレーン」という言葉は、まったく意味が異なり、同じ単語に文脈によって異なる表現が割り当てられることがあります。 

これは、単語の表現が文脈に依存する、*文脈依存型*単語表現の開発の動機となります。したがって、トークン $x$ のコンテキスト依存表現は $x$ とそのコンテキスト $c(x)$ の両方に依存する関数 $f(x, c(x))$ である。一般的な状況依存表現には、tagLM (言語モデル拡張シーケンスタガー) :cite:`Peters.Ammar.Bhagavatula.ea.2017`、CoVE (コンテキストベクトル) :cite:`McCann.Bradbury.Xiong.ea.2017`、ELMO (言語モデルからの埋め込み) :cite:`Peters.Neumann.Iyyer.ea.2018` などがあります。 

たとえば、シーケンス全体を入力として取ると、elMo は入力シーケンスの各単語に表現を割り当てる関数になります。具体的には、eLMO は事前学習済みの双方向 LSTM からのすべての中間層表現を出力表現として結合します。次に、ElMO 表現は、既存のモデル内の ElMO 表現とトークンの元の表現 (例:GLOVE) を連結するなどして、追加機能として下流タスクの既存の教師ありモデルに追加されます。一方では、事前学習済みの双方向 LSTM モデルのすべての重みは、ElMO 表現が追加された後に固定されます。一方、既存の教師ありモデルは、特定のタスクに合わせて特別にカスタマイズされています。当時のさまざまなタスクにさまざまなベストモデルを活用することで、感情分析、自然言語推論、セマンティックロールラベリング、共参照解決、名前付きエンティティ認識、質問応答という6つの自然言語処理タスクにわたって最先端の技術が向上しました。 

## タスク固有からタスク非依存へ

ElMO は多様な自然言語処理タスクに対するソリューションを大幅に改善しましたが、各ソリューションは依然として「タスク固有の」アーキテクチャに依存しています。ただし、すべての自然言語処理タスクに対して特定のアーキテクチャを作成することは、事実上自明ではありません。GPT (Generative Pre-Training) モデルは、状況依存表現 :cite:`Radford.Narasimhan.Salimans.ea.2018` のための一般的な *タスクにとらわれない* モデルを設計する作業を表しています。トランスフォーマーデコーダー上に構築された GPT は、テキストシーケンスを表すために使用される言語モデルを事前にトレーニングします。GPT を下流のタスクに適用すると、言語モデルの出力が追加された線形出力レイヤーに送られ、タスクのラベルが予測されます。事前学習済みモデルのパラメーターを固定する eLMO とは対照的に、GPT は下流タスクの教師あり学習中に事前学習済みのトランスフォーマーデコーダー内のパラメーターを「すべて」微調整します。GPTは、自然言語推論、質問応答、文の類似性、分類の12のタスクについて評価され、モデルアーキテクチャの変更を最小限に抑えながら9つの最先端の技術を改善しました。 

ただし、言語モデルには自己回帰的な性質があるため、GPT は前方 (左から右) のみを見ています。「現金を預けるために銀行に行った」と「座るために銀行に行った」というコンテキストでは、「銀行」は左側のコンテキストに敏感であるため、GPT は「銀行」に対して同じ表現を返しますが、意味は異なります。 

## BERT: 両方の長所を組み合わせる

これまで見てきたように、ElMO はコンテキストを双方向にエンコードしますが、タスク固有のアーキテクチャを使用します。GPT はタスクにとらわれず、コンテキストを左から右にエンコードします。BERT (Transformers からの双方向エンコーダ表現) は、コンテキストを双方向にエンコードし、幅広い自然言語処理タスクに対して最小限のアーキテクチャ変更で済みます :cite:`Devlin.Chang.Lee.ea.2018`。事前学習済みのトランスエンコーダーを使用することで、BERT は双方向コンテキストに基づいて任意のトークンを表現できます。下流タスクの教師あり学習では、BERT は 2 つの点で GPT と似ています。まず、BERT 表現が追加された出力レイヤーに入力され、各トークンの予測とシーケンス全体の予測など、タスクの性質に応じてモデルアーキテクチャに最小限の変更が加えられます。次に、事前学習済みのトランスエンコーダーのすべてのパラメーターが微調整され、追加の出力層はゼロからトレーニングされます。:numref:`fig_elmo-gpt-bert` は、eLMO、GPT、および BERT の違いを示しています。 

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT は、(i) 単一テキスト分類 (センチメント分析など)、(ii) テキストペア分類 (例:自然言語推論)、(iii) 質問応答、(iv) テキストタグ付け (名前付きエンティティ認識など) の幅広いカテゴリの下で、11 の自然言語処理タスクに関する最先端技術をさらに改善しました。。2018年に提案された、状況依存型eLMOからタスクにとらわれないGPT、BERTまで、概念的にシンプルでありながら経験的に強力な自然言語のディープ表現の事前トレーニングは、さまざまな自然言語処理タスクのソリューションに革命をもたらしました。 

この章の残りの部分では、BERT の事前トレーニングについて詳しく説明します。:numref:`chap_nlp_app` で自然言語処理アプリケーションについて説明すると、ダウンストリームアプリケーションに対する BERT の微調整について解説します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## 入力リプレゼンテーション
:label:`subsec_bert_input_rep`

自然言語処理では、一部のタスク (センチメント分析など) は単一のテキストを入力として受け取り、他のタスク (自然言語推論など) では入力がテキストシーケンスのペアになります。BERT 入力シーケンスは、単一テキストとテキストペアの両方を明確に表します。前者では、BERT 入力シーケンスは特殊分類トークン「<cls>」、テキストシーケンスのトークン、および特殊分離トークン「<sep>」を連結したものです。後者の場合、BERT 入力シーケンスは「<cls>」、最初のテキストシーケンスのトークン、「<sep>」、2 番目のテキストシーケンスのトークン、および「<sep>」を連結したものです。「BERT 入力シーケンス」という用語を他のタイプの「シーケンス」と一貫して区別します。たとえば、1 つの*BERT 入力シーケンス* には、1 つの*text シーケンス* または 2 つの*text シーケンス* を含めることができます。 

テキストペアを区別するために、学習されたセグメント埋め込み $\mathbf{e}_A$ と $\mathbf{e}_B$ が、それぞれ 1 番目のシーケンスと 2 番目のシーケンスのトークン埋め込みに追加されます。単一テキスト入力の場合、$\mathbf{e}_A$ のみが使用されます。 

次の `get_tokens_and_segments` は、1 文または 2 つの文を入力として受け取り、BERT 入力シーケンスのトークンとそれに対応するセグメント ID を返します。

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT は、双方向アーキテクチャとしてトランスエンコーダを選択しました。トランスエンコーダでは一般的に位置埋め込みが BERT 入力シーケンスのすべての位置に追加されます。ただし、元のトランスエンコーダとは異なり、BERT は*学習可能* 位置埋め込みを使用します。まとめると、:numref:`fig_bert-input` は、BERT 入力シーケンスの埋め込みが、トークンの埋め込み、セグメントの埋め込み、および位置埋め込みの合計であることを示しています。 

![BERT 入力シーケンスの埋め込みは、トークンの埋め込み、セグメントの埋め込み、および位置埋め込みの合計です。](../img/bert-input.svg) :label:`fig_bert-input` 

次の `BERTEncoder` クラスは :numref:`sec_transformer` で実装された `TransformerEncoder` クラスと似ています。`TransformerEncoder` とは異なり、`BERTEncoder` はセグメント埋め込みと学習可能な位置埋め込みを使用します。

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

ボキャブラリのサイズが 10000 であるとします。`BERTEncoder` の前方推論を実証するために、そのインスタンスを作成し、パラメーターを初期化しましょう。

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

`tokens` は、長さ 8 の 2 つの BERT 入力シーケンスと定義し、各トークンはボキャブラリのインデックスです。入力 `tokens` で `BERTEncoder` の前方推論を行うと、エンコードされた結果が返されます。各トークンは、ハイパーパラメータ `num_hiddens` で事前に定義された長さのベクトルで表されます。このハイパーパラメータは通常、トランスエンコーダの*隠れサイズ* (隠れユニットの数) と呼ばれます。

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## 事前トレーニングタスク
:label:`subsec_bert_pretraining_tasks`

`BERTEncoder` の前方推論により、入力テキストの各トークンと、挿入された特殊トークン「<cls>」と「<seq>」の BERT 表現が得られます。次に、これらの表現を使用して、BERT の事前学習のための損失関数を計算します。事前学習は、マスク言語モデリングと次文予測の2つのタスクから構成されます。 

### 仮面言語モデリング
:label:`subsec_mlm`

:numref:`sec_language_model` に示すように、言語モデルは左側のコンテキストを使用してトークンを予測します。各トークンを表すためにコンテキストを双方向にエンコードするために、BERT はトークンをランダムにマスクし、双方向コンテキストのトークンを使用して、マスクされたトークンを自己監視方式で予測します。このタスクを*マスク言語モデル* と呼びます。 

この事前トレーニングタスクでは、予測のためにマスクされたトークンとして 15% のトークンがランダムに選択されます。ラベルを使用してチートせずにマスクされたトークンを予測するには、<mask>BERT 入力シーケンスで常に特別な「」トークンに置き換えるという簡単なアプローチがあります。ただし、<mask>微調整では人工的な特殊トークン「」は表示されません。事前トレーニングと微調整のミスマッチを避けるために、トークンが予測のためにマスクされている場合 (「この映画は素晴らしい」でマスクされ、予測されるように「great」が選択された場合)、入力では次のように置き換えられます。 

* <mask>80％の時間の特別な「」トークン（例えば、「この映画は素晴らしい」が「この映画は」になる<mask>）
* 10% の確率でランダムなトークン（例えば、「この映画は素晴らしい」が「この映画は飲み物」になる）
* 10% の時間変更されていないラベル・トークン (例:「この映画は素晴らしい」が「この映画は素晴らしい」になる)

15% の 10% の間、ランダムなトークンが挿入されることに注意してください。この時折発生するノイズにより、BERT の双方向コンテキストエンコーディングにおいて、BERT はマスクされたトークンに対するバイアスが小さくなります（特に、ラベルトークンが変更されない場合）。 

BERT 事前学習のマスク言語モデルタスクでマスクされたトークンを予測するために、次の `MaskLM` クラスを実装します。この予測では、隠れ層が 1 層の MLP (`self.mlp`) が使用されます。前方推論では、エンコードされた結果の `BERTEncoder` と予測用のトークン位置の 2 つの入力が必要です。出力は、これらの位置での予測結果です。

```{.python .input}
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

`MaskLM` の前方推論を実証するために、インスタンス `mlm` を作成して初期化します。`BERTEncoder` の前方推論による `encoded_X` は 2 つの BERT 入力シーケンスを表していることを思い出してください。`mlm_positions` は、いずれかの BERT 入力シーケンス `encoded_X` で予測する 3 つのインデックスとして定義します。`mlm` の前方推論では、`encoded_X` のマスクされたすべての位置 `mlm_positions` で予測結果 `mlm_Y_hat` が返されます。予測ごとに、結果のサイズは語彙のサイズと等しくなります。

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

予測されたトークン `mlm_Y_hat` のグラウンドトゥルースラベル `mlm_Y` をマスクの下に置くと、BERT 事前学習におけるマスク言語モデルタスクのクロスエントロピー損失を計算できます。

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### 次のセンテンス予測
:label:`subsec_nsp`

マスク言語モデリングでは、単語を表す双方向コンテキストをエンコードできますが、テキストペア間の論理的な関係を明示的にモデル化することはできません。2 つのテキストシーケンスの関係を理解しやすくするために、BERT は事前学習でバイナリ分類タスク (*次の文の予測*) を考慮します。事前トレーニング用のセンテンスペアを生成する場合、半分の時間は実際には「True」というラベルの付いた連続したセンテンスです。残りの半分の時間は、コーパスから「False」というラベルの付いた2番目のセンテンスがランダムにサンプリングされます。 

次の `NextSentencePred` クラスは、1 隠れ層 MLP を使用して、2 番目の文が BERT 入力シーケンスの最初の文の次の文であるかどうかを予測します。トランスエンコーダでの自己注意のため、特別なトークン「<cls>」の BERT 表現は、入力からの 2 つの文の両方をエンコードします。したがって、MLP 分類器の出力層 (`self.output`) は `X` を入力として使用します。`X` は、入力がエンコードされた「<cls>」トークンである MLP 隠れ層の出力です。

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

`NextSentencePred` インスタンスの前方推論により、BERT 入力シーケンスごとにバイナリ予測が返されることがわかります。

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

2 つのバイナリ分類のクロスエントロピー損失も計算できます。

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

前述の両方の事前トレーニングタスクのすべてのラベルは、手作業によるラベル付け作業なしに、事前トレーニングコーパスから簡単に取得できることは注目に値します。オリジナルの BERT は BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015` と英語版ウィキペディアの連結について事前にトレーニングされています。この2つのテキストコーパスは巨大で、それぞれ8億語と25億語があります。 

## すべてのものをまとめる

BERT を事前学習する場合、最終的な損失関数は、マスク言語モデリングと次の文の予測の両方の損失関数の線形結合になります。これで `BERTEncoder`、`MaskLM`、`NextSentencePred` の 3 つのクラスをインスタンス化して `BERTModel` クラスを定義できます。前方推論は、エンコードされた BERT 表現 `encoded_X`、マスク言語モデリング `mlm_Y_hat` の予測、および次の文の予測 `nsp_Y_hat` を返します。

```{.python .input}
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## [概要

* word2vec や GLOVE などの単語埋め込みモデルはコンテキストに依存しません。単語のコンテキスト (存在する場合) に関係なく、同じ事前学習済みベクトルを同じ単語に代入します。多義性や複雑な意味論を自然言語でうまく扱うのは難しい。
* eLMO や GPT などの文脈依存の単語表現では、単語の表現は文脈に依存します。
* ElMO はコンテキストを双方向にエンコードしますが、タスク固有のアーキテクチャーを使用します (ただし、自然言語処理タスクごとに特定のアーキテクチャーを作成することは事実上自明ではありません)。一方、GPT はタスクにとらわれず、コンテキストを左から右にエンコードします。
* BERT は、コンテキストを双方向にエンコードし、幅広い自然言語処理タスクに対して最小限のアーキテクチャ変更で済むという、両方の長所を兼ね備えています。
* BERT 入力シーケンスの埋め込みは、トークン埋め込み、セグメント埋め込み、位置埋め込みの合計です。
* BERT の事前学習は、マスク言語モデリングと次文予測の 2 つのタスクで構成されます。前者は単語を表現するための双方向コンテキストをエンコードでき、後者はテキストペア間の論理関係を明示的にモデル化します。

## 演習

1. BERTはなぜ成功するのですか？
1. 他のすべてが等しい場合、マスクされた言語モデルは、左から右への言語モデルよりも収束するのに必要な事前トレーニングステップが多かれ少なかれ必要ですか？なぜ？
1. BERT の元の実装では、`BERTEncoder` (`d2l.EncoderBlock` 経由) の位置方向フィードフォワードネットワークと `MaskLM` の完全接続層の両方で、ガウス誤差線形単位 (GELU) :cite:`Hendrycks.Gimpel.2016` がアクティベーション関数として使用されています。GELU と ReLU の違いを調査します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:

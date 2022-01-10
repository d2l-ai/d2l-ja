# サブワード埋め込み
:label:`sec_fasttext`

英語では、「助けて」、「助けた」、「助ける」などの単語は、同じ単語「ヘルプ」の活用形です。犬」と「犬」の関係は「猫」と「猫」の関係と同じで、「男の子」と「彼氏」の関係は「女の子」と「彼女」の関係と同じです。フランス語やスペイン語などの他の言語では、多くの動詞に40以上の活用形がありますが、フィンランド語では名詞に最大15のケースがあります。言語学では、形態学は単語形成と単語関係を研究します。しかし、単語の内部構造は word2vec でも GLOVE でも探究されていませんでした。 

## FastText モデル

word2vec で単語がどのように表現されているかを思い出してください。スキップグラムモデルと連続バッグオブワードモデルの両方で、同じ単語の異なる屈折形は、共有パラメーターを持たない異なるベクトルによって直接表されます。形態学的情報を使用するために、*FastText* モデルは*サブワード埋め込み* アプローチを提案しました。サブワードは文字 $n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017` です。単語レベルのベクトル表現を学習する代わりに、FastText をサブワードレベルのスキップグラムと見なすことができます。この場合、各*中心語* はサブワードベクトルの和で表されます。 

FastText で「where」という単語を使用して、各センターワードのサブワードを取得する方法を説明しましょう。まず、<」and「> 接頭辞と接尾辞を他のサブワードと区別するために、単語の先頭と末尾に特殊文字「」を追加します。次に、単語から $n$ グラムという文字を抽出します。たとえば、$n=3$ の場合、長さ 3 のすべてのサブワード「<wh」,「whe」,「her」,「ere」,「re>」と、特殊なサブワード "<where>」が取得されます。 

FastText では、任意の単語 $w$ について、長さが 3 から 6 までのすべてのサブワードとその特殊サブワードの和集合を $\mathcal{G}_w$ で表します。語彙は、すべての単語のサブワードの和集合です。$\mathbf{z}_g$ を辞書のサブワード $g$ のベクトルとすると、スキップグラムモデルの中心語である単語 $w$ のベクトル $\mathbf{v}_w$ は、そのサブワードベクトルの和になります。 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

FastText の残りの部分はスキップグラムモデルと同じです。スキップグラムモデルと比較すると、FastText の語彙が大きくなり、モデルパラメーターが多くなります。さらに、単語の表現を計算するには、そのすべてのサブワードベクトルを合計する必要があり、計算の複雑さが高くなります。ただし、類似した構造を持つ単語間でサブワードのパラメーターが共有されるため、まれな単語や語彙外の単語でも、FastText ではより優れたベクトル表現が得られる可能性があります。 

## バイトペアエンコーディング
:label:`subsec_Byte_Pair_Encoding`

FastText では、抽出されるすべてのサブワードは $3$ から $6$ のように指定した長さでなければならないため、ボキャブラリのサイズを事前に定義することはできません。固定サイズの語彙に可変長のサブワードを含めるには、*byte pair encoding* (BPE) と呼ばれる圧縮アルゴリズムを適用してサブワード :cite:`Sennrich.Haddow.Birch.2015` を抽出します。 

バイトペア符号化は、トレーニングデータセットの統計分析を実行して、任意の長さの連続する文字など、単語内の共通記号を検出します。バイトペアエンコーディングは、長さ 1 のシンボルから始めて、最も頻度の高い連続するシンボルのペアを繰り返しマージして、新しい長いシンボルを生成します。効率化のため、単語の境界を越えるペアは考慮されないことに注意してください。最後に、サブワードなどの記号を使用して単語をセグメント化できます。バイトペアエンコーディングとそのバリアントは、GPT-2 :cite:`Radford.Wu.Child.ea.2019` や RoberTA :cite:`Liu.Ott.Goyal.ea.2019` などの一般的な自然言語処理事前学習モデルで入力表現に使用されています。以下では、バイト・ペア・エンコーディングの仕組みを説明します。 

最初に、記号の語彙をすべての英小文字、特殊な単語末尾記号 `'_'`、特殊不明記号 `'[UNK]'` として初期化します。

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

単語の境界を越えるシンボルペアは考慮しないため、必要なのは、単語をデータセット内の頻度 (出現回数) にマッピングするディクショナリ `raw_token_freqs` だけです。出力シンボルのシーケンス (「a_ tall er_ man」など) から単語シーケンス (「a taller man」など) を簡単に復元できるように、各単語に特殊記号 `'_'` が追加されることに注意してください。マージ処理は 1 文字と特殊記号だけのボキャブラリから開始するので、各単語内の連続する文字のペアごとにスペースが挿入されます (辞書 `token_freqs` のキー)。言い換えると、スペースは単語内の記号間の区切り文字です。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

以下の `get_max_freq_pair` 関数を定義します。この関数は、単語が入力辞書 `token_freqs` のキーから取られる、単語内で最も頻繁に連続する記号のペアを返します。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

連続するシンボルの頻度に基づく貪欲なアプローチとして、バイトペアエンコーディングでは次の `merge_symbols` 関数を使用して、最も頻度の高い連続するシンボルのペアをマージして新しいシンボルを生成します。

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

ここで、ディクショナリ `token_freqs` のキーに対してバイトペア符号化アルゴリズムを繰り返し実行します。最初の反復では、連続するシンボルの最も頻繁なペアは `'t'` と `'a'` です。したがって、バイトペアエンコーディングはこれらをマージして新しいシンボル `'ta'` を生成します。2 回目の反復では、バイトペアエンコーディングが `'ta'` と `'l'` をマージし続け、新しいシンボル `'tal'` が生成されます。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

バイトペアエンコーディングを 10 回繰り返すと、リスト `symbols` に、他のシンボルから繰り返しマージされたシンボルがさらに 10 個含まれていることがわかります。

```{.python .input}
#@tab all
print(symbols)
```

ディクショナリ `raw_token_freqs` のキーで指定されている同じデータセットに対して、データセット内の各単語は、バイトペアエンコードアルゴリズムの結果として、サブワード「fast_」、「fast」、「er_」、「tall_」、「tall」でセグメント化されるようになりました。たとえば、「faster_」と「taller_」という単語は、それぞれ「fast er_」と「tall er_」としてセグメント化されます。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

バイトペアエンコーディングの結果は、使用するデータセットによって異なることに注意してください。また、あるデータセットから学習したサブワードを使用して、別のデータセットの単語をセグメント化することもできます。貪欲なアプローチとして、次の `segment_BPE` 関数は、単語を入力引数 `symbols` から可能な限り長いサブワードに分割しようとします。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

以下では、前述のデータセットから学習したリスト `symbols` のサブワードを、別のデータセットを表すセグメント `tokens` に使用しています。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## [概要

* FastText モデルでは、サブワード埋め込みアプローチが提案されています。word2vec のスキップグラムモデルに基づき、サブワードベクトルの和としてセンターワードを表します。
* バイトペア符号化では、トレーニングデータセットの統計解析を実行して、単語内の共通記号を検出します。欲張りなアプローチとして、バイトペアエンコーディングは、最も頻度の高い連続するシンボルのペアを繰り返しマージします。
* サブワードを埋め込むと、まれな単語や辞書外の単語の表現の質が向上します。

## 演習

1. 一例として、英語で約$3\times 10^8$の可能な$6$グラムがあります。サブワードが多すぎるとどうなりますか？どのように問題に対処するのですか？ヒント: refer to the end of Section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`。
1. 連続バッグオブワードモデルに基づいてサブワード埋め込みモデルを設計するにはどうすればよいですか？
1. サイズ$m$のボキャブラリを取得するには、最初のシンボルボキャブラリサイズが $n$ の場合、いくつのマージ操作が必要ですか？
1. バイトペアエンコーディングの考え方を拡張してフレーズを抽出するにはどうすればよいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

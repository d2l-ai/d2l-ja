# 機械翻訳とデータセット
:label:`sec_machine_translation`

私たちは、自然言語処理の鍵となる言語モデルの設計にRNNを使ってきました。もう1つの主要なベンチマークは、入力シーケンスを出力シーケンスに変換する*シーケンス変換* モデルの中心的な問題領域である*機械翻訳*です。現代のさまざまなAIアプリケーションで重要な役割を果たす配列変換モデルは、この章の残りの部分と :numref:`chap_attention` で焦点を当てます。そのために、このセクションでは機械翻訳の問題と、後で使用するデータセットを紹介します。 

*機械翻訳* とは
ある言語から別の言語へのシーケンスの自動翻訳。実際、この分野は、デジタルコンピューターが発明された直後の1940年代にさかのぼる可能性があります。特に、第二次世界大戦で言語コードの解読にコンピューターを使用することを検討したためです。ニューラルネットワークを使用したエンドツーエンド学習が登場する前は、この分野では数十年にわたって統計的アプローチが主流でした :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`。後者はよく呼ばれます
*ニューラル機械翻訳*
それ自身と区別するために
*統計的機械翻訳*
これには、翻訳モデルや言語モデルなどのコンポーネントの統計分析が含まれます。 

本書では、エンドツーエンドの学習に重点を置き、ニューラル機械翻訳の手法に焦点を当てます。コーパスが単一言語である :numref:`sec_language_model` の言語モデル問題とは異なり、機械翻訳データセットは、それぞれソース言語とターゲット言語のテキストシーケンスのペアで構成されます。したがって、言語モデリングに前処理ルーチンを再利用する代わりに、機械翻訳データセットを前処理する別の方法が必要となります。以下では、前処理されたデータをトレーニング用にミニバッチに読み込む方法を示します。

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## [**データセットのダウンロードと前処理**]

まず、[bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/) で構成される英仏語のデータセットをダウンロードします。データセットの各行は、英語のテキストシーケンスと翻訳されたフランス語のテキストシーケンスのタブ区切りのペアです。各テキストシーケンスは、1 つのセンテンスでも、複数のセンテンスで構成される 1 つの段落でもかまいません。英語がフランス語に翻訳されるこの機械翻訳の問題では、英語が*ソース言語*、フランス語が*ターゲット言語*です。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

データセットをダウンロードした後、生のテキストデータに対して [**いくつかの前処理ステップを続行**] します。たとえば、改行しないスペースをスペースに置き換えたり、大文字を小文字に変換したり、単語と句読点の間にスペースを挿入したりします。

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## [**トークン化**]

:numref:`sec_language_model` の文字レベルのトークン化とは異なり、ここでは機械翻訳では単語レベルのトークン化が好まれます (最先端のモデルでは、より高度なトークン化技術が使用される場合があります)。次の `tokenize_nmt` 関数は、最初の `num_examples` テキストシーケンスのペアをトークン化します。各トークンは単語または句読点です。この関数は、`source` と `target` の 2 つのトークンリストのリストを返します。具体的には、`source[i]` はソース言語 (ここでは英語) の $i^\mathrm{th}$ テキストシーケンスのトークンのリストで、`target[i]` はターゲット言語 (ここではフランス語) のトークンのリストです。

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

[**テキストシーケンスあたりのトークン数のヒストグラムをプロットしてみよう。**] この単純な英仏語のデータセットでは、ほとんどのテキストシーケンスのトークン数は20個未満です。

```{.python .input}
#@tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

## [**ボキャブラリー**]

機械翻訳データセットは言語のペアで構成されているため、ソース言語とターゲット言語の両方に対して 2 つのボキャブラリを別々に構築できます。単語レベルのトークン化では、文字レベルのトークン化を使用する場合よりも語彙のサイズが大幅に大きくなります。これを軽減するために、ここでは 2 回未満出現する頻度の低いトークンを同じ不明 (」<unk>「) トークンとして扱います。それ以外に、<pad>ミニバッチで同じ長さのパディング (」「) シーケンスを指定したり、<bos><eos>シーケンスの先頭 (」「) または終了 (」「) をマークしたりするための特別なトークンを追加します。このような特殊なトークンは、自然言語処理タスクでよく使われます。

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## データセットの読み取り
:label:`subsec_mt_data_loading`

言語モデリング [**各シーケンスの例**] では、1 つの文のセグメントまたは複数の文にまたがる (**固定長**) ことを思い出してください。これは :numref:`sec_language_model` の `num_steps` (タイムステップまたはトークンの数) 引数によって指定されました。機械翻訳では、各例は原文と訳文のテキストシーケンスのペアであり、各テキストシーケンスの長さは異なる場合があります。 

計算効率を高めるために、*truncation* と*padding* によって、テキストシーケンスのミニバッチを一度に処理できます。同じミニバッチ内のすべての配列が同じ長さの `num_steps` であると仮定します。テキストシーケンスのトークン数が `num_steps` より少ない場合、<pad>その長さが `num_steps` に達するまで、特殊な "" トークンを末尾に追加し続けます。それ以外の場合は、最初の `num_steps` トークンのみを取り、残りを破棄して、テキストシーケンスを切り捨てます。このようにすると、すべてのテキストシーケンスは同じ長さになり、同じ形状のミニバッチに読み込まれます。 

次の `truncate_pad` 関数 (**テキストシーケンスの切り捨てまたはパディング**) は、前述のとおり。

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

ここで、[**テキストシーケンスをトレーニング用のミニバッチに変換する**] 関数を定義します。<eos>各シーケンスの最後に特殊な「」トークンを追加して、シーケンスの終わりを示します。モデルがトークンの後にシーケンストークンを生成して予測している場合、<eos>"" トークンの生成は出力シーケンスの完了を示唆します。また、パディングトークンを除いた各テキストシーケンスの長さも記録します。この情報は、後で説明する一部のモデルで必要になります。

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## [**すべてのものをまとめる**]

最後に、`load_data_nmt` 関数を定義して、ソース言語とターゲット言語の両方のボキャブラリとともにデータイテレータを返します。

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

[**英仏のデータセットから最初のミニバッチを読み上げます**]

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## [概要

* 機械翻訳とは、ある言語から別の言語へのシーケンスの自動翻訳のことです。
* 単語レベルのトークン化を使用すると、文字レベルのトークン化を使用する場合よりも語彙のサイズが大幅に大きくなります。これを軽減するために、頻度の低いトークンを同じ未知のトークンとして扱うことができます。
* テキストシーケンスをすべて同じ長さでミニバッチでロードできるように、テキストシーケンスを切り捨てて埋め込むことができます。

## 演習

1. `load_data_nmt` 関数で `num_examples` 引数の値を変えてみてください。これはソース言語とターゲット言語のボキャブラリーサイズにどのような影響を与えますか？
1. 中国語や日本語などの一部の言語のテキストには、単語境界インジケーター (スペースなど) がありません。このような場合、単語レベルのトークン化はまだ良い考えですか？なぜ、なぜそうではないのですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

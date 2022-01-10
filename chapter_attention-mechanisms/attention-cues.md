# アテンションキュー
:label:`sec_attention-cues`

この本に注目していただきありがとうございます。注意は乏しいリソースです。現時点では、この本を読んでいて、残りは無視しています。したがって、お金と同様に、あなたの注意は機会費用で支払われています。あなたの今注目の投資が価値あるものであることを保証するために、私たちは素晴らしい本を作るために注意深く注意を払うことに非常に意欲的です。注意は人生のアーチの要であり、あらゆる作品の例外主義の鍵を握っています。 

経済学は希少資源の配分を研究しているので、人間の注意は交換可能な限定的で価値のある、希少な商品として扱われるアテンションエコノミーの時代です。それを活用するために、数多くのビジネスモデルが開発されてきました。音楽やビデオのストリーミングサービスでは、広告に注意を払うか、お金を払って非表示にします。オンラインゲームの世界での成長のために、新しいゲーマーを引き付ける戦闘に参加するか、お金を払って即座に強力になることに注意を払います。無料で提供されるものはありません。 

全体として、私たちの環境内の情報は乏しくなく、注意が必要です。視覚シーンを検査するとき、私たちの視神経は毎秒$10^8$ビットというオーダーで情報を受け取り、私たちの脳が完全に処理できるものをはるかに超えています。幸いなことに、私たちの先祖は経験（データとも呼ばれる）から、*すべての感覚入力が同じように作られるわけではない* ということを学んでいました。人類の歴史を通じて、関心のある情報のほんの一部に注意を向けることができたことで、私たちの脳は、捕食者、獲物、仲間を検出するなど、生き残り、成長し、社会化するために、よりスマートに資源を割り当てることができました。 

## 生物学における注意の合図

私たちの注意が視覚世界にどのように展開されているかを説明するために、2つの要素からなるフレームワークが登場し、普及しています。このアイデアは、「アメリカ心理学の父」と見なされている1890年代のウィリアム・ジェームズにまでさかのぼります。:cite:`James.2007`。この枠組みでは、被験者は*不意志的合図*と*意志的合図*の両方を用いて注意のスポットライトを選択的に指示する。 

ノンヴォリショナルキューは、環境内の物体の顕著性と目立ち度に基づいています。あなたの目の前には、新聞、研究論文、一杯のコーヒー、ノート、:numref:`fig_eye-coffee` のような本の 5 つの物があるとします。紙製品はすべて白黒で印刷されていますが、コーヒーカップは赤です。言い換えれば、このコーヒーは、この視覚環境では本質的に顕著で目立ち、自動的にそして思わず注意を引く。:numref:`fig_eye-coffee`に示すように、中心窩部（視力が最も高い黄斑の中心）をコーヒーの上に持ってきます。 

![Using the nonvolitional cue based on saliency (red cup, non-paper), attention is involuntarily directed to the coffee.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

コーヒーを飲んだ後、カフェインが入って本を読みたくなる。それで、あなたは頭を向けて、再び目を向け、:numref:`fig_eye-book`に描かれている本を見てください。:numref:`fig_eye-coffee` のコーヒーが顕著性に基づく選択に偏る場合とは異なり、このタスク依存のケースでは、認知と意志の制御下で本を選択します。変数選択基準に基づく意志キューを使用すると、この形式の注意はより慎重になります。また、被験者の自発的な努力により、より強力になります。 

![Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`

## クエリ、キー、値

以下では、注意の展開を説明する非意志的および意欲的な注意の手がかりに触発され、これら2つの注意の手がかりを組み込むことによって注意メカニズムを設計するためのフレームワークについて説明します。 

まず、非意志キューしか利用できない、より単純なケースを考えてみましょう。感覚入力よりも選択にバイアスをかけるには、パラメーター化された完全結合層、またはパラメーター化されていない最大プーリングまたは平均プーリングを単純に使用できます。 

したがって、これらの完全に接続された層またはプーリング層とは別に注意メカニズムを設定するのは、意欲的な手がかりを含めることです。アテンションメカニズムの文脈では、意志的手がかりを*クエリ*と呼びます。クエリが与えられた場合、注意メカニズムは、*アテンションプーリング*を介して、感覚入力（例えば、中間的な特徴表現）よりも選択にバイアスをかける。これらの感覚入力は、注意メカニズムの文脈では*values* と呼ばれます。より一般的には、すべての値が*key* とペアになっています。これは、その感覚入力の非自発的な合図と考えることができます。:numref:`fig_qkv` に示すように、アテンションプーリングを設計して、与えられたクエリ (意志キュー) がキー (非意志キュー) と相互作用し、値 (感覚入力) に対するバイアス選択の指針となります。 

![Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).](../img/qkv.svg)
:label:`fig_qkv`

注意メカニズムの設計には多くの選択肢があることに注意してください。たとえば、強化学習法 :cite:`Mnih.Heess.Graves.ea.2014` を使用して学習できる、微分不可能な注意モデルを設計できます。:numref:`fig_qkv` のフレームワークの優位性を考えると、この章ではこのフレームワークのモデルが注目されます。 

## 注意の可視化

平均プーリングは、重みが一様な入力の加重平均として扱うことができます。実際には、アテンションプーリングは加重平均を使用して値を集計します。加重平均では、特定のクエリと異なるキーとの間で重みが計算されます。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

アテンションウェイトを可視化するために、`show_heatmaps` 関数を定義します。入力 `matrices` の形状は (表示する行数、表示する列数、クエリ数、キー数) です。

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

デモンストレーションのために、クエリとキーが同じ場合にのみアテンションウェイトが 1 になり、それ以外の場合はゼロになるという単純なケースを考えます。

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

以降のセクションでは、アテンションウェイトを視覚化するためにこの関数を頻繁に呼び出します。 

## [概要

* 人間の注意は限られており、貴重で、希少な資源です。
* 被験者は、不意志的手がかりと意志的手がかりの両方を使用して選択的に注意を向けます。前者は顕著性に基づき、後者はタスクに依存する。
* アテンションメカニズムは、意志キューが含まれているため、完全結合層やプーリング層とは異なります。
* 注意メカニズムは、クエリ（意志的キュー）とキー（非意志的キュー）を組み込んだアテンションプーリングを介して、値（感覚入力）よりも選択にバイアスをかけます。キーと値はペアになっています。
* クエリとキーの間のアテンションウェイトを視覚化できます。

## 演習

1. 機械翻訳でシーケンストークンをトークンごとにデコードするときの意志の合図は何ですか？非意志的手がかりと感覚入力は何ですか？
1. $10 \times 10$ 行列を無作為に生成し、softmax 演算を使用して各行が有効な確率分布であることを確認します。出力アテンションウェイトを可視化します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:

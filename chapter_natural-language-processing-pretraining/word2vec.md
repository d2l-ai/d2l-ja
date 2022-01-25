# 単語埋め込み (word2vec)
:label:`sec_word2vec`

自然言語は意味を表現するために使われる複雑なシステムです。このシステムでは、単語は意味の基本単位です。その名のとおり
*単語ベクトル* は単語を表すために使われるベクトルで、
また、特徴ベクトルまたは単語の表現と見なすこともできます。単語を実数ベクトルにマッピングする手法を*単語埋め込み* と呼びます。近年、単語の埋め込みは自然言語処理の基礎知識になりつつあります。 

## ワンホットベクトルは悪い選択です

:numref:`sec_rnn_scratch` では、単語 (文字は単語) を表すためにワンホットベクトルを使用しました。辞書に含まれる単語の数 (辞書のサイズ) が $N$ で、各単語が $0$ から $N−1$ までの異なる整数 (インデックス) に対応するとします。インデックス $i$ の単語に対してワンホットベクトル表現を得るために、すべて 0 をもつ長さ $N$ ベクトルを作成し、$i$ の位置にある要素を 1 に設定します。このように、各単語は長さ $N$ のベクトルとして表され、ニューラルネットワークで直接使用できます。 

ワンホットワードベクトルは簡単に作成できますが、通常は適していません。主な理由は、よく使う*余弦類似度*のように、ワンホットワードベクトルでは異なる単語間の類似度を正確に表現できないからです。ベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ の場合、余弦類似度はベクトル間の角度の余弦になります。 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

2 つの異なる単語のワンホットベクトル間のコサイン類似度は 0 であるため、ワンホットベクトルは単語間の類似度をエンコードできません。 

## 自己教師ありword2vec

[word2vec](https://code.google.com/archive/p/word2vec/) ツールは、上記の問題に対処するために提案されました。各単語を固定長ベクトルにマッピングし、これらのベクトルは異なる単語間の類似性と類推関係をよりよく表現できます。word2vec ツールには、*スキップグラム* :cite:`Mikolov.Sutskever.Chen.ea.2013` と*連続した単語の袋* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013` の 2 つのモデルが含まれています。意味的に意味のある表現の場合、そのトレーニングは条件付き確率に依存しています。条件付き確率は、コーパス内で周囲の単語の一部を使用して一部の単語を予測すると見なすことができます。監視はラベルのないデータから行われるため、スキップグラムと連続した単語の袋はどちらも自己教師付きモデルです。 

以下では、この2つのモデルとそのトレーニング方法を紹介します。 

## スキップ・グラム・モデル
:label:`subsec_skip-gram`

*skip-gram* モデルでは、ある単語を使用して周囲の単語をテキストシーケンス内で生成できることを前提としています。テキストシーケンス「the」、「man」、「loves」、「his」、「son」を例にとります。*center word* として「loves」を選択し、コンテキストウィンドウのサイズを 2 に設定します。:numref:`fig_skip_gram` に示されているように、スキップグラムモデルは、中心語から 2 語以内の距離にある「the」、「man」、「his」、「son」の「context words」を生成する条件付き確率を考慮します。 

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

コンテキストワードはセンターワードから独立して生成される (つまり、条件付き独立性) と仮定します。この場合、上記の条件付き確率は次のように書き換えることができます。 

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg)
:label:`fig_skip_gram`

スキップグラムモデルでは、条件付き確率を計算するために、各単語に 2 つの $d$ 次元ベクトル表現があります。より具体的には、辞書内のインデックス $i$ を持つ単語について、$\mathbf{v}_i\in\mathbb{R}^d$ と $\mathbf{u}_i\in\mathbb{R}^d$ でそれぞれ*center* 単語と*context* 単語として使用される場合は、その 2 つのベクトルを表します。中心語 $w_c$ (ディクショナリのインデックス $c$) が与えられた場合にコンテキストワード $w_o$ (ディクショナリのインデックス $o$) が生成される条件付き確率は、ベクトルドット積に対するソフトマックス演算によってモデル化できます。 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

ボキャブラリーインデックスは $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ に設定されています。長さ $T$ のテキストシーケンスが与えられ、タイムステップ $t$ の単語は $w^{(t)}$ と表されます。コンテキストワードは、任意のセンターワードから独立して生成されると仮定します。コンテキストウィンドウサイズ $m$ の場合、スキップグラムモデルの尤度関数は、任意のセンターワードが与えられた場合にすべてのコンテキストワードを生成する確率です。 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

$1$ より小さい、または $T$ より大きいタイムステップは省略できます。 

### 訓練

スキップグラムモデルパラメーターは、語彙内の各単語の中心語ベクトルと文脈語ベクトルです。トレーニングでは、尤度関数を最大化 (最尤推定) することでモデルパラメーターを学習します。これは、次の損失関数を最小化することと等価です。 

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

確率的勾配降下法を使用して損失を最小化する場合、各反復で短い部分列をランダムにサンプリングして、この部分列の (確率的) 勾配を計算し、モデルパラメーターを更新できます。この (確率的) 勾配を計算するには、中心語ベクトルと文脈語ベクトルに関する対数条件付き確率の勾配を取得する必要があります。一般に、:eqref:`eq_skip-gram-softmax` によると、中心語 $w_c$ と文脈語 $w_o$ の任意のペアを含む対数条件付き確率は次のようになります。 

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

微分により、センターワードベクトル $\mathbf{v}_c$ に対する勾配を次のように求めることができます。 

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

:eqref:`eq_skip-gram-grad` の計算では、辞書内の $w_c$ を中心語とするすべての単語の条件付き確率が必要であることに注意してください。他の単語ベクトルの勾配も同じ方法で取得できます。 

学習後、辞書内のインデックス $i$ を持つ単語に対して、単語ベクトル $\mathbf{v}_i$ (中心語) と $\mathbf{u}_i$ (文脈語) の両方を取得します。自然言語処理アプリケーションでは、スキップグラムモデルの中心単語ベクトルが単語表現として一般的に使用されます。 

## 連続した言葉の袋 (CBOW) モデル

*連続した単語の袋* (CBOW) モデルは、スキップグラムモデルに似ています。スキップ・グラム・モデルとの大きな違いは、連続した単語バッグ・モデルでは、テキスト・シーケンス内の周囲のコンテキスト・ワードに基づいてセンター・ワードが生成されることを前提としている点です。たとえば、同じテキストシーケンス「the」、「man」、「loves」、「his」、「son」で、「loves」が中心語、コンテキストウィンドウのサイズが 2 の場合、連続単語バッグモデルは、コンテキストワード「the」、「man」、「his」、「son」に基づいて、中心語「loves」を生成する条件付き確率を考慮します。「(:numref:`fig_cbow` に示されるように)、これは次のようになります。 

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg)
:eqlabel:`fig_cbow`

連続語袋モデルには複数の文脈語が存在するため、これらの文脈語ベクトルは条件付き確率の計算で平均化されます。具体的には、辞書内のインデックス $i$ を持つ単語について、$\mathbf{v}_i\in\mathbb{R}^d$ と $\mathbf{u}_i\in\mathbb{R}^d$ で、*context* 単語と*center* 単語 (スキップグラムモデルでは意味が入れ替わる) としてそれぞれ 2 つのベクトルを表します。周囲の文脈語 $w_{o_1}, \ldots, w_{o_{2m}}$ (ディクショナリではインデックス $o_1, \ldots, o_{2m}$) からセンターワード $w_c$ (ディクショナリではインデックス $c$) が生成される条件付き確率は、次の式でモデル化できます。 

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

簡潔にするために、$\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ と $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$ としましょう。:eqref:`fig_cbow-full` は次のように簡略化できます。 

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

長さ $T$ のテキストシーケンスが与えられ、タイムステップ $t$ の単語は $w^{(t)}$ と表されます。文脈ウィンドウサイズ $m$ では、連続語袋モデルの尤度関数は、文脈語が与えられた場合にすべての中心語を生成する確率です。 

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### 訓練

連続した単語バッグモデルのトレーニングは、スキップグラムモデルのトレーニングとほとんど同じです。連続的な単語バッグモデルの最尤推定は、次の損失関数を最小化することと等価です。 

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

次の点に注意してください。 

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

微分により、任意のコンテキストワードベクトル $\mathbf{v}_{o_i}$ ($i = 1, \ldots, 2m$) に対する勾配を次のように求めることができます。 

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

他の単語ベクトルの勾配も同じ方法で取得できます。スキップグラムモデルとは異なり、連続的な単語バッグモデルでは通常、単語表現として文脈語ベクトルが使用されます。 

## [概要

* 単語ベクトルは単語を表すために使用されるベクトルで、特徴ベクトルまたは単語の表現と見なすこともできます。単語を実数ベクトルにマッピングする手法を単語埋め込みといいます。
* word2vec ツールには、スキップグラムモデルと連続単語バッグモデルの両方が含まれています。
* スキップ・グラム・モデルでは、単語を使用して周囲の単語をテキスト・シーケンスに生成できると想定しています。一方、連続的な単語バッグ・モデルでは、周囲のコンテキスト・ワードに基づいて中心語が生成されることを前提としています。

## 演習

1. 各勾配を計算するための計算の複雑さはどれくらいですか？辞書のサイズが大きいと何が問題になりますか？
1. 英語の固定フレーズには、「ニューヨーク」のように複数の単語で構成されるものがあります。単語ベクトルをどのように訓練するのですか？ヒント: see Section 4 in the word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`。
1. スキップグラムモデルを例に取って、word2vecのデザインを振り返ってみましょう。スキップグラムモデルにおける 2 つの単語ベクトルの内積と余弦類似度の関係は何ですか？セマンティクスが似ている単語のペアで、単語ベクトル (スキップグラムモデルで学習された) の余弦類似度が高くなるのはなぜですか?

[Discussions](https://discuss.d2l.ai/t/381)

# 近似トレーニング
:label:`sec_approx_train`

:numref:`sec_word2vec` での議論を思い出してください。スキップグラムモデルの主な目的は、softmax 演算を使用して :eqref:`eq_skip-gram-softmax` で指定されたセンターワード $w_c$ に基づいてコンテキストワード $w_o$ を生成する条件付き確率を計算することです。この対数損失は :eqref:`eq_skip-gram-log` の反対によって与えられます。 

softmax 演算の性質上、文脈語は辞書 $\mathcal{V}$ の任意のものになる可能性があるため、:eqref:`eq_skip-gram-log` の反対には、語彙全体のサイズと同じ数の項目の合計が含まれます。したがって、:eqref:`eq_skip-gram-grad` のスキップグラムモデルの勾配計算と :eqref:`eq_cbow-gradient` の連続バッグオブワードモデルの勾配計算の両方に、合計が含まれます。残念なことに、大きな辞書 (多くの場合、数十万または数百万の単語) を合計するような勾配の計算コストは莫大です。 

前述の計算の複雑さを軽減するために、このセクションでは 2 つの近似的なトレーニング方法を紹介します。
*負のサンプリング* と*階層 softmax*。
スキップグラムモデルと連続単語バッグモデルは類似しているため、スキップグラムモデルを例として取り上げて、これら 2 つの近似学習方法を説明します。 

## 負サンプリング
:label:`subsec_negative-sampling`

負のサンプリングは元の目的関数を変更します。センターワード $w_c$ のコンテキストウィンドウを考えると、任意の (コンテキスト) ワード $w_o$ がこのコンテキストウィンドウから来ているという事実は、次の式でモデル化された確率をもつ事象とみなされます。 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

$\sigma$ は、シグモイド活性化関数の定義を使用します。 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

まず、テキストシーケンス内のこのようなすべてのイベントの同時確率を最大化して、単語の埋め込みを学習させましょう。具体的には、長さ $T$ のテキストシーケンスがタイムステップ $t$ の単語を $w^{(t)}$ で表し、コンテキストウィンドウサイズを $m$ とすると、結合確率を最大化することを検討してください。 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

ただし、:eqref:`eq-negative-sample-pos` では、肯定的な例を含むイベントのみが考慮されます。その結果、:eqref:`eq-negative-sample-pos` の結合確率は、すべての単語ベクトルが無限大に等しい場合にのみ 1 に最大化されます。もちろん、そのような結果は無意味です。目的関数をより意味のあるものにするために、
*ネガティブサンプリング*
定義済みの分布からサンプリングされた負の例を加算します。 

コンテキストワード $w_o$ がセンターワード $w_c$ のコンテキストウィンドウから来るイベントを $S$ で表します。$w_o$ が関係するこの事象について、事前定義済みの分布 $P(w)$ から、このコンテキストウィンドウに由来しない $K$ *ノイズワード* をサンプルします。$w_c$ のコンテキストウィンドウからノイズワード $w_k$ ($k=1, \ldots, K$) が発生しないイベントを $N_k$ で表します。ポジティブな例とネガティブな例 $S, N_1, \ldots, N_K$ の両方が関係するこれらの事象は相互に独立していると仮定します。負のサンプリングは、:eqref:`eq-negative-sample-pos` の結合確率 (ポジティブな例のみを含む) を次のように書き換えます。 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

条件付き確率は事象 $S, N_1, \ldots, N_K$ によって近似されます。 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

テキストシーケンスのタイムステップ $t$ における単語 $w^{(t)}$ とノイズワード $w_k$ のインデックスをそれぞれ $i_t$ と $h_k$ で表します。:eqref:`eq-negative-sample-conditional-prob` の条件付き確率に関する対数損失は次のようになります。 

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

これで、各学習ステップでの勾配の計算コストはディクショナリサイズとは無関係ですが、$K$ に線形的に依存することがわかります。ハイパーパラメーター $K$ を小さい値に設定すると、負のサンプリングを使用した各学習ステップでの勾配の計算コストは小さくなります。 

## 階層的ソフトマックス

近似トレーニングの代替方法として、
*階層的softmax*
では、:numref:`fig_hi_softmax` に示すデータ構造であるバイナリツリーを使用します。ツリーの各リーフノードは、辞書 $\mathcal{V}$ の 1 つの単語を表します。 

![Hierarchical softmax for approximate training, where each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

ルートノードから二分木の単語 $w$ を表すリーフノードまでの経路上のノード (両端を含む) の数を $L(w)$ で表します。$n(w,j)$ をこのパス上の $j^\mathrm{th}$ ノードとし、コンテキストワードベクトルを $\mathbf{u}_{n(w, j)}$ とします。たとえば、:numref:`fig_hi_softmax` の場合は $L(w_3) = 4$ と入力します。階層的ソフトマックスは :eqref:`eq_skip-gram-softmax` の条件付き確率を次のように近似します。 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

関数 $\sigma$ は :eqref:`eq_sigma-f` で定義され、$\text{leftChild}(n)$ はノード $n$ の左の子ノードです。$x$ が真なら $ [\![x]\!]= 1$; otherwise $ [\![x]\!]= -1$。 

説明のために、:numref:`fig_hi_softmax` の単語 $w_c$ から単語 $w_3$ を生成する条件付き確率を計算してみましょう。これには $w_c$ のワードベクトル $\mathbf{v}_c$ と、ルートから $w_3$ までのパス (:numref:`fig_hi_softmax` の太字のパス) 上の非リーフノードベクトルとの間にドット積が必要です。$w_3$ は左、右、左に走査されます。 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$ 以降、辞書 $\mathcal{V}$ のすべての単語を任意の単語 $w_c$ に基づいて生成する条件付き確率は、合計で 1 になると考えられます。 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

幸いなことに、$L(w_o)-1$ は 2 分木構造のため $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ のオーダーであるため、辞書のサイズ $\mathcal{V}$ が大きい場合、階層的 softmax を使用する各学習ステップの計算コストは、近似学習を行わない場合と比較して大幅に削減されます。 

## [概要

* 負のサンプリングは、正と負の両方の例を含む相互に独立した事象を考慮して、損失関数を構成します。学習の計算コストは、各ステップでのノイズワードの数に線形的に依存します。
* Hierarchical softmax は、二分木のルートノードからリーフノードまでのパスを使用して損失関数を構築します。学習の計算コストは、各ステップでのディクショナリサイズの対数に依存します。

## 演習

1. ノイズワードをネガティブサンプリングでサンプリングするにはどうしたらいいですか？
1. :eqref:`eq_hi-softmax-sum-one` が成立することを確認します。
1. 負のサンプリングと階層的softmaxをそれぞれ使って連続した単語袋モデルを訓練する方法は?

[Discussions](https://discuss.d2l.ai/t/382)

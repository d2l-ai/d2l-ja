# グローバルベクトルを使用した単語の埋め込み (GLOVE)
:label:`sec_glove`

コンテキストウィンドウ内で単語と単語が共起すると、豊富なセマンティック情報が伝達される可能性があります。たとえば、大きなコーパスの単語では、「固体」は「蒸気」よりも「氷」と共存する可能性が高いですが、「ガス」という言葉は「氷」よりも「蒸気」と共存する頻度が高いでしょう。さらに、このような同時発生のグローバルコーパス統計を事前に計算することができ、これによりより効率的なトレーニングが可能になります。コーパス全体の統計情報を単語の埋め込みに活用するために、:numref:`subsec_skip-gram` のスキップグラムモデルを再検討し、共起回数などのグローバルコーパス統計を使用して解釈します。 

## グローバルコーパス統計によるスキップグラム
:label:`subsec_skipgram-global`

$q_{ij}$で単語$w_j$の条件付き確率$P(w_j\mid w_i)$を表すと、スキップグラムモデルで単語$w_i$が与えられると、次のようになります。 

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

ここで、インデックス $i$ の場合、ベクトル $\mathbf{v}_i$ と $\mathbf{u}_i$ はそれぞれ $w_i$ という単語をセンターワードとコンテキストワードとして表し、$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ はボキャブラリのインデックスセットです。 

コーパス内で複数回出現する可能性のある単語 $w_i$ を考えてみましょう。コーパス全体で、$w_i$ が中心語とされるすべての文脈語は、同じ要素の複数インスタンスを*許容する*multiset* $\mathcal{C}_i$ の単語インデックスを形成します。どの要素についても、そのインスタンスの数は*多重度* と呼ばれます。例を挙げて、単語 $w_i$ がコーパスに 2 回出現し、2 つの文脈ウィンドウで $w_i$ を中心語とする文脈語のインデックスが $k, j, m, k$ と $k, l, k, j$ であると仮定します。したがって、要素 $j, k, l, m$ の多重度がそれぞれ 2、4、1、1 であるマルチセット $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$。 

ここで、マルチセット $\mathcal{C}_i$ の要素 $j$ の多重度を $x_{ij}$ と表します。これは、コーパス全体で同じコンテキストウィンドウ内にある $w_j$ (コンテキストワード) とワード $w_i$ (中央ワード) のグローバル共起カウントです。このようなグローバルコーパス統計を使用すると、スキップグラムモデルの損失関数は次のようになります。 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

さらに $x_i$ によって、$w_i$ がセンターワードとして出現するコンテキストウィンドウ内のすべてのコンテキストワードの数を示します。これは $|\mathcal{C}_i|$ に相当します。$p_{ij}$ を条件付き確率 $x_{ij}/x_i$ とすると、センターワード $w_i$、:eqref:`eq_skipgram-x_ij` が与えられた場合にコンテキストワード $w_j$ を生成する条件付き確率を次のように書き換えることができます。 

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

:eqref:`eq_skipgram-p_ij` では、$-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ は、グローバルコーパス統計量の条件付き分布 $p_{ij}$ とモデル予測の条件付き分布 $q_{ij}$ のクロスエントロピーを計算します。この損失も、上で説明したように $x_i$ によって重み付けされます。:eqref:`eq_skipgram-p_ij` で損失関数を最小化すると、予測される条件付き分布をグローバルコーパス統計の条件付き分布に近づけることができます。 

確率分布間の距離を測定するためによく使用されますが、クロスエントロピー損失関数はここでは適さない場合があります。一方では、:numref:`sec_approx_train` で述べたように、$q_{ij}$ を適切に正規化するとボキャブラリ全体の合計が得られ、計算コストが高くなる可能性があります。一方、大きなコーパスからの多数の希少事象は、重み付けしすぎるクロスエントロピー損失によってモデル化されることが多い。 

## GLOVEモデル

このことから、*Glove* モデルは、二乗損失 :cite:`Pennington.Socher.Manning.2014` に基づいて、スキップグラムモデルに 3 つの変更を加えます。 

1. $p'_{ij}=x_{ij}$ と $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ という変数を使う 
これは確率分布ではなく、両方の対数を取るため、損失の二乗項は $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$ になります。
2. 単語 $w_i$ ごとに 2 つのスカラーモデルパラメーター (センターワードバイアス $b_i$ とコンテキストワードバイアス $c_i$) を追加します。
3. 各損失項の重みを重み関数 $h(x_{ij})$ に置き換えます。$h(x)$ は $[0, 1]$ の区間で増加します。

すべてをまとめると、GloVE のトレーニングは次の損失関数を最小化することです。 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

加重関数では、$x < c$ ($c = 100$ など) の場合は $h(x) = (x/c) ^\alpha$ ($\alpha = 0.75$ など)、それ以外の場合は $h(x) = 1$ を選択することをお勧めします。この場合、$h(0)=0$ であるため、任意の $x_{ij}=0$ の二乗損失項は計算効率のために省略できます。たとえば、学習にミニバッチ確率的勾配降下法を使用する場合、各反復で*非ゼロ* $x_{ij}$ のミニバッチをランダムにサンプリングして勾配を計算し、モデルパラメーターを更新します。これらのゼロ以外の $x_{ij}$ は事前計算されたグローバルコーパス統計であるため、このモデルは*Global Vectors* の Glove と呼ばれます。 

単語 $w_j$ のコンテキストウィンドウに単語 $w_i$ が表示された場合、*その逆* が強調されます。したがって、$x_{ij}=x_{ji}$。非対称条件付き確率 $p_{ij}$ に適合する word2vec とは異なり、GLOVE は対称的な $\log \, x_{ij}$ に適合します。したがって、GLOVE モデルでは、任意の単語の中心語ベクトルと文脈語ベクトルは数学的に同等です。ただし、実際には、初期化値が異なるため、学習後も同じ単語がこれら 2 つのベクトルで異なる値を取得することがあります。GLOVE は、これらを出力ベクトルとして合計します。 

## 共起確率の比率からGLOVEを解釈する

また、GLOVE モデルを別の観点から解釈することもできます。:numref:`subsec_skipgram-global` で同じ表記法を使用して、$w_i$ をコーパスの中心語として与えられた文脈語 $w_j$ を生成する条件付き確率を $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ とします。:numref:`tab_glove` は、"ice」と「steam」という単語が与えられた複数の共起確率と、それらの比率を大きなコーパスからの統計に基づいてリストします。 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

:numref:`tab_glove` から次のことがわかります。 

* $w_k=\text{solid}$ のように、「氷」に関連しているが「蒸気」とは無関係な単語 $w_k$ の場合、共起確率の比は 8.9 のように大きくなると予想されます。
* $w_k=\text{gas}$ のように、「蒸気」に関連しているが「氷」とは無関係な単語 $w_k$ の場合、共起確率の比は 0.085 のように小さくなると予想されます。
* $w_k=\text{water}$ のように「氷」と「蒸気」の両方に関連する単語 $w_k$ の場合、共起確率の比は 1.36 のように 1 に近いと予想されます。
* $w_k=\text{fashion}$ のように「氷」と「蒸気」の両方に関係のない単語 $w_k$ の場合、共起確率の比は 0.96 のように 1 に近いと予想されます。

共起確率の比率は、単語間の関係を直感的に表現できることが分かります。したがって、この比率に合うように 3 つの単語ベクトルの関数を設計できます。$w_i$ がセンターワードで $w_j$ と $w_k$ がコンテキストワードである共起確率 ${p_{ij}}/{p_{ik}}$ の比率については、関数 $f$ を使用してこの比率を近似します。 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

$f$ の考えられる多くの設計の中から、次の中から妥当な選択肢のみを選択します。共起確率の比はスカラーなので、$f$ は $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$ のようなスカラー関数である必要があります。:eqref:`eq_glove-f` でワードインデックス $j$ と $k$ を切り替える場合、その $f(x)f(-x)=1$ を保持しなければならないので、1 つの可能性は $f(x)=\exp(x)$ です。  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

ここで $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$ を選びましょう。$\alpha$ は定数です。$p_{ij}=x_{ij}/x_i$以降、両辺の対数をとると、$\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$が得られます。$- \log\, \alpha + \log\, x_i$ をあてはめるために、センターワードのバイアス $b_i$ やコンテキストワードのバイアス $c_j$ など、追加のバイアス項を使用することもできます。 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

:eqref:`eq_glove-square` の二乗誤差を重み付きで測定すると、:eqref:`eq_glove-loss` の GLOVE 損失関数が得られます。 

## [概要

* スキップグラムモデルは、単語と単語の共起回数などのグローバルコーパス統計を使用して解釈できます。
* クロスエントロピー損失は、特にコーパスが大きい場合、2 つの確率分布の差の測定には適さない場合があります。GLOVE は、二乗損失を使用して、事前計算されたグローバルコーパス統計を近似します。
* センターワードベクトルとコンテキストワードベクトルは、GLOVE 内のどの単語でも数学的に同等です。
* GLOVE は、単語と単語の共起確率の比率から解釈できます。

## 演習

1. $w_i$ と $w_j$ という単語が同じコンテキストウィンドウに共存する場合、条件付き確率 $p_{ij}$ を計算する方法を再設計するために、テキストシーケンスでそれらの距離をどのように使用すればよいのでしょうか。ヒント: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`。
1. どの単語でも、その中心語バイアスと文脈語バイアスはGLOVEで数学的に同等ですか？なぜ？

[Discussions](https://discuss.d2l.ai/t/385)
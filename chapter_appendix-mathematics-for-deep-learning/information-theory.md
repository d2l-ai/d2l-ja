# 情報理論
:label:`sec_information_theory`

宇宙は情報であふれている。シェイクスピアのソネットから Cornell arXiv に関する研究者の論文、ヴァン・ゴッホの『星月夜』、ベートーヴェンの音楽「交響曲第5番」、最初のプログラミング言語Plankalkül、最先端の機械学習アルゴリズムまで、さまざまな分野に共通する言葉が情報によって提供されます。形式に関係なく、すべてが情報理論のルールに従わなければなりません。情報理論により、さまざまな信号にどれだけの情報が存在するかを測定して比較することができます。本セクションでは、情報理論の基礎概念と情報理論の機械学習への応用について考察する。 

始める前に、機械学習と情報理論の関係について概説しましょう。機械学習は、データから興味深い信号を抽出し、重要な予測を行うことを目的としています。一方、情報理論では、情報の符号化、復号化、伝送、操作について研究しています。その結果、情報理論は機械学習システムにおける情報処理を議論するための基礎言語となる。たとえば、:numref:`sec_softmax` で説明されているように、多くの機械学習アプリケーションではクロスエントロピー損失が使用されます。この損失は、情報理論上の考察から直接導き出すことができます。 

## 情報

情報理論の「魂」である情報から始めましょう。*Information* は、1 つまたは複数のエンコードフォーマットの特定のシーケンスを使用して、何にでもエンコードできます。情報の概念を定義しようと自分自身に課すと仮定します。私たちの出発点は何でしょうか？ 

次の思考実験を考えてみましょう。一組のカードを持つ友人がいる。彼らはデッキをシャッフルし、いくつかのカードを裏返し、カードについての声明を教えてくれます。各声明の情報内容を評価するよう努めます。 

まず、カードをひっくり返して、「カードが見える」と言ってくれます。これは私たちに全く情報を提供しません。これが事実であることはすでに確信していたので、情報がゼロであることを願っています。 

次に、カードをひっくり返して「ハートが見える」と言います。これは私たちにいくらかの情報を提供しますが、実際には$4$種類の訴訟しかできず、それぞれ同じ可能性が高いため、この結果に驚くことはありません。情報の尺度が何であれ、このイベントの情報量が少ないことを願っています。 

次に、カードをひっくり返して、「これはスペードの$3$だ」と言います。これはもっと詳しい情報です。実際、$52$も同様に起こり得る結果があり、私たちの友人はそれがどれであるかを教えてくれました。これは中程度の情報量である必要があります。 

これを論理的に極端に考えてみましょう。最後に、デッキのすべてのカードを裏返して、シャッフルされたデッキのシーケンス全体を読み取るとします。52ドルあります！$デッキへのさまざまな注文、再びすべて同じ可能性が高いため、どれであるかを知るには多くの情報が必要です。 

私たちが開発する情報の概念は、この直感に従わなければなりません。実際、次のセクションでは、これらのイベントがそれぞれ $0\text{ bits}$、$2\text{ bits}$、$~5.7\text{ bits}$, and $~225.6\text{ bits}$ の情報を持つことを計算する方法を学びます。 

これらの思考実験を読み通せば、自然な発想がわかります。出発点として、知識に気を配るのではなく、情報は驚きの度合いや出来事の抽象的な可能性を表すという考え方から構築されるかもしれません。たとえば、異常なイベントを記述する場合は、多くの情報が必要です。一般的なイベントでは、あまり情報が必要ない場合があります。 

In 1948, Claude E. Shannon published *A Mathematical Theory of Communication* :cite:`Shannon.1948` establishing the theory of information.  In his article, Shannon introduced the concept of information entropy for the first time. We will begin our journey here. 

### 自己情報

情報は事象の抽象的な可能性を体現しているので、その可能性をビット数にどのようにマッピングすればよいのでしょうか。シャノンは情報の単位として*bit*という用語を導入しました。この用語は、もともとジョン・テューキーによって作成されました。では、「ビット」とは何で、なぜそれを情報の測定に使うのでしょうか？従来、アンティーク送信機は $0$ と $1$ の 2 種類のコードしか送受信できません。実際、バイナリエンコーディングは、現代のすべてのデジタルコンピュータで今でも一般的に使用されています。このようにして、すべての情報は $0$ と $1$ の系列でエンコードされます。したがって、長さ $n$ の一連の 2 進数には $n$ ビットの情報が含まれます。 

ここで、一連のコードに対して $0$ または $1$ がそれぞれ $\frac{1}{2}$ の確率で発生すると仮定します。したがって、$n$ の長さの一連のコードをもつイベント $X$ は $\frac{1}{2^n}$ の確率で発生します。同時に、前述したように、このシリーズには$n$ビットの情報が含まれています。では、確率$p$をビット数に伝達できる数学関数に一般化できますか？シャノンは*自己情報*を定義することで答えを出しました 

$$I(X) = - \log_2 (p),$$

このイベントについて受け取った情報の「ビット」として $X$このセクションでは常に 2 を底とする対数を使用することに注意してください。わかりやすくするために、この節の残りの部分では対数表記の添字 2 を省略します。つまり、$\log(.)$ は常に $\log_2(.)$ を指します。たとえば、コード「0010」には自己情報があります。 

$$I(\text{"0010"}) = - \log (p(\text{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \text{ bits}.$$

自己情報は以下のように計算できます。その前に、まずこのセクションで必要なパッケージをすべてインポートしておきましょう。

```{.python .input}
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch doesn't offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## エントロピー

自己情報は単一の離散事象の情報のみを測定するため、離散分布または連続分布のいずれかの確率変数に対して、より一般化された尺度が必要です。 

### やる気を起こさせるエントロピー

私たちが望むものについて具体的に考えてみましょう。これは、*シャノンエントロピーの公理*として知られているものの非公式な声明になるでしょう。以下の常識的な声明の集まりは、私たちに情報の独自の定義を強いることが分かります。これらの公理の正式なバージョンは、他のいくつかの公理とともに :cite:`Csiszar.2008` にあります。 

1.  確率変数を観測することによって得られる情報は、要素と呼ばれるものや、確率がゼロの追加の要素の存在に依存しません。
2.  2つの確率変数を観測することで得られる情報は、それらを別々に観測することで得られる情報の合計にすぎません。それらが独立していれば、それはまさにその合計です。
3.  特定のイベントを (ほぼ) 観測したときに得られる情報は (ほぼ) ゼロです。

この事実を証明することは私達のテキストの範囲を超えているが、エントロピーがとらなければならない形態を独自に決定することを知ることは重要である。これらが許す唯一のあいまいさは、基本単位の選択にあります。これは、単一の公正なコインフリップによって提供される情報が1ビットであるという選択をすることによって最も頻繁に正規化されます。 

### 定義

確率密度関数 (p.d.f.) または確率質量関数 (p.mf.) $p(x)$ で確率分布 $P$ に従う確率変数 $X$ について、*エントロピー* (または*シャノンエントロピー*) によって期待される情報量を測定します。 

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

具体的には、$X$ がディスクリートの場合、$H(X) = - \sum_i p_i \log p_i \text{, where } p_i = P(X_i).$ ドル 

そうでなければ、$X$ が連続であれば、エントロピーも*微分エントロピー* と呼びます。 

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

エントロピーは以下のように定義できます。

```{.python .input}
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### 解釈

あなたは好奇心が強いかもしれません: in the entropy definition :eqref:`eq_ent_def`、なぜ負の対数の期待値を使うのですか？ここにいくつかの直感があります。 

まず、なぜ*対数*関数$\log$を使うのですか？$p(x) = f_1(x) f_2(x) \ldots, f_n(x)$ で、各成分関数 $f_i(x)$ が互いに独立しているとします。つまり、各 $f_i(x)$ は $p(x)$ から取得した情報全体に対して独立して寄与します。上で説明したように、エントロピー式は独立確率変数に対して加法性である必要があります。幸いなことに、$\log$ は、確率分布の積を個々の項の和に自然に変換できます。 

次に、なぜ*負* $\log$を使うのですか？直感的には、より頻繁なイベントには、あまり一般的でないイベントよりも少ない情報を含める必要があります。これは、通常のケースよりも異常なケースからより多くの情報を得ることが多いためです。ただし、$\log$ は確率とともに単調に増加し、$[0, 1]$ のすべての値で負になります。事象の確率とエントロピーとの間に単調に減少する関係を構築する必要があります。これは理想的には常に正です（私たちが観察したことは何もないので、私たちが知っていることを忘れさせることはありません）。したがって、$\log$ 関数の前に負の符号を追加します。 

最後に、*expectation*関数はどこから来たのですか？確率変数 $X$ を考えてみましょう。自己情報 ($-\log(p)$) は、特定の結果を見たときの*驚き*の量と解釈できます。確かに、確率がゼロに近づくにつれて、驚きは無限大になります。同様に、エントロピーは $X$ を観測したときの驚きの平均量として解釈できます。たとえば、スロットマシンシステムが、確率が ${p_1, \ldots, p_k}$ の統計的に独立したシンボル ${s_1, \ldots, s_k}$ を放出するとします。このシステムのエントロピーは、各出力を観測することによる平均的な自己情報と等しくなります。 

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$

### エントロピーの性質

以上の例と解釈により、エントロピー:eqref:`eq_ent_def`の以下の性質を導き出すことができる。ここでは、$X$ を事象として、$P$ を $X$ の確率分布と呼びます。 

* 高さ (X)\ geq 0$ for all discrete $X$ (entropy can be negative for continuous $X$)。

* $X \sim P$ に p.f.f. または午後 $p(x)$ を指定し、新しい確率分布 $Q$ で $P$ を推定しようとすると、$q(x)$、$$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)], \text{ with equality if and only if } P = Q.$$  Alternatively, $H (X) $ は、次の処理に必要な平均ビット数の下限になります。$P$ から引き出されたシンボルをエンコードします。

* $X \sim P$ の場合、$x$ は、考えられるすべての結果に均等に分散する場合に最大量の情報を伝達します。具体的には、確率分布 $P$ が $k$ クラス $\{p_1, \ldots, p_k \}$ で離散的である場合、$H(X) \leq \log(k), \text{ with equality if and only if } p_i = \frac{1}{k}, \forall i.$$ $P$ が連続確率変数である場合、ストーリーはより複雑になります。ただし、$P$ が有限区間 (すべての値が $0$ and $1 $ の間) でサポートされることをさらに強制すると、$P$ がその区間の一様分布であれば $P$ のエントロピーが最も高くなります。

## 相互情報

以前、単一確率変数 $X$ のエントロピーを定義しましたが、一対の確率変数 $(X, Y)$ のエントロピーはどうですか？これらの手法は、「$X$ と $Y$ には、それぞれ別々にどのような情報が一緒に含まれているのか」という質問に答えようとしていると考えることができます。冗長な情報があるのか、それともすべてが一意なのか？」 

以下の説明では、$(X, Y)$ は、確率分布 $P$ と午後の結合確率分布 $p_{X, Y}(x, y)$ に従う確率変数のペアとして常に使用し、$X$ と $Y$ は確率分布 $p_X(x)$ と $p_Y(y)$ にそれぞれ従います。 

### ジョイントエントロピー

単一確率変数 :eqref:`eq_ent_def` のエントロピーと同様に、一対の確率変数 $(X, Y)$ の*ジョイントエントロピー* $H(X, Y)$ を次のように定義します。 

$$H(X, Y) = −E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

正確には、一方では $(X, Y)$ が離散確率変数のペアである場合、 

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

一方、$(X, Y)$ が連続確率変数のペアである場合、*微分ジョイントエントロピー* を次のように定義します。 

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

:eqref:`eq_joint_ent_def` は、確率変数のペアの全ランダム性を示すものと考えることができます。極値のペアとして、$X = Y$ が 2 つの同一の確率変数である場合、ペアの情報は正確に 1 つの情報になり、$H(X, Y) = H(X) = H(Y)$ になります。一方、$X$ と $Y$ が独立していれば $H(X, Y) = H(X) + H(Y)$ になります。実際、確率変数のペアに含まれる情報は、いずれかの確率変数のエントロピーより小さくはなく、両方の合計以下であることが常にわかります。 

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

ジョイントエントロピーをゼロから実装してみましょう。

```{.python .input}
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

これは前と同じ*code* ですが、今度は 2 つの確率変数の同時分布を扱うものとして解釈が異なります。 

### 条件付きエントロピー

一対の確率変数に含まれる情報量より上に定義されるジョイントエントロピー。これは便利ですが、私たちが気にかけていることではないことがよくあります。機械学習の設定を考えてみましょう。$X$ をイメージのピクセル値を記述する確率変数 (または確率変数のベクトル)、$Y$ をクラスラベルである確率変数とします。$X$ にはかなりの情報が含まれているはずです。自然なイメージは複雑なものです。ただし、イメージが表示された後の $Y$ に含まれる情報は低くなるはずです。実際、数字の画像には、数字が判読不能でない限り、数字が何桁であるかについての情報がすでに含まれているはずです。したがって、情報理論の語彙を拡張し続けるためには、別の条件付き確率変数内の情報内容について推論できる必要があります。 

確率論では、変数間の関係を測る*条件付き確率*の定義を見た。ここで、*条件付きエントロピー* $H(Y \mid X)$ を同様に定義したいと思います。これを次のように書くことができます。 

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

$p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$ は条件付き確率です。具体的には、$(X, Y)$ が離散確率変数のペアである場合、 

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

$(X, Y)$ が連続確率変数のペアである場合、*微分条件付きエントロピー* は次のように定義されます。 

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$

*条件付きエントロピー* $H(Y \mid X)$は、エントロピー$H(X)$とジョイントエントロピー$H(X, Y)$とどのように関連しているのか、と尋ねるのは自然なことです。上記の定義を使うと、これをきれいに表現できます。 

$$H(Y \mid X) = H(X, Y) - H(X).$$

これは直観的に解釈できます。$X$ ($H(Y \mid X)$) に与えられた $Y$ の情報は $X$ と $Y$ の両方に含まれる情報 ($H(X, Y)$) からすでに $X$ に含まれている情報を引いたものと同じです。これにより $Y$ の情報が得られますが、$X$ にも表されていません。 

さて、条件付きエントロピー :eqref:`eq_cond_ent_def` をゼロから実装してみましょう。

```{.python .input}
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### 相互情報

確率変数 $(X, Y)$ の以前の設定を考えてみると、「$Y$ には含まれているが$X$ にはどれだけの情報が含まれていないかがわかったので、$X$ と $Y$ の間で共有されている情報の量を同様に尋ねることはできますか？」答えは$(X, Y)$の*相互情報*で、$I(X, Y)$と書きます。 

形式的な定義にそのまま飛び込むのではなく、前に構築した用語に完全に基づいて相互情報の表現を導き出すことで、直感を実践しましょう。2つの確率変数間で共有される情報を求めます。これを行う方法の 1 つは、$X$ と $Y$ の両方に含まれるすべての情報をまとめてから、共有されていない部分を削除することです。$X$ と $Y$ の両方に含まれる情報は $H(X, Y)$ と書かれています。$X$ には含まれていても $Y$ には含まれていない情報と $Y$ には含まれていても $X$ には含まれていない情報をこれから差し引く必要があります。前のセクションで見たように、これはそれぞれ $H(X \mid Y)$ と $H(Y \mid X)$ で与えられます。したがって、私たちは相互情報が 

$$
I(X, Y) = H(X, Y) - H(Y \mid X) − H(X \mid Y).
$$

実際、これは相互情報の有効な定義です。これらの用語の定義を拡張して組み合わせると、小さな代数からすると、これは次の式と同じであることがわかります。 

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`

イメージ :numref:`fig_mutual_information` では、これらすべての関係をまとめることができます。次の文がすべて $I(X, Y)$ と同等である理由を理解することは、直感の優れたテストです。 

* $H (X) − H (X\ ミッドY) $
* $H (Y) − H (Y\ 中央 X) $
* $H (X) + H (Y) − H (X, Y) $

![Mutual information's relationship with joint entropy and conditional entropy.](../img/mutual-information.svg)
:label:`fig_mutual_information`

相互情報 :eqref:`eq_mut_ent_def` は、いろいろな意味で :numref:`sec_random_variables` で見られた相関係数の原則的な拡張と考えることができます。これにより、変数間の線形関係だけでなく、あらゆる種類の2つの確率変数間で共有される最大限の情報を求めることができます。 

それでは、相互情報をゼロから実装しましょう。

```{.python .input}
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # Operator `nansum` will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### 相互情報の性質

相互情報 :eqref:`eq_mut_ent_def` の定義を覚えるのではなく、その注目すべき特性を覚えておくだけです。 

* 相互情報は対称的です。つまり $I(X, Y) = I(Y, X)$ です。
* 相互情報は負ではない、つまり $I(X, Y) \geq 0$ です。
* $I(X, Y) = 0$ は $X$ と $Y$ が独立している場合に限ります。たとえば、$X$ と $Y$ が独立している場合、$Y$ を知っていても $X$ に関する情報は得られず、その逆も同様であるため、相互の情報はゼロになります。
* あるいは、$X$ が $Y$ の可逆関数である場合、$Y$ と $X$ はすべての情報を共有し、$I(X, Y) = H(Y) = H(X).$$

### ポイントワイズ相互情報

この章の冒頭でエントロピーを扱った時、$-\log(p_X(x))$ の解釈を、その特定の結果にどれほど驚いたかという解釈を出すことができました。相互情報における対数項と同様の解釈をすることができます。対数項は、しばしば*ポイントワイズ相互情報*と呼ばれます。 

$$\mathrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

:eqref:`eq_pmi_def` は、結果の $x$ と $y$ の特定の組み合わせが、独立したランダムな結果に対して予想されるものと比較される可能性がどれだけ高いか低いかを測定するものと考えることができます。大きくて正の場合、これら2つの特定の結果は、ランダムチャンスに比べてはるかに頻繁に発生します (*注*: 分母は $p_X(x) p_Y(y)$ で、2つの結果が独立している確率です)。一方、大きくて負の場合は、2つの結果が遠くに発生していることを表します。偶然に予想されるよりも少ない 

これにより、相互情報 :eqref:`eq_mut_ent_def` を、2 つの結果が共に発生するのを見て驚いた平均量を、それらが独立している場合に予想されるものと比較して解釈することができます。 

### 相互情報の応用

相互情報は純粋な定義では少し抽象的かもしれないが、機械学習とどう関係しているのか？自然言語処理において最も難しい問題の一つは、単語の意味が文脈からはっきりしないという*曖昧性の解決*です。たとえば、最近、ニュースの見出しで「Amazonが燃えている」と報じられました。アマゾン社に建物が燃えているのか、アマゾンの熱帯雨林が燃えているのか疑問に思うかもしれません。 

この場合、相互情報はこのあいまいさを解決するのに役立ちます。まず、eコマース、テクノロジー、オンラインなど、Amazon社との相互情報が比較的大きい単語のグループを見つけます。第二に、雨、森、熱帯など、アマゾンの熱帯雨林との相互情報が比較的大きい別の単語グループを見つけます。「Amazon」を明確にする必要がある場合、Amazonという単語の文脈でどのグループがより多く出現しているかを比較できます。この場合、記事は森を説明し、文脈を明確にするために続けられます。 

## カルバック・ライブラー・ダイバージェンス

:numref:`sec_linear-algebra` で説明したように、ノルムを使用して、任意の次元の空間における 2 点間の距離を測定できます。確率分布でも同様のタスクを実行できるようにしたいと考えています。これについては多くの方法がありますが、情報理論は最も素晴らしい方法の1つです。ここで、2つの分布が近接しているかどうかを測定する方法を提供する、*Kullback—Leibler (KL) ダイバージェンス*について調べます。 

### 定義

確率分布 $P$ にp.d.f.またはp.m.f.$p(x)$ で従う確率変数 $X$ を仮定し、$P$ を別の確率分布 $Q$ でp.d.f. または午後 $q(x)$ で推定します。$P$ と $Q$ の間の*カルバック・ライブラー (KL) ダイバージェンス* (または*相対エントロピー*) は次のようになります。 

$$D_{\mathrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

ポイントワイズ相互情報 :eqref:`eq_pmi_def` と同様に、対数項の解釈を再び提供できます。$x$ が $Q$ で予想されるよりもはるかに頻繁に見られる場合 $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$ は大きくて正になり、予想よりはるかに少ない結果が見られる場合は大きくて負になります。このようにして、参照分布から観察するとどれほど驚いたかと比較して、結果を観察したときの*相対的*驚きと解釈できます。 

KLダイバージェンスをゼロから実装してみましょう。

```{.python .input}
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### KL 発散プロパティ

KLダイバージェンス :eqref:`eq_kl_def` のいくつかの特性を見てみましょう。 

* KLダイバージェンスは非対称です。つまり、$D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P).$ドルになるような $P,Q$ があります。
* クアラルンプールダイバージェンスは負ではありません。つまり、$$D_{\mathrm{KL}}(P\|Q) \geq 0.$$ Note that the equality holds only when $P = Q$ となります。
* $p(x) > 0$ と $q(x) = 0$ のような $x$ が存在する場合、$D_{\mathrm{KL}}(P\|Q) = \infty$ になります。
* KLダイバージェンスと相互情報には密接な関係がある。:numref:`fig_mutual_information` に示された関係以外に、$I(X, Y)$ は次の項と数値的に同等です。
    1. $D_{\mathrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\mathrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\mathrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$。

  第1項では、相互情報量を $P(X, Y)$ と $P(X)$ と $P(Y)$ の積との間のKLダイバージェンスとして解釈します。したがって、ジョイント分布が独立している場合、ジョイント分布が分布とどの程度異なるかを示す尺度になります。2 番目の項では、相互情報から $X$ の分布の値を学習した結果、$Y$ に関する不確実性が平均的に減少することがわかります。第3項と同様です。 

### 例

おもちゃの例を見て、非対称性を明示的に見てみましょう。 

まず、長さ $10,000$ の 3 つのテンソルを生成してソートします。1 つは正規分布 $N(0, 1)$ に従う目的テンソル $p$ と、それぞれ正規分布 $N(-1, 1)$ と $N(1, 1)$ に従う 2 つの候補テンソル $q_1$ と $q_2$ です。

```{.python .input}
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

$q_1$ と $q_2$ は Y 軸に対して対称であるため (つまり $x=0$)、$D_{\mathrm{KL}}(p\|q_1)$ と $D_{\mathrm{KL}}(p\|q_2)$ の間で同様の KL ダイバージェンスが予想されます。ご覧のとおり、$D_{\mathrm{KL}}(p\|q_1)$と$D_{\mathrm{KL}}(p\|q_2)$の間は 3% 未満の割引しかありません。

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

対照的に、$D_{\mathrm{KL}}(q_2 \|p)$と$D_{\mathrm{KL}}(p \| q_2)$は大幅にオフになっており、以下に示すように約40％オフになっていることがわかります。

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## クロスエントロピー

ディープラーニングにおける情報理論の応用について知りたい方は、簡単な例を挙げてみましょう。真の分布 $P$ は確率分布 $p(x)$ で、推定分布 $Q$ は確率分布 $q(x)$ で定義し、このセクションの残りの部分で使用します。 

与えられた $n$ のデータ例 {$x_1, \ldots, x_n$} に基づいて二項分類問題を解く必要があるとします。$1$ と $0$ をそれぞれ陽性および陰性のクラスラベル $y_i$ としてエンコードし、ニューラルネットワークが $\theta$ によってパラメーター化されると仮定します。$\hat{y}_i= p_{\theta}(y_i \mid x_i)$ となる最良の $\theta$ を見つけることを目指すのであれば、:numref:`sec_maximum_likelihood` に見られるような最大対数尤度アプローチを適用するのが自然です。具体的には、真のラベル $y_i$ と予測 $\hat{y}_i= p_{\theta}(y_i \mid x_i)$ の場合、陽性と分類される確率は $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$ です。したがって、対数尤度関数は次のようになります。 

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

対数尤度関数 $l(\theta)$ の最大化は $- l(\theta)$ の最小化と同じなので、ここから最良の $\theta$ を見つけることができます。上記の損失を任意の分布に一般化するために、$-l(\theta)$ を*クロスエントロピー損失* $\mathrm{CE}(y, \hat{y})$ と呼びます。$y$ は真の分布 $P$ に続き、$\hat{y}$ は推定分布 $Q$ に従います。 

これはすべて、最尤の観点から作業することによって導き出されました。しかし、よく見ると $\log(\pi_i)$ のような項が計算に入っていることが分かります。これは、情報理論の観点から表現を理解できるという確かな指標です。 

### 形式的定義

KL ダイバージェンスと同様に、確率変数 $X$ に対して、推定分布 $Q$ と真の分布 $P$ の間のダイバージェンスを*クロスエントロピー* で測定することもできます。 

$$\mathrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

上で説明したエントロピーの特性を利用することで、エントロピー $H(P)$ と $P$ と $Q$ の間のKLダイバージェンスの和として解釈することもできます。 

$$\mathrm{CE} (P, Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$

クロスエントロピー損失は以下のように実装できます。

```{.python .input}
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # `tf.gather_nd` is used to select specific indices of a tensor.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

ラベルと予測に 2 つのテンソルを定義し、それらのクロスエントロピー損失を計算します。

```{.python .input}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### プロパティ

このセクションの冒頭で触れたように、クロスエントロピー :eqref:`eq_ce_def` は最適化問題の損失関数を定義するのに使用できます。次のことが同等であることが判明しました。 

1. 分布 $P$ に対する予測確率 $Q$ の最大化 (つまり、$E_ {x
\ sim P} [\ log (q (x))] $);
1. クロスエントロピーを最小化する $\mathrm{CE} (P, Q)$;
1. KL ダイバージェンスの最小化 $D_{\mathrm{KL}}(P\|Q)$

クロスエントロピーの定義は、真のデータ $H(P)$ のエントロピーが一定である限り、目的 2 と目的 3 の等価関係を間接的に証明します。 

### マルチクラス分類の目的関数としてのクロスエントロピー

クロスエントロピー損失 $\mathrm{CE}$ をもつ分類目的関数を深く掘り下げてみると、$\mathrm{CE}$ を最小化することは対数尤度関数 $L$ の最大化に等しいことがわかります。 

まず、$n$ の例を含むデータセットが与えられ、$k$ クラスに分類できるとします。データ例 $i$ ごとに、任意の $k$ クラスラベル $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$ を*ワンホットエンコーディング* で表します。具体的には、例の $i$ がクラス $j$ に属する場合、$j$ 番目のエントリを $1$ に設定し、その他のすべてのコンポーネントを $0$ に設定します。つまり、 

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\text{otherwise.}\end{cases}$$

たとえば、複数クラス分類問題に $A$、$B$、$C$ の 3 つのクラスが含まれている場合、ラベル $\mathbf{y}_i$ は {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$} でエンコードできます。 

ニューラルネットワークが $\theta$ によってパラメーター化されていると仮定します。真のラベルベクトル $\mathbf{y}_i$ と予測の場合 $\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$ 

したがって、*クロスエントロピー損失*は次のようになります。 

$$
\mathrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

反対に、最尤推定によって問題にアプローチすることもできます。はじめに、$k$ クラスのマルチヌーイ分布を簡単に紹介しましょう。これはベルヌーイ分布をバイナリクラスからマルチクラスに拡張したものです。確率変数 $\mathbf{z} = (z_{1}, \ldots, z_{k})$ が確率$\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$) をもつ $k$ クラス *マルチヌーイ分布* に従う場合、つまり $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \mathrm{Multi} (p_1, \ldots, p_k), \text{ where } \sum_{i=1}^k p_i = 1,$$ then the joint probability mass function(p.m.f.) of $\ mathbf {z} $ は $$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$ $になります 

各データ例のラベル $\mathbf{y}_i$ は、確率が $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$) の $k$ クラスのマルチヌーイ分布に従っていることがわかります。したがって、各データ例 $\mathbf{y}_i$ のジョイント午後は $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$ となります。したがって、対数尤度関数は次のようになります。 

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

最尤推定では、$\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$ を持つことで目的関数 $l(\theta)$ を最大化します。したがって、マルチクラス分類の場合、上記の対数尤度関数 $l(\theta)$ を最大化することは、CE 損失 $\mathrm{CE}(y, \hat{y})$ を最小化することと等価です。 

上記の証明をテストするために、組み込みのメジャー `NegativeLogLikelihood` を適用してみましょう。前の例と同じ `labels` と `preds` を使用すると、前の例と同じ小数点以下 5 桁までの数値損失が得られます。

```{.python .input}
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of cross-entropy loss in PyTorch combines `nn.LogSoftmax()`
# and `nn.NLLLoss()`
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # Convert labels to one-hot vectors.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # We will not calculate negative log-likelihood from the definition.
    # Rather, we will follow a circular argument. Because NLL is same as
    # `cross_entropy`, if we calculate cross_entropy that would give us NLL
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## [概要

* 情報理論は、情報の符号化、復号化、送信、操作に関する研究分野です。
* エントロピーは、さまざまな信号でどれだけの情報が表示されているかを測定する単位です。
* KL ダイバージェンスは、2 つの分布間のダイバージェンスを測定することもできます。
* クロスエントロピーはマルチクラス分類の目的関数と見なすことができます。クロスエントロピー損失を最小化することは、対数尤度関数の最大化と同等です。

## 演習

1. 最初のセクションのカードの例に、実際に要求されたエントロピーがあることを確認します。
1. KL ダイバージェンス $D(p\|q)$ がすべての分布 $p$ および $q$ で非負であることを示します。ヒント:ジェンセンの不等式を使う、つまり $-\log x$ が凸関数であるという事実を利用する。
1. いくつかのデータソースからエントロピーを計算してみましょう。
    * 猿が生成した出力をタイプライターで見ていると仮定します。猿はタイプライターの$44$キーのいずれかをランダムに押します（特殊キーやShiftキーはまだ発見されていないと想定できます）。1文字あたり何ビットの乱数が見られますか？
    * 猿に不満を感じて、酔っ払ったタイプセッターに置き換えました。首尾一貫していませんが、単語を生成することができます。代わりに、$2,000$ 語の語彙からランダムな単語を選び出します。単語の平均的な長さが英語で$4.5$文字であると仮定します。現在、1文字あたり何ビットのランダム性が観察されていますか？
    * それでも結果に不満を抱いているので、タイプセッターを高品質の言語モデルに置き換えます。言語モデルでは現在、単語あたり $15$ ポイントという低いパープレキシティを得ることができます。言語モデルの文字*perplexity*は、一連の確率の幾何平均の逆数として定義され、各確率は単語内の文字に対応します。具体的には、与えられた単語の長さが $l$ の場合、$\mathrm{PPL}(\text{word}) = \left[\prod_i p(\text{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\text{character}_i)} \right].$ テストワードが 4.5 文字であると仮定します。1文字あたり何ビットの乱数が見られますか？
1. $I(X, Y) = H(X) - H(X|Y)$の理由を直感的に説明してください。次に、双方をジョイント分布に対する期待として表現することで、これが真実であることを示す。
1. 2 つのガウス分布 $\mathcal{N}(\mu_1, \sigma_1^2)$ と $\mathcal{N}(\mu_2, \sigma_2^2)$ の間の KL ダイバージェンスは何になりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1105)
:end_tab:

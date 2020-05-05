# 確率
:label:`sec_prob`

形は様々ですが、機械学習が行っていることはすべて予測です。患者の臨床歴を考慮して、来年に心臓発作に苦しむであろう患者の*確率*を予測しようとするかもしれません。異常検出では、飛行機が正常に動作していたときのジェットエンジンからの一連の測定値から、今後とりうる値を評価することができます。強化学習では、環境下でエージェントが知的に行動することを望むでしょう。これは、利用可能な各行動の下で、高い報酬を得る確率について考えることを意味します。また、推薦システムを構築する際も確率について考慮する必要があります。たとえば、大規模なオンライン書店で*仮に*働いていたとします。ある顧客が特定の本を購入する確率を推定したいと考えるでしょう。このためには、確率と統計という言語を使用する必要があります。あらゆるコース、専攻、学位論文、キャリア、さらには学科まで、確率に深く関わっているのです。したがって当然ですが、この節の目標はこのトピック全体について説明することではありません。代わりに、読者に基礎を固めてもらうようにし、最初の深層学習モデルを構築するために必要な部分だけ説明し、あとは必要に応じて読者が自分自身で情報を探し求められるようにしたいと思います。


前の節では、確率を正確に説明したり、具体的な例を挙げたりすることはしませんでしたが、すでに確率については話をしていました。写真からイヌとネコを区別する問題について、より深く考えてみましょう。これは簡単に聞こえるかもしれませんが実際は手ごわいです。まず、イヌとネコを区別する問題の難易度は画像の解像度に依存する場合があります。


![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat_dog_pixels.png)
:width:`300px`
:label:`fig_cat_dog`

:numref:`fig_cat_dog` に示すように、人間は $160 \times 160$ピクセルの解像度で猫と犬を簡単に認識できますが、$40 \times 40$ピクセルでは難しく、$10 \times 10$ pixelsピクセルではほとんど不可能になります。言い換えると、猫と犬を遠距離から区別する (つまり解像度が低い状態で区別する) 能力は、十分な情報が与えられていない状況での推測に近いとも言えます。確率は、私たちの確信する度合いについて説明する、形式的な方法を提供します。画像に猫が写っていることを完全に確信している場合、対応するラベル$y$が$\mathrm{cat}$であるということ、つまり$P(l=\mathrm{cat})$が $1$ に等しいことを意味します。$y = \mathrm{cat}$または$l=\mathrm{dog}$を示唆する証拠がなかった場合、両者に判定される確率は等しく、$P(y=$ "cat"$) = P(y=$ "dog"$) = 0.5$ に*近い*表現になります。ある程度確信はあるが、画像に猫が写っているかどうかわからない場合は、 $0.5  < P(y=$ "cat"$) < 1$ の確率が割り当てられるでしょう。


次に、2つ目のケースを考えてみましょう。いくつかの気象観測データから、明日台北で雨が降る確率を予測したいと思います。夏の場合、$0.5$の確率で雨になるとします。

画像の分類と天気予報、どちらの場合も注目している値があります。そして、どちらの場合も結果については不確かであるという共通点があります。
しかし、2つのケースには重要な違いがあります。最初のケースでは、画像は実際には犬または猫のいずれかであり、ただそのどちらかが分からないだけです。 2番目のケースでは、私達が思っているように (そしてほとんどの物理学者も思っているように)、結果はランダムに決まる事象です。したがって、確率は、私たちの確信の度合いを説明するために用いられるフレキシブルな言語であり、幅広いコンテキストで効果的に利用されています。

## 確率理論の基礎

サイコロを投げて、別の数字ではなく$1$が表示される可能性を知りたいとします。サイコロが公平なものであれば、6種類の出目$\mathcal{X} = \{1, \ldots, 6\}$がすべて同様に発生する可能性があり、$6$つの場合の数のうち$1$つが$1$として観測されるでしょう。正式に記述すれば、$1$は確率$\frac{1}{6}$で発生するといえるでしょう。

工場から受け取った実際のサイコロについて、それらの比率がわからない可能性があれば、偏りがないかどうかを確認する必要があります。サイコロを調査する唯一の方法は、サイコロを何度も振って結果を記録することです。サイコロを振るたびに値\{1, \ldots, 6\}$を観察します。これらの結果を考慮して、各出目を観測する確率を調査したいと思います。

それぞれの確率を求める際の自然なアプローチとして、各出目の出現回数をカウントし、それをサイコロを投げた総回数で割ることです。これにより、特定の*イベント*の確率の*推定*を得ることができます。*大数の法則*では、サイコロを振る回数が増えると、潜在的な真の確率に推定の確率がますます近づくことが示されています。ここで何が起こっているか、詳細に入りこむ前に、試すことから始めてみましょう。

まず、必要なパッケージをインポートします。

```{.python .input  n=1}
%matplotlib inline
import d2l
from mxnet import np, npx
import random
npx.set_np()
```


次に、サイコロを振れるようにします。統計では、確率分布からデータ例を引いてくるこのプロセスを*サンプリング*と呼びます。確率をいくつかの離散的な選択肢に割り当てる分布は*多項分布*と呼ばれます。*分布*のより正式な定義については後ほど述べますが、抽象的な見方をすると、イベントに対する単なる確率の割り当てと考えてください。MXNetでは、まさに相応しい名前をもった`np.random.multinomial`という関数によって、多項分布からサンプリングすることができます。 この関数はさまざまな方法で呼び出すことができますが、ここでは最も単純なものに焦点を当てます。 単一のサンプルを引くためには、単純に確率のベクトルを渡します。

```{.python .input  n=2}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

上記のサンプラーを何度も実行すると、毎回ランダムな値を取得することがわかります。サイコロの公平性を推定する場合と同様に、同じ分布から多くのサンプルを生成することがよくあります。 Pythonの`for`ループでこれを行うと耐えられないほど遅いため、`random.multinomial`は複数のサンプルを一度に生成することをサポートし、任意のshapeをもった独立したサンプルの配列を返します。


```{.python .input  n=3}
np.random.multinomial(10, fair_probs)
```

We can also conduct, say $3$, groups of experiments, where each group draws $10$ samples, all at once.

```{.python .input  n=4}
counts = np.random.multinomial(10, fair_probs, size=3)
counts
```

サイコロの出目をサンプリングする方法がわかったので、1000個の出目をシミュレーションすることができます。その後、1000回サイコロを振った後に、各出目が出た回数を数えます。
具体的には、その相対的な頻度を、真の確率の推定値とします。


```{.python .input  n=5}
# Store the results as 32-bit floats for division
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000  # Reletive frequency as the estimate
```

Because we generated the data from a fair die, we know that each outcome has true probability $\frac{1}{6}$, roughly $0.167$, so the above output estimates look good.

We can also visualize how these probabilities converge over time towards the true probability.
Let's conduct $500$ groups of experiments where each group draws $10$ samples.

```{.python .input  n=6}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

各実線の曲線は、サイコロの6つの出目のうちの1つに対応しており、各グループの実験のあとに予想される、サイコロの出目が出る推定確率を表します。黒い破線は、潜在的な真の確率を示しています。より多くのデータを取得すると、$6$本の実線の曲線は真の解に向かって収束します。


### Axioms of Probability Theory

When dealing with the rolls of a die,
we call the set $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ the *sample space* or *outcome space*, where each element is an *outcome*.
An *event* is a set of outcomes from a given sample space.
For instance, "seeing a $5$" ($\{5\}$) and "seeing an odd number" ($\{1, 3, 5\}$) are both valid events of rolling a die.
Note that if the outcome of a random experiment is in event $\mathcal{A}$,
then event $\mathcal{A}$ has occurred.
That is to say, if $3$ dots faced up after rolling a die, since $3 \in \{1, 3, 5\}$,
we can say that the event "seeing an odd number" has occurred.

Formally, *probability* can be thought of a function that maps a set to a real value.
The probability of an event $\mathcal{A}$ in the given sample space $\mathcal{S}$,
denoted as $P(\mathcal{A})$, satisfies the following properties:

* For any event $\mathcal{A}$, its probability is never negative, i.e., $P(\mathcal{A}) \geq 0$;
* Probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any happens is equal to the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

These are also the axioms of probability theory, proposed by Kolmogorov in 1933.
Thanks to this axiom system, we can avoid any philosophical dispute on randomness;
instead, we can reason rigorously with a mathematical language.
For instance, by letting event $\mathcal{A}_1$ be the entire sample space and $\mathcal{A}_i = \emptyset$ for all $i > 1$, we can prove that $P(\emptyset) = 0$, i.e., the probability of an impossible event is $0$.


### Random Variables

サイコロを振るというランダムな実験において**確率変数**の概念を導入しました。確率変数は、ほぼすべての値を取る可能性があり決定的ではありません。確率変数はとりうる可能性の集合の中から、1回のランダムな実験により、1つの値をとることができます。Consider a random variable $X$ whose value is in the sample space $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ of rolling a die. We can denote the event "seeing a $5$" as $\{X = 5\}$ or $X = 5$, and its probability as $P(\{X = 5\})$ or $P(X = 5)$.
By $P(X = a)$, we make a distinction between the random variable $X$ and the values (e.g., $a$) that $X$ can take.
However, such pedantry results in a cumbersome notation.
For a compact notation,
on one hand, we can just denote $P(X)$ as the *distribution* over the random variable $X$:
the distribution tells us the probability that $X$ takes any value.
On the other hand,
we can simply write $P(a)$ to denote the probability that a random variable takes the value $a$.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$,
which means $\{X = 1, 2, \text{or}, 3\}$. Equivalently, $P(1 \leq X \leq 3)$ represents the probability that the random variable $X$ can take a value from $\{1, 2, 3\}$.


サイコロの面のような*離散*確率変数と、人の体重や身長のような*連続*確率変数との間には微妙な違いがあることに注意してください。 2人の人の身長がまったく同じかどうかを尋ねても意味がありません。十分に正確な測定を行うと、地球上の2人の人がまったく同じ身長にならないことがわかります。実際、十分に細かい測定を行った場合、目覚めたときと寝ているときの身長は同じになりません。そのため、ある人の身長が$1.80139278291028719210196740527486202$メートルである確率について尋ねる人はまずいないでしょう。世界人口を考えると確率は事実上0です。この場合、誰かの身長が$1.79$から$1.81$メートルの間など、指定された間隔に収まるかどうかを確認する方が理にかなっています。こういった場合、可能性を*密度*という見える値で定量化します。ちょうど$1.8$メートルの高さをとる確率はありませんが、密度はゼロではありません。任意の2つの異なる高さの間には、ゼロ以外の確率があります。残りの節では、離散空間における確率について考えます。連続変数に関する確率については、:numref:`sec_random_variables` を参照してください。

## 複数の確率変数の取り扱い

一度に複数の確率変数を扱いたくなることが多くあります。例えば、病気と症状の関係をモデル化したい場合を考えましょう。例えば、「インフルエンザ」と「せき」のような病気と症状が与えられていて、ある確率で患者に発生したり、しなかったりするとします。その双方の確率がゼロであることを望みますが、その確率と関係性を推定することで、その推論をより良い医療看護につなげることができるでしょう。

より複雑な例としては、数百万ピクセルの画像は、数百万の確率変数を含んでいると言えます。多くの場合、画像の中に写る物体を表すラベルを伴います。ラベルもまた確率変数と考えることができます。さらには、すべてのメタデータを確率変数と考えることもできるでしょう。例えば、場所、時間、レンズの口径、焦点距離、ISO、集束距離、カメラの種類などです。これらはすべて、同時に発生する確率変数です。複数の確率変数を扱う場合、いくつかの重要な概念があります。


### 結合確率

1つ目は*結合分布*$\P(A=a,B=b)$です。$a$と$b$の値が与えられたとき、結合分布は$A=a$と$B=b$が同時に発生する確率を示します。あらゆる$a, b$に対して、$\P(A=a, B=b) \leq \P(A=a)$が成立することに注意してください。これはつまり、$A$と$B$が発生するためには、$A$が発生して、*かつ*、$B$も発生する必要があるからです (逆も同様です)。したがって、$A, B$が個別に発生するよりも、$A$と$B$が同時に発生することはありません。

### 条件付き確率

上記によって、 $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$ という、確率の比に関する興味深い式を導くことができます。これを**条件付き確率**と呼び、 $P(B=b \mid A=a)$ で表します。$A=a$が発生する条件のもとで$B=b$が発生する確率を表します。

### ベイズの定理

条件付き確率の定義にもとづいて、ベイズ統計学で最も有用かる有名な方程式の1つを導くことができます。それは以下の通りです。定義から、積の法則より $P(A, B) = P(B \mid A) P(A)$ となります。対称性より $P(A, B) = P(A \mid B) P(B)$ も成立します。$P(b)>0$ を仮定します。これらの式から、条件の変数に関して解くと、次のようになります。

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

Note that here we use the more compact notation where $P(A, B)$ is a *joint distribution* and $P(A \mid B)$ is a *conditional distribution*. Such distributions can be evaluated for particular values $A = a, B=b$.

### 周辺化

This is very useful if we want to infer one thing from another, say cause and effect but we only know the properties in the reverse direction. One important operation that we need, to make this work, is *marginalization*, i.e., the operation of determining $\Pr(A)$ and $\Pr(B)$ from $\Pr(A,B)$. We can see that the probability of seeing $A$ amounts to accounting for all possible choices of $B$ and aggregating the joint probabilities over all of them, i.e.

ベイズの定理は、原因と結果など、あるものを別のものから推測したい場合に有用ですが、後で述べるように、逆方向の特性しかわからない場合があります。これを解決するために重要な操作として *marginalization (周辺化)*があります。つまり、$\P(A,B)$から$\P(B)$を決定する演算です。$B$を確認する確率は、$A$のすべての可能性を考慮し、それらすべてに関する結合確率を集約することで得られます。

$$P(B) = \sum_{A} P(A, B),$$

which is also known as the *sum rule*. The probability or distribution as a result of marginalization is called a *marginal probability* or a *marginal distribution*.

### 独立性

チェックすべきもう一つの特性として、*dependence(依存性)*と*independence(独立性)*があります。2つの確率変数 $A$ と $B$ が独立であるということは、あるイベント$A$が発生しても、イベント $B$ の発生に関する情報が明らかにならないことを意味します。この場合、$\P(B \mid A) = \P(B)$です。統計学者は通常、これを $A \perp B$ と表現します。ベイズの定理から、ただちに$\P(A \mid B) = \Pr(A)$であることがわかります。その他の場合は、すべて、$A$および$B$に依存する (dependence) ことになります。たとえば、サイコロを連続で2回振ったときの出目は独立しています。一方、照明スイッチの位置と部屋の明るさは独立していません (ただし、電球、停電、またはスイッチが破損する可能性が常にあるため、
独立しているかどうかは完全に決定論的とはいえません)。


Since $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ is equivalent to $P(A, B) = P(A)P(B)$, two random variables are independent if and only if their joint distribution is the product of their individual distributions.
Likewise, two random variables $A$ and $B$ are *conditionally independent* given another random variable $C$
if and only if $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. This is expressed as $A \perp B \mid C$.

### 応用
:label:`subsec_probability_hiv_app`

ここまで学んだことを試してみましょう。医師が患者にエイズ検査を実施するとします。このテストはかなり正確であり、患者が健康であるにもかかわらず、感染していると誤診する確率は、たったの$1\%$しかありません。また、患者が実際にHIVに感染していれば、HIVの検出に失敗することはありません。診断を$D_1$ ($1$は陽性、$0$は陰性)、HIVの感染の状態を$H$ ($1$は陽性、$0$は陰性)で表します。:numref:`conditional_prob_D1` に条件付き確率の一覧を示します。

:$P(D_1 \mid H)$ の条件付き確率

| 条件付き確率 | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`


条件付き確率は一般の確率と同様に足して$1$になる必要があるため、列方向の和はすべて$1$となる（ただし、行方向は1とならない）ことに注意してください。陽性の結果がでたときに、患者がAIDSに感染している確率、つまり $P(H = 1 \mid D_1 = 1)$ を計算しましょう。明らかに、この計算にはこの病気がどれくらい一般的かに依存するでしょう。なぜなら、それによって誤検知の値が変わるからです。ここでは、母集団が非常に健康的な場合を考え、$P(H=1) = 0.0015$ としましょう。ベイズの定理を適用すると

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

したがって、

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$


言い換えれば、検査の結果が99%正しいにも関わらず、患者が実際にAIDSに感染している確率は13.06%にすぎないことがわかります。このように統計は直感に反することがあります。


もし、患者が上記のようなおそろしいニュースを聞いたとしたらどうするでしょうか。おそらく、医者に対して、より正確な結果を得るための、追加のテストを実施するよう依頼するでしょう。:numref:`conditional_prob_D2` に示すように、2回目のテストはさきほどのテストとは異なる特徴があり、最初のものよりも精度が悪いとしましょう。

| 条件付き確率 | $H=1$ | $H=0$ |
|---|---|---|
|テスト 陽性|          0.98 |0.03 |
|テスト 陰性|          0.02 |          0.97
:label:`conditional_prob_D2`

残念ながら、2回目も陽性だったとしましょう。ベイズ理論を利用して必要な確率を計算しましょう。

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

ここで周辺化と積の法則を適用します。

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$


最後に、両方の検査の結果、陽性であると判断される確率は

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

2回目の検査によって、HIVの感染に関してより確信を得ることができました。2回目の検査は1回目の検査よりもかなり精度が低いにも関わらず、推定の結果を改善しました。



## Expectation and Variance

To summarize key characteristics of probability distributions,
we need some measures.
The *expectation* (or average) of the random variable $X$ is denoted as

$$E[X] = \sum_{x} x P(X = x).$$

When the input of a function $f(x)$ is a random variable drawn from the distribution $P$ with different values $x$,
the expectation of $f(x)$ is computed as

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$


In many cases we want to measure by how much the random variable $X$ deviates from its expectation. This can be quantified by the variance

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

Its square root is called the *standard deviation*.
The variance of a function of a random variable measures
by how much the function deviates from the expectation of the function,
as different values $x$ of the random variable are sampled from its distribution:

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$


## Summary

* We can use MXNet to sample from probability distributions.
* We can analyze multiple random variables using joint distribution, conditional distribution, Bayes' theorem, marginalization, and independence assumptions.  
* Expectation and variance offer useful measures to summarize key characteristics of probability distributions.



## 練習

1. 各グループが10サンプルを引くような実験を $m=500$ グループで行いました。$m$と$n$を変えて、実験結果を観測・分析してみてください。
1. 確率$P(\mathcal{A})$ と $P(\mathcal{B})$で発生する2つの事象について、$P(\mathcal{A} \cup \mathcal{B})$ と $P(\mathcal{A} \cap \mathcal{B})$の上限と下限を計算してください。ヒント: [ベン図](https://en.wikipedia.org/wiki/Venn_diagram)を利用して状況を表してみましょう。
1. $A, $B$, $C$ という事象の系列があり、$B$は$A$と$C$に依存し、$A$と$C$は$B$のみに依存する場合、その同時確率を単純化することは可能ですか? ヒント: これは[マルコフ連鎖](https://en.wikipedia.org/wiki/Markov_chain)です。
1. :numref:`subsec_probability_hiv_app`においては最初の検査のほうが精度が良かったです。最初の検査を2回目に実施したらどうなりますか?

## [議論](https://discuss.mxnet.io/t/2319)

![](../img/qr_probability.svg)

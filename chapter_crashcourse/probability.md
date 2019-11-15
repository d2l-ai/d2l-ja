# 確率と統計

形は様々ですが、機械学習が行っていることはすべて予測です。患者の臨床歴を考慮して、来年に心臓発作に苦しむであろう患者の*確率*を予測することができます。異常検出では、飛行機が正常に動作している場合に、ジェットエンジンからの一連の測定値がとりうる値を評価することができます。強化学習では、環境下でエージェントが知的に行動することを望むでしょう。これは、利用可能な各行動の下で、高い報酬を得る確率について考えることを意味します。また、推薦システムを構築する際も確率について考慮する必要があります。たとえば、大規模なオンライン書店で*仮に*働いていたとします。ある顧客が特定の本を購入する確率を推定したいと考えるでしょう。このためには、確率と統計という言語を使用する必要があります。あらゆるコース、専攻、学位論文、キャリア、さらには学科まで、確率に深く関わっているのです。したがって当然ですが、この節の目標はこのトピック全体について説明することではありません。代わりに、最初の機械学習モデルを構築するために必要な部分だけ説明し、あとは必要に応じて読者が自分自身で情報を探し求められるようにしたいと思います。


前の節では、確率を正確に説明したり、具体的な例を挙げたりすることはしませんでしたが、すでに確率については話をしていました。写真からイヌとネコを区別する問題について、より深く考えてみましょう。これは簡単に聞こえるかもしれませんが実際は手ごわいです。まず、イヌとネコを区別する問題の難易度は画像の解像度に依存する場合があります。



| 10px | 20px | 40px | 80px | 160px |
|:----:|:----:|:----:|:----:|:-----:|
|![](../img/whitecat10.jpg)|![](../img/whitecat20.jpg)|![](../img/whitecat40.jpg)|![](../img/whitecat80.jpg)|![](../img/whitecat160.jpg)|
|![](../img/whitedog10.jpg)|![](../img/whitedog20.jpg)|![](../img/whitedog40.jpg)|![](../img/whitedog80.jpg)|![](../img/whitedog160.jpg)|

人間は320ピクセルの解像度で猫と犬を簡単に認識できますが、40ピクセルでは難しく、10ピクセルではほとんど不可能になります。言い換えると、猫と犬を遠距離で区別する (つまり解像度が低い状態で区別する) 能力は、十分な情報が与えられていない状況での推測に近いとも言えます。確率は、私たちの確信する度合いについて説明する、形式的な方法を提供します。画像に猫が写っていることを完全に確信している場合、対応するラベル$l$が$\mathrm{cat}$であるという*確率* $P(l=\mathrm{cat})$が1.0に等しいことを意味します。$l = \mathrm{cat}$または$l=\mathrm{dog}$を示唆する証拠がなかった場合、両者に判定される確率は等しく二分され、$P(l = \mathrm{cat})= 0.5$になります。ある程度確信はあるが、画像に猫が写っているかどうかわからない場合は、$.5 <P(l =\mathrm{cat})<1.0$の確率が割り当てられるでしょう。


次に、2つ目のケースを考えてみましょう。いくつかの気象観測データから、明日台北で雨が降る確率を予測したいと思います。夏の場合、$.5$の確率で雨になるとします。画像の分類と天気予報、どちらの場合も注目している値があり、そして、どちらの場合も結果については不確かであるという共通点があります。
しかし、2つのケースには重要な違いがあります。最初のケースでは、画像は実際には犬または猫のいずれかであり (ランダムなわけではない)、ただそのどちらかが分からないだけです。 2番目のケースでは、私達が思っているように (そしてほとんどの物理学者も思っているように)、結果はランダムに決まる事象です。したがって、確率は、私たちの確信の度合いを説明するために用いられるフレキシブルな言語であり、幅広いコンテキストで効果的に利用されています。

## 確率理論の基礎

サイコロを投げて、別の数字ではなく$1$が表示される可能性を知りたいとします。サイコロが公平なものであれば、6種類の出目$\mathcal{X} = \{1, \ldots, 6\}$がすべて同様に発生する可能性があり、$6$つの場合の数のうち$1$つが$1$として観測されるでしょう。正式に記述すれば、$1$は確率$\frac{1}{6}$で発生するといえるでしょう。

工場から受け取った実際のサイコロについて、それらの比率がわからない可能性があれば、偏りがないかどうかを確認する必要があります。サイコロを調査する唯一の方法は、サイコロを何度も振って結果を記録することです。サイコロを振るたびに値\{1, 2, \ldots, 6\}$を観察します。これらの結果を考慮して、各出目を観測する確率を調査したいと思います。

それぞれの確率を求める際の自然なアプローチとして、各出目の出現回数をカウントし、それをサイコロを投げた総回数で割ることです。これにより、特定のイベントの確率の*推定*を得ることができます。大数の法則では、サイコロを振る回数が増えると、潜在的な真の確率に推定の確率がますます近づくことが示されています。ここで何が起こっているか、詳細に入りこむ前に、試すことから始めてみましょう。

まず、必要なパッケージをインポートします。

```{.python .input}
import mxnet as mx
from mxnet import nd
import numpy as np
from matplotlib import pyplot as plt
```


次に、サイコロを振れるようにします。統計では、確率分布からデータ例を引いてくるこのプロセスを*サンプリング*と呼びます。確率をいくつかの離散的な選択肢に割り当てる分布は*多項分布*と呼ばれます。*分布*のより正式な定義については後ほど述べますが、抽象的な見方をすると、イベントに対する単なる確率の割り当てと考えてください。MXNetでは、まさに相応しい名前をもった`nd.random.multinomial`という関数によって、多項分布からサンプリングすることができます。 この関数はさまざまな方法で呼び出すことができますが、ここでは最も単純なものに焦点を当てます。 単一のサンプルを引くためには、単純に確率のベクトルを渡します。

```{.python .input}
probabilities = nd.ones(6) / 6
nd.random.multinomial(probabilities)
```

上記のサンプラーを何度も実行すると、毎回ランダムな値を取得することがわかります。サイコロの公平性を推定する場合と同様に、同じ分布から多くのサンプルを生成することがよくあります。 Pythonの`for`ループでこれを行うと耐えられないほど遅いため、`random.multinomial`は複数のサンプルを一度に生成することをサポートし、任意のshapeをもった独立したサンプルの配列を返します。

```{.python .input}
print(nd.random.multinomial(probabilities, shape=(10)))
print(nd.random.multinomial(probabilities, shape=(5,10)))
```

サイコロの出目をサンプリングする方法がわかったので、1000個の出目をシミュレーションすることができます。その後、1000回サイコロを振った後に、各出目が出た回数を数えます。


```{.python .input}
rolls = nd.random.multinomial(probabilities, shape=(1000))
counts = nd.zeros((6,1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals
```

まず、$1000$回振った最終結果を見てみましょう。

```{.python .input}
totals / 1000
```

結果を見ると、いずれの出目についても最小の確率は$0.15$で、最大の確率は$0.188$となっています。公平なサイコロからデータを生成したので、各出目は$1/6$の確率、つまり$.167$で現れることがわかっていますので、これらの推定値は非常に良いです。確率がれらの推定値にどのように収束していくかを可視化することもできます。

まず、shapeが`(6, 1000)`の配列`counts`を見てみましょう。各時間ステップ (1000回中)、 `counts`はその出目が何回現れたかを表しています。したがって、 そのカウントを表すベクトルの$j$番目の列を、サイコロを振った回数で正規化すれば、ある時点における`現在の`推定確率を求めることができます。カウントを表すobjectは以下のようになります。

```{.python .input}
counts
```

振った回数で正規化すると、以下を得ることができます。

```{.python .input}
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
```

ご覧のとおり、最初にサイコロを振った際は、数字の1つが$1.0$の確率で現れ、他の数字が$0$の確率となるような極端な推論が得られます。$100$回振ると、もう少しまともな結果を見ることができます。グラフ化パッケージ `matplotlib` を使用して、この収束を視覚化することができます。 インストールしていない場合は、[インストール](https://matplotlib.org/)をお勧めします。


```{.python .input}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

plt.figure(figsize=(8, 6))
for i in range(6):
    plt.plot(estimates[i, :].asnumpy(), label=("P(die=" + str(i) +")"))

plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()
```

各実線の曲線は、サイコロの6つの出目のうちの1つに対応しており、1000回振ったあとに評価される出目の確率を表します。黒い破線は、潜在的な真の確率を示しています。より多くのデータを取得すると、実線の曲線は真の解に向かって収束します。

サイコロを振る例では**確率変数**の概念を導入しました。$X$として表される確率変数は、ほぼすべての値を取る可能性があり決定的ではありません。確率変数はとりうる可能性の集合の中から1つの値をとることができます。その集合を角括弧で示します（例：$\{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit} \}$）。集合に含まれる項目は*要素*と呼ばれ、$x\in S$と書くことで、要素$x$は集合Sに*含まれる*といえます。記号$\in$は"in"と読まれ、集合の要素であることを示します。たとえば、$\mathrm{dog} \in \{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit} \}$と確実に言うことができます。サイコロの出目を扱うとき、変数$X \in \{1, 2, 3, 4, 5, 6 \}$について関心があるといえるでしょう。

サイコロの面のような離散確率変数と、人の体重や身長のような連続確率変数との間には微妙な違いがあることに注意してください。 2人の人の身長がまったく同じかどうかを尋ねても意味がありません。十分に正確な測定を行うと、地球上の2人の人がまったく同じ身長にならないことがわかります。実際、十分に細かい測定を行った場合、目覚めたときと寝ているときの身長は同じになりません。そのため、ある人の身長が$ 2.00139278291028719210196740527486202 $メートルである確率について尋ねる人はまずいないでしょう。世界人口を考えると確率は事実上0です。この場合、誰かの身長が1.99から2.01メートルの間など、指定された間隔に収まるかどうかを確認する方が理にかなっています。こういった場合、可能性を*密度*という見える値で定量化します。ちょうど2.0メートルの高さをとる確率はありませんが、密度はゼロではありません。任意の2つの異なる高さの間には、ゼロ以外の確率があります。

覚えておくべき確率に関する重要な公理を以下に示します。

* 任意の事象 $z$ について, その確率は必ず非負となります。つまり $\Pr(Z=z) \geq 0$。
* 任意の2つの事象 $Z=z$ と $X=x$ について、その結合事象は各事象の和ほど、起こりうることはありません。つまり$\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)$。
* どの確率変数も、その値をとるすべての確率の和は必ず1です。つまり、$\sum_{i=1}^n \Pr(Z=z_i) = 1$。
* 相互に排他的な2つの事象$Z=z$ と $X=x$ について、どちらかが起こる確率は、それぞれの確率の和に等しい。つまり$\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=x)$。

## 複数の確率変数の取り扱い

一度に複数の確率変数を扱いたくなることが多くあります。例えば、病気と症状の関係をモデル化したい場合を考えましょう。例えば、「インフルエンザ」と「せき」のような病気と症状が与えられていて、ある確率で患者に発生したり、しなかったりするとします。その双方の確率がゼロであることを望みますが、その確率と関係性を推定することで、その推論をより良い医療看護につなげることができるでしょう。

より複雑な例としては、数百万ピクセルの画像は、数百万の確率変数を含んでいると言えます。多くの場合、画像の中に写る物体を表すラベルを伴います。ラベルもまた確率変数と考えることができます。さらには、すべてのメタデータを確率変数と考えることもできるでしょう。例えば、場所、時間、レンズの口径、焦点距離、ISO、集束距離、カメラの種類などです。これらはすべて、同時に発生する確率変数です。複数の確率変数を扱う場合、いくつかの重要な概念があります。 1つ目は結合分布$\Pr(A,B)$です。結合分布は、$a$と$b$の要素が与えられたとき、$A=a$と$B=b$が同時に発生する確率を示します。あらゆる$a, b$に対して、$\Pr(A=a, B =b) \leq \Pr(A=a)$が成立することに注意してください。

This has to be the case, since for $A$ and $B$ to happen, $A$ has to happen *and* $B$ also has to happen (and vice versa). Thus $A,B$ cannot be more likely than $A$ or $B$ individually. This brings us to an interesting ratio: $0 \leq \frac{\Pr(A,B)}{\Pr(A)} \leq 1$. We call this a **conditional probability**
and denote it by $\Pr(B | A)$, the probability that $B$ happens, provided that
$A$ has happened.

Using the definition of conditional probabilities, we can derive one of the most useful and celebrated equations in statistics—Bayes' theorem.
It goes as follows: By construction, we have that $\Pr(A, B) = \Pr(B | A) \Pr(A)$. By symmetry, this also holds for $\Pr(A,B) = \Pr(A | B) \Pr(B)$. Solving for one of the conditional variables we get:

$$\Pr(A | B) = \frac{\Pr(B | A) \Pr(A)}{\Pr(B)}$$

This is very useful if we want to infer one thing from another, say cause and effect but we only know the properties in the reverse direction. One important operation that we need, to make this work, is **marginalization**, i.e., the operation of determining $\Pr(A)$ and $\Pr(B)$ from $\Pr(A,B)$. We can see that the probability of seeing $A$ amounts to accounting for all possible choices of $B$ and aggregating the joint probabilities over all of them, i.e.

$$\Pr(A) = \sum_{B'} \Pr(A,B') \text{ and
} \Pr(B) = \sum_{A'} \Pr(A',B)$$

Another useful property to check for is **dependence** vs. **independence**.
Independence is when the occurrence of one event does not reveal any information about the occurrence of the other. In this case $\Pr(B | A) = \Pr(B)$. Statisticians typically exress this as $A \perp\!\!\!\perp B$. From Bayes' Theorem, it follows immediately that also $\Pr(A | B) = \Pr(A)$. In all other cases we call $A$ and $B$ dependent. For instance, two successive rolls of a die are independent. On the other hand, the position of a light switch and the brightness in the room are not (they are not perfectly deterministic, though, since we could always have a broken lightbulb, power failure, or a broken switch).

Let's put our skills to the test. Assume that a doctor administers an AIDS test to a patient. This test is fairly accurate and it fails only with 1% probability if the patient is healthy by reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it. We use $D$ to indicate the diagnosis and $H$ to denote the HIV status. Written as a table the outcome $\Pr(D | H)$ looks as follows:

|
outcome| HIV positive | HIV negative |
|:------------|-------------:|-------------:|
|Test positive|            1 |
0.01 |
|Test negative|            0 |         0.99 |

Note that the column sums are all one (but the row sums aren't), since the conditional probability needs to sum up to $1$, just like the probability. Let us work out the probability of the patient having AIDS if the test comes back positive. Obviously this is going to depend on how common the disease is, since it affects the number of false alarms. Assume that the population is quite healthy, e.g. $\Pr(\text{HIV positive}) = 0.0015$. To apply Bayes' Theorem, we need to determine
$$\begin{aligned}
\Pr(\text{Test positive}) =& \Pr(D=1 | H=0) \Pr(H=0) + \Pr(D=1
| H=1) \Pr(H=1) \\
=& 0.01 \cdot 0.9985 + 1 \cdot 0.0015 \\
=& 0.011485
\end{aligned}
$$

Thus, we get

$$\begin{aligned} \Pr(H = 1 | D = 1) =& \frac{\Pr(D=1 | H=1) \Pr(H=1)}{\Pr(D=1)} \\ =& \frac{1 \cdot 0.0015}{0.011485} \\ =& 0.131 \end{aligned} $$

In other words, there's only a 13.1% chance that the patient actually has AIDS, despite using a test that is 99% accurate. As we can see, statistics can be quite counterintuitive.

## Conditional independence
What should a patient do upon receiving such terrifying news? Likely, he/she
would ask the physician to administer another test to get clarity. The second
test has different characteristics (it isn't as good as the first one).

|
outcome |  HIV positive |  HIV negative |
|:------------|--------------:|--------------:|
|Test positive|          0.98 |
0.03 |
|Test negative|          0.02 |          0.97 |

Unfortunately, the second test comes back positive, too. Let us work out the requisite probabilities to invoke Bayes' Theorem.

* $\Pr(D_1 = 1 \text{ and } D_2 = 1 | H = 0) = 0.01 \cdot 0.03 = 0.0003$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1 | H = 1) = 1 \cdot 0.98 = 0.98$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1) = 0.0003 \cdot 0.9985 + 0.98 \cdot 0.0015 = 0.00176955$
* $\Pr(H = 1 | D_1 = 1 \text{ and } D_2 = 1) = \frac{0.98 \cdot 0.0015}{0.00176955} = 0.831$

That is, the second test allowed us to gain much higher confidence that not all is well. Despite the second test being considerably less accurate than the first one, it still improved our estimate quite a bit. You might ask, *why couldn't we just run the first test a second time?* After all, the first test was more accurate. The reason is that we needed a second test whose result is *independent* of the first test (given the true diagnosis). In other words, we made the tacit assumption that $\Pr(D_1, D_2 | H) = \Pr(D_1 | H) \Pr(D_2 | H)$. Statisticians call such random variables **conditionally independent**. This is expressed as $D_1 \perp\!\!\!\perp D_2  | H$.

## Sampling

Often, when working with probabilistic models, we'll want not just to estimate distributions from data, but also to generate data by sampling from distributions. One of the simplest ways to sample random numbers is to invoke the `random` method from Python's `random` package.

```{.python .input}
import random
for i in range(10):
    print(random.random())
```

### Uniform Distribution

These numbers likely *appear* random. Note that their range is between 0 and 1 and they are evenly distributed. Because these numbers are generated by default from the uniform distribution, there should be no two sub-intervals of $[0,1]$ of equal size where numbers are more likely to lie in one interval than the other. In other words, the chances of any of these numbers to fall into the interval $[0.2,0.3)$ are the same as in the interval $[.593264, .693264)$. In fact, these numbers are pseudo-random, and the computer generates them by first producing a random integer and then dividing it by its maximum range. To sample random integers directly, we can run the following snippet, which generates integers in the range between 1 and 100.

```{.python .input}
for i in range(10):
    print(random.randint(1, 100))
```

How might we check that ``randint`` is really uniform? Intuitively, the best
strategy would be to run sampler many times, say 1 million, and then count the
number of times it generates each value to ensure that the results are approximately uniform.

```{.python .input}
import math
counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# Mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid
for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
plt.show()
```

We can see from these figures that the initial number of counts looks *strikingly* uneven. If we sample fewer than 100 draws from a distribution over
100 outcomes this should be expected. But even for 1000 samples there is a
significant variability between the draws. What we are really aiming for is a
situation where the probability of drawing a number $x$ is given by $p(x)$.

### The categorical distribution

Drawing from a uniform distribution over a set of 100 outcomes is simple. But what if we have nonuniform probabilities? Let's start with a simple case, a biased coin which comes up heads with probability 0.35 and tails with probability 0.65. A simple way to sample from that is to generate a uniform random variable over $[0,1]$ and if the number is less than $0.35$, we output heads and otherwise we generate tails. Let's try this out.

```{.python .input}
# Number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# Count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

plt.figure(figsize=(15, 8))
plt.semilogx(x, p0)
plt.semilogx(x, p1)
plt.show()
```

As we can see, on average, this sampler will generate 35% zeros and 65% ones.
Now what if we have more than two possible outcomes? We can simply generalize
this idea as follows. Given any probability distribution, e.g. $p = [0.1, 0.2, 0.05, 0.3, 0.25, 0.1]$ we can compute its cumulative distribution (python's ``cumsum`` will do this for you) $F = [0.1, 0.3, 0.35, 0.65, 0.9, 1]$. Once we have this we draw a random variable $x$ from the uniform distribution $U[0,1]$ and then find the interval where $F[i-1] \leq x < F[i]$. We then return $i$ as the sample. By construction, the chances of hitting interval $[F[i-1], F[i])$ has probability $p(i)$.

Note that there are many more efficient algorithms for sampling than the one above. For instance, binary search over $F$ will run in $O(\log n)$ time for $n$ random variables. There are even more clever algorithms, such as the [Alias
Method](https://en.wikipedia.org/wiki/Alias_method) to sample in constant time,
after $O(n)$ preprocessing.

### The Normal distribution

The standard Normal distribution (aka the standard Gaussian distribution) is given by $p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2} x^2\right)$. Let's plot it to get a feel for it.

```{.python .input}
x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
plt.figure(figsize=(10, 5))
plt.plot(x, p)
plt.show()
```

Sampling from this distribution is less trivial. First off, the support is
infinite, that is, for any $x$ the density $p(x)$ is positive. Secondly, the
density is nonuniform. There are many tricks for sampling from it - the key idea in all algorithms is to stratify $p(x)$ in such a way as to map it to the
uniform distribution $U[0,1]$. One way to do this is with the probability
integral transform.

Denote by $F(x) = \int_{-\infty}^x p(z) dz$ the cumulative distribution function (CDF) of $p$. This is in a way the continuous version of the cumulative sum that we used previously. In the same way we can now define the inverse map $F^{-1}(\xi)$, where $\xi$ is drawn uniformly. Unlike previously where we needed to find the correct interval for the vector $F$ (i.e. for the piecewise constant function), we now invert the function $F(x)$.

In practice, this is slightly more tricky since inverting the CDF is hard in the case of a Gaussian. It turns out that the *twodimensional* integral is much easier to deal with, thus yielding two normal random variables than one, albeit at the price of two uniformly distributed ones. For now, suffice it to say that there are built-in algorithms to address this.

The normal distribution has yet another desirable property. In a way all distributions converge to it, if we only average over a sufficiently large number of draws from any other distribution. To understand this in a bit more detail, we need to introduce three important things: expected values, means and variances.

* The expected value $\mathbf{E}_{x \sim p(x)}[f(x)]$ of a function $f$ under a distribution $p$ is given by the integral $\int_x p(x) f(x) dx$. That is, we average over all possible outcomes, as given by $p$.
* A particularly important expected value is
that for the function $f(x) = x$, i.e. $\mu := \mathbf{E}_{x \sim p(x)}[x]$. It
provides us with some idea about the typical values of $x$.
* Another important quantity is the variance, i.e. the typical deviation from the mean $\sigma^2 := \mathbf{E}_{x \sim p(x)}[(x-\mu)^2]$. Simple math shows (check it as an exercise) that $\sigma^2 = \mathbf{E}_{x \sim p(x)}[x^2] - \mathbf{E}^2_{x \sim p(x)}[x]$.

The above allows us to change both mean and variance of random variables. Quite obviously for some random variable $x$ with mean $\mu$, the random variable $x + c$ has mean $\mu + c$. Moreover, $\gamma x$ has the variance $\gamma^2 \sigma^2$. Applying this to the normal distribution we see that one with mean $\mu$ and variance $\sigma^2$ has the form $p(x) = \frac{1}{\sqrt{2 \sigma^2 \pi}} \exp\left(-\frac{1}{2 \sigma^2} (x-\mu)^2\right)$. Note the scaling factor $\frac{1}{\sigma}$—it arises from the fact that if we stretch the distribution by $\sigma$, we need to lower it by $\frac{1}{\sigma}$ to retain the same probability mass (i.e. the weight under the distribution always needs to integrate out to 1).

Now we are ready to state one of the most fundamental theorems in statistics, the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). It states that for sufficiently well-behaved random variables, in particular random variables with well-defined mean and variance, the sum tends toward a normal distribution. To get some idea, let's repeat the experiment described in the beginning, but now using random variables with integer values of $\{0, 1, 2\}$.

```{.python .input}
# Generate 10 random sequences of 10,000 uniformly distributed random variables
tmp = np.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))

# Cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()
```

This looks very similar to the initial example, at least in the limit of averages of large numbers of variables. This is confirmed by theory. Denote by
mean and variance of a random variable the quantities

$$\mu[p] := \mathbf{E}_{x \sim p(x)}[x] \text{ and } \sigma^2[p] := \mathbf{E}_{x \sim p(x)}[(x - \mu[p])^2]$$

Then we have that $\lim_{n\to \infty} \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{x_i - \mu}{\sigma} \to \mathcal{N}(0, 1)$. In other words, regardless of what we started out with, we will always converge to a Gaussian. This is one of the reasons why Gaussians are so popular in statistics.


### More distributions

Many more useful distributions exist. If you're interested in going deeper, we recommend consulting a dedicated book on statistics or looking up some common distributions on Wikipedia for further detail. Some important distirbutions to be aware of include:

* **Binomial Distribution** It is used to describe the distribution over multiple draws from the same distribution, e.g. the number of heads when tossing a biased coin (i.e. a coin with probability $\pi \in [0, 1]$ of returning heads) 10 times. The binomial probability is given by $p(x) = {n \choose x} \pi^x (1-\pi)^{n-x}$.
* **Multinomial Distribution** Often, we are concerned with more than two
outcomes, e.g. when rolling a dice multiple times. In this case, the
distribution is given by $p(x) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k \pi_i^{x_i}$.
* **Poisson Distribution** This distribution models the occurrence of point events that happen with a given rate, e.g. the number of raindrops arriving within a given amount of time in an area (weird fact - the number of Prussian soldiers being killed by horses kicking them followed that distribution). Given a rate $\lambda$, the number of occurrences is given by $p(x) = \frac{1}{x!} \lambda^x e^{-\lambda}$.
* **Beta, Dirichlet, Gamma, and Wishart Distributions** They are what statisticians call *conjugate* to the Binomial, Multinomial, Poisson and Gaussian respectively. Without going into detail, these distributions are often used as priors for coefficients of the latter set of distributions, e.g. a Beta distribution as a prior for modeling the probability for binomial outcomes.



## Summary

So far, we covered probabilities, independence, conditional independence, and how to use this to draw some basic conclusions. We also introduced some fundamental probability distributions and demonstrated how to sample from them using Apache MXNet. This is already a powerful bit of knowledge, and by itself a sufficient set of tools for developing some classic machine learning models. In the next section, we will see how to operationalize this knowlege to build your first machine learning model: the Naive Bayes classifier.

## Exercises

1. Given two events with probability $\Pr(A)$ and $\Pr(B)$, compute upper and lower bounds on $\Pr(A \cup B)$ and $\Pr(A \cap B)$. Hint - display the situation using a [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Assume that we have a sequence of events, say $A$, $B$ and $C$, where $B$ only depends on $A$ and $C$ only on $B$, can you simplify the joint probability? Hint - this is a [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).

## Scan the QR Code to
[Discuss](https://discuss.mxnet.io/t/2319)

![](../img/qr_probability.svg)

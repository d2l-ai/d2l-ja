# ランダム変数
:label:`sec_random_variables`

:numref:`sec_prob` では、離散確率変数を扱う方法の基本を見ました。この場合は、可能な値の有限集合または整数のいずれかをとる確率変数を指します。このセクションでは、任意の実数値をとることができる確率変数である*連続確率変数*の理論を開発します。 

## 連続確率変数

連続確率変数は、離散確率変数よりもはるかに微妙なトピックです。技術的な飛躍は、数値リストの追加と関数の積分との間のジャンプに匹敵するということです。そのため、理論を発展させるには少し時間をかける必要があります。 

### 離散から連続へ

To understand the additional technical challenges encountered when working with continuous random variables, let us perform a thought experiment.  Suppose that we are throwing a dart at the dart board, and we want to know the probability that it hits exactly $2 \text{cm}$ from the center of the board.

To start with, we imagine measuring a single digit of accuracy, that is to say with bins for $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$, and so on.  We throw say $100$ darts at the dart board, and if $20$ of them fall into the bin for $2\text{cm}$ we conclude that $20\%$ of the darts we throw hit the board $2 \text{cm}$ away from the center.

However, when we look closer, this does not match our question!  We wanted exact equality, whereas these bins hold all that fell between say $1.5\text{cm}$ and $2.5\text{cm}$.

Undeterred, we continue further.  We measure even more precisely, say $1.9\text{cm}$, $2.0\text{cm}$, $2.1\text{cm}$, and now see that perhaps $3$ of the $100$ darts hit the board in the $2.0\text{cm}$ bucket.  Thus we conclude the probability is $3\%$.

However, this does not solve anything!  We have just pushed the issue down one digit further.  Let us abstract a bit. Imagine we know the probability that the first $k$ digits match with $2.00000\ldots$ and we want to know the probability it matches for the first $k+1$ digits. It is fairly reasonable to assume that the ${k+1}^{\mathrm{th}}$ digit is essentially a random choice from the set $\{0, 1, 2, \ldots, 9\}$.  At least, we cannot conceive of a physically meaningful process which would force the number of micrometers away form the center to prefer to end in a $7$ vs a $3$.

What this means is that in essence each additional digit of accuracy we require should decrease probability of matching by a factor of $10$.  Or put another way, we would expect that

$$
P(\text{distance is}\; 2.00\ldots, \;\text{to}\; k \;\text{digits} ) \approx p\cdot10^{-k}.
$$

The value $p$ essentially encodes what happens with the first few digits, and the $10^{-k}$ handles the rest.

Notice that if we know the position accurate to $k=4$ digits after the decimal. that means we know the value falls within the interval say $[(1.99995,2.00005]$ which is an interval of length $2.00005-1.99995 = 10^{-4}$.  Thus, if we call the length of this interval $\epsilon$, we can say

$$
P(\text{distance is in an}\; \epsilon\text{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

Let us take this one final step further.  We have been thinking about the point $2$ the entire time, but never thinking about other points.  Nothing is different there fundamentally, but it is the case that the value $p$ will likely be different.  We would at least hope that a dart thrower was more likely to hit a point near the center, like $2\text{cm}$ rather than $20\text{cm}$.  Thus, the value $p$ is not fixed, but rather should depend on the point $x$.  This tells us that we should expect

$$P(\text{distance is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Indeed, :eqref:`eq_pdf_deriv` precisely defines the *probability density function*.  It is a function $p(x)$ which encodes the relative probability of hitting near one point vs. another.  Let us visualize what such a function might look like.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

関数値が大きい位置は、乱数値が検出される可能性が高い領域を示します。低い部分は、ランダム値を見つける可能性が低い領域です。 

### 確率密度関数

これについてさらに調べてみましょう。確率変数$X$の確率密度関数が直感的に何であるかをすでに見てきました。つまり、密度関数は関数$p(x)$なので、 

$$P(X \; \text{is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

しかし、これは$p(x)$の特性にとって何を意味するのでしょうか？ 

まず、確率が負になることはないので、$p(x) \ge 0$ も期待すべきです。 

次に、$\mathbb{R}$ を $\epsilon$ 幅の無限数のスライス、たとえばスライス $(\epsilon\cdot i, \epsilon \cdot (i+1)]$ にスライスするとします。これらのそれぞれについて、:eqref:`eq_pdf_def` から、確率はおよそ 

$$
P(X \; \text{is in an}\; \epsilon\text{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

全部まとめるとそうなるはず 

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

これは :numref:`sec_integral_calculus` で説明した積分の近似に過ぎないので、次のように言えます。 

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

$P(X\in\mathbb{R}) = 1$、確率変数は*いくらか*の数をとらなければならないので、どんな密度でも結論を出すことができます。 

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

実際、これをさらに掘り下げてみると、$a$、$b$ について、 

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

これをコードで近似するには、以前と同じ離散近似法を使用します。この場合、青色の領域に落ちる確率を近似することができます。

```{.python .input}
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

これらの2つの特性は、可能な確率密度関数（または一般的に見られる略語には*p.d.f.*）の空間を正確に表していることが分かります。それらは非負の関数 $p(x) \ge 0$ であり、 

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

積分を使用してこの関数を解釈し、確率変数が特定の区間にある確率を求めます。 

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions`ではいくつかの一般的なディストリビューションが見られますが、要約の作業を続けましょう。 

### 累積分布関数

前のセクションでは、p.dfの概念を見ました。実際には、これは連続確率変数を議論するためによく見られる方法ですが、重要な落とし穴が1つあります。それは、p.dfの値自体が確率ではなく、積分して生成しなければならない関数であるということです。確率。密度の長さが $1/10$ よりも長く $10$ より大きくない限り、密度が $10$ より大きくても問題はありません。これは直観に反する可能性があるため、人々はしばしば*累積分布関数*、つまり確率*であるc.d.f. という観点から考えることもあります。 

特に、:eqref:`eq_pdf_int_int` を使用して、密度が $p(x)$ の確率変数 $X$ の c.d.f. を次のように定義します。 

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

いくつかの特性を観察してみましょう。 

* $F(x) \rightarrow 0$ は $x\rightarrow -\infty$ と表記されています。
* $F(x) \rightarrow 1$ は $x\rightarrow \infty$ と表示されます。
* $F(x)$ は減少しません ($y > x \implies F(y) \ge F(x)$)。
* $X$ が連続確率変数の場合、$F(x)$ は連続 (ジャンプなし) です。

4 番目の箇条書きでは、$X$ が離散的である場合、つまり $0$ と $1$ の値を両方とも確率 $1/2$ でとると、これは当てはまりません。その場合 

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

この例では、c.d.f. を使用する利点の 1 つ、同じフレームワーク内で連続または離散の確率変数を処理する機能、または実際に 2 つの混合 (コインを投げる:ヘッドがダイスのロールを返す場合、尻尾がダーツの中心から投げられる距離を返す場合) を処理できることが分かります。ボード)。 

### 手段

確率変数 $X$ を扱っているとします。分布そのものは解釈しにくい場合があります。確率変数の振る舞いを簡潔に要約できると便利な場合がよくあります。確率変数の振る舞いを捉えるのに役立つ数値を*要約統計量*と呼びます。最も一般的なものは、*平均*、*分散*、*標準偏差* です。 

*mean* は確率変数の平均値をエンコードします。確率$p_i$の値 $x_i$ を取る離散確率変数 $X$ がある場合、平均は加重平均で与えられます。その値に確率変数がかかる確率を掛けた値を合計します。 

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

平均を解釈する必要があるのは（注意が必要ですが）、確率変数がどこに位置するのかを本質的に教えてくれるということです。 

このセクション全体で検討する最小限の例として、$X$ を確率が $p$ の $a-2$、確率が $p$ の $a+2$、確率が $1-2p$ の $a$ を取る確率変数とします。:eqref:`eq_exp_def` を使用して計算できます。$a$ と $p$ の選択肢があれば、平均は次のようになります。 

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

したがって、平均値は $a$ であることがわかります。$a$ は確率変数を中心とした位置なので、これは直感と一致します。 

これらは役に立つので、いくつかのプロパティをまとめましょう。 

* 任意の確率変数 $X$ と数値 $a$ と $b$ については、その $\mu_{aX+b} = a\mu_X + b$ があります。
* 2 つの確率変数 $X$ と $Y$ がある場合、$\mu_{X+Y} = \mu_X+\mu_Y$ になります。

平均は確率変数の平均動作を理解するのに役立ちますが、平均値では完全に直感的に理解することすらできません。1回の販売で$\$10\ pm\ $1$の利益を上げることは、平均値が同じであるにもかかわらず、1回の販売で$\$10\ pm\ $15$を稼ぐこととは大きく異なります。2つ目は変動の度合いがはるかに大きいため、リスクがはるかに大きくなります。したがって、確率変数の振る舞いを理解するには、少なくとももう1つ測度が必要です。確率変数がどれだけ広く変動するかを測る尺度です。 

### 差異

これにより、確率変数の*分散*が考慮されます。これは、確率変数が平均からどれだけ逸脱しているかを示す量的尺度です。$X - \mu_X$ という式を考えてみましょう。これは、確率変数の平均値からの偏差です。この値は正でも負でもかまいません。そのため、偏差の大きさを測定するために、何かを正にする必要があります。 

$\left|X-\mu_X\right|$を見てみるのが妥当で、これは*平均絶対偏差*と呼ばれる有用な量につながりますが、数学や統計学の他の分野とのつながりから、人々はしばしば異なる解を使います。 

特に、$(X-\mu_X)^2.$を見ています。この量の典型的な大きさを平均を取って見ると、分散に到達します。 

$$\sigma_X^2 = \mathrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

:eqref:`eq_var_def` の最後の等価性は、途中で定義を拡張し、期待の性質を適用することによって成り立ちます。 

$X$ は確率が $p$ で $a-2$、確率が $p$ で $a+2$、確率が $1-2p$ の $a$ を取る確率変数である例を見てみましょう。この場合 $\mu_X = a$ なので、計算する必要があるのは $E\left[X^2\right]$ だけです。これは簡単に行えます。 

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

したがって、:eqref:`eq_var_def`までに分散は次のようになります。 

$$
\sigma_X^2 = \mathrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

この結果もまた意味があります。最大の$p$は$1/2$で、これは$a-2$または$a+2$をコインフリップで選ぶことに相当します。この分散が $4$ であることは、$a-2$ と $a+2$ の両方が平均値から $2$ 単位離れていることと $2^2 = 4$ であるという事実に対応しています。スペクトルの反対側では、$p=0$ の場合、この確率変数は常に値 $0$ を取るため、分散はまったくありません。 

以下に、分散のプロパティをいくつか挙げます。 

* $X$ が定数である場合に限り、$\mathrm{Var}(X) = 0$ となる確率変数 $X$、$\mathrm{Var}(X) \ge 0$。
* 任意の確率変数 $X$ と数値 $a$ と $b$ については、その $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$ があります。
* 2つの「独立した」確率変数 $X$ と $Y$ がある場合、$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ になります。

これらの値を解釈すると、多少の混乱が生じることがあります。特に、この計算で単位を追跡するとどうなるか想像してみましょう。Web ページ上の製品に割り当てられた星評価を使用して作業しているとします。$a$、$a-2$、$a+2$ は、すべて星の単位で測定されます。同様に、平均$\mu_X$も星 (加重平均) で測定されます。しかし、分散に達すると、すぐに問題が発生します。つまり、*二乗星*の単位である $(X-\mu_X)^2$ を調べます。これは、分散そのものが元の測定値と比較できないことを意味します。解釈可能にするには、元の単位に戻す必要があります。 

### 標準偏差

この要約統計量は、平方根を取ることによって分散から常に推定できます。したがって、*標準偏差*は次のように定義されます。 

$$
\sigma_X = \sqrt{\mathrm{Var}(X)}.
$$

この例では、標準偏差が $\sigma_X = 2\sqrt{2p}$ になったことを意味します。レビュー例で星の単位を扱っている場合、$\sigma_X$ は再び星の単位で表されます。 

分散の特性は、標準偏差で言い換えることができます。 

* 任意の確率変数 $X$、$\sigma_{X} \ge 0$ です。
* 任意の確率変数 $X$ と数値 $a$ と $b$ については、$\sigma_{aX+b} = |a|\sigma_{X}$ があります。
* 2つの「独立した」確率変数 $X$ と $Y$ がある場合、$\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$ になります。

このとき、「標準偏差が元の確率変数の単位である場合、その確率変数に関して描画できるものを表しているのか」と尋ねるのは自然なことです。答えははっきりとイエスです！実際、平均が確率変数の典型的な位置を教えてくれたように、標準偏差はその確率変数の典型的な変動範囲を与えます。チェビシェフの不等式として知られるものを用いて、これを厳格にすることができます。 

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

あるいは、$\alpha=10$の場合は99ドル\ %$ of the samples from any random variable fall within $10ドルの平均の標準偏差を口頭で述べるといいでしょう。これにより、標準の要約統計量を即座に解釈できます。 

このステートメントがいかに微妙であるかを確認するために、実行例をもう一度見てみましょう。$X$ は確率が $p$ で $a-2$、確率が $p$ で $a+2$、確率が $1-2p$ である $a$ を取る確率変数です。平均値は $a$ で、標準偏差は $2\sqrt{2p}$ であることがわかりました。つまり、チェビシェフの不等式 :eqref:`eq_chebyshev` を $\alpha = 2$ とすると、式は次のようになります。 

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

This means that $75\%$ of the time, this random variable will fall within this interval for any value of $p$.  Now, notice that as $p \rightarrow 0$, this interval also converges to the single point $a$.  But we know that our random variable takes the values $a-2, a$, and $a+2$ only so eventually we can be certain $a-2$ and $a+2$ will fall outside the interval!  The question is, at what $p$ does that happen.  So we want to solve: for what $p$ does $a+4\sqrt{2p} = a+2$, which is solved when $p=1/8$, which is *exactly* the first $p$ where it could possibly happen without violating our claim that no more than $1/4$ of samples from the distribution would fall outside the interval ($1/8$ to the left, and $1/8$ to the right).

Let us visualize this.  We will show the probability of getting the three values as three vertical bars with height proportional to the probability.  The interval will be drawn as a horizontal line in the middle.  The first plot shows what happens for $p > 1/8$ where the interval safely contains all points.

```{.python .input}
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

2 つ目は、$p = 1/8$ では、間隔が 2 つのポイントに正確に接していることを示しています。これは、不等式が*sharp* であることを示しています。これは、不等式を真に保ちながらこれより小さい区間を取ることができないためです。

```{.python .input}
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

3 つ目は、$p < 1/8$ では区間に中心のみが含まれていることを示しています。確率の $1/4$ 以下が区間外になるようにする必要があるだけなので、これによって不等式は無効になりません。つまり、$p < 1/8$ になると $a-2$ と $a+2$ の 2 つの点を破棄できます。

```{.python .input}
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### 連続体の平均と分散

これはすべて離散確率変数に関するものですが、連続確率変数の場合も同様です。これがどのように機能するかを直感的に理解するために、実数線を $(\epsilon i, \epsilon (i+1)]$ で与えられる長さ $\epsilon$ の間隔に分割するとします。これを行うと、連続確率変数は離散化され、:eqref:`eq_exp_def`を使うことができます。 

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

$p_X$ は$X$の密度です。これは $xp_X(x)$ の積分に対する近似なので、次のように結論付けることができます。 

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

同様に、:eqref:`eq_var_def` を使用すると、分散は次のように記述できます。 

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

この場合でも、平均、分散、および標準偏差について前述したすべてが適用されます。例えば、密度をもつ確率変数を考えると 

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \text{otherwise}.
\end{cases}
$$

計算できる 

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

そして 

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

警告として、*Cauchy 分布* として知られる、もう 1 つの例を調べてみましょう。これは p.d.f. が次で与えられる分布です 

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

この関数は無意味に見え、実際に積分の表を調べると、その下に面積1があることが示され、連続確率変数を定義します。 

何がうまくいかないかを見るために、これの分散を計算してみましょう。これには :eqref:`eq_var_def` コンピューティングの使用が含まれます 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

内側の関数は次のようになります。

```{.python .input}
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

この関数は本質的にゼロに近い小さなディップを持つ定数であるため、この関数は明らかにその下に無限の面積を持っています。 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

これは、明確に定義された有限分散を持たないことを意味します。 

ただし、深く見ると、さらに厄介な結果が得られます。:eqref:`eq_exp_def` を使って平均を計算してみましょう。変数の変更式を使用すると、 

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

内側の積分は対数の定義なので、これは本質的に $\log(\infty) = \infty$ なので、明確な平均値もありません。 

機械学習の科学者は、ほとんどの場合、これらの問題に対処する必要がないようにモデルを定義しています。また、ほとんどの場合、平均と分散が明確に定義された確率変数を扱うことになります。しかし、*重い尾* を持つ確率変数 (つまり、大きな値を得る確率が平均や分散などを未定義にするほど大きい確率変数) を持つ確率変数は、物理システムのモデリングに役立つので、それらが存在することを知っておく価値があります。 

### ジョイント密度関数

上記の作業はすべて、単一の実数値の確率変数を使用して作業していることを前提としています。しかし、2つ以上の潜在的に相関の高い確率変数を扱っているとしたらどうでしょうか？このような状況は機械学習では当たり前のことです。$R_{i, j}$ のような確率変数は、イメージの $(i, j)$ 座標のピクセルの赤の値をエンコードしたり、$t$ の株価によって与えられる確率変数である $P_t$ を想像してみてください。近くのピクセルは色が似ている傾向があり、近くの時間は似た価格になる傾向があります。これらを個別の確率変数として扱うことはできず、成功するモデルが作成されることが期待されます (:numref:`sec_naive_bayes` では、このような仮定のためにパフォーマンスが低下するモデルが見られます)。これらの相関する連続確率変数を扱う数学言語を開発する必要があります。 

ありがたいことに、:numref:`sec_integral_calculus` の多重積分によって、このような言語を開発することができます。わかりやすくするために、相関できる 2 つの確率変数 $X, Y$ があるとします。次に、単一の変数の場合と同様に、次の質問をすることができます。 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ).
$$

単一変数の場合と同様の推論は、これがおおよその値になるはずであることを示しています。 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

いくつかの関数$p(x, y)$。これは $X$ および $Y$ のジョイント密度と呼ばれます。単一変数の場合で見たように、同様のプロパティがこれにも当てはまります。すなわち: 

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$。

このようにして、相関する可能性のある複数の確率変数を扱うことができます。3 つ以上の確率変数を扱う場合、$p(\mathbf{x}) = p(x_1, \ldots, x_n)$ を考慮すると、多変量密度を必要な数の座標に拡張できます。非負であり、総積分が1であるという同じ性質が依然として保持されます。 

### 周辺分布複数の変数を扱うとき、その関係を無視して、「この変数はどのように分布しているのか？」と聞きたくなることがよくあります。このような分布を*周辺分布* と呼びます。 

具体的には、$p _ {X, Y}(x, y)$ で与えられる結合密度をもつ 2 つの確率変数 $X, Y$ があるとします。下付き文字を使用して、密度がどの確率変数であるかを示します。周辺分布を求める問題は、この関数を使用して $p _ X(x)$ を求めることです。 

ほとんどの場合と同様に、何が真実であるかを理解するために、直感的な図に戻るのが最善です。密度は関数$p _ X$であることを思い出してください。 

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

$Y$ については言及されていませんが、$p _{X, Y}$ がすべて与えられている場合、何とかして $Y$ を含める必要があります。まず、これが次のものと同じであることがわかります。 

$$
P(X \in [x, x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

密度は、この場合に何が起こるかを直接教えてくれません。$y$でも小さな間隔に分割する必要があるので、これを次のように書くことができます 

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![By summing along the columns of our array of probabilities, we are able to obtain the marginal distribution for just the random variable represented along the $x$-axis.](../img/marginal.svg)
:label:`fig_marginal`

これは :numref:`fig_marginal` に示すように、一直線に並んだ一連の正方形に沿って密度の値を足し合わせるように指示します。実際、両側からイプシロンの1つの因数をキャンセルし、右側の合計が$y$を超える積分であることを認識した後、次のように結論付けることができます。 

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

こうして私達は見る 

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

これは、周辺分布を得るために、関心のない変数を積分することを示しています。このプロセスはしばしば、不要な変数を*積分* または*疎外*する*と呼ばれます。 

### 共分散

複数の確率変数を扱う場合、知っておくと便利な要約統計量が 1 つあります。*共分散* です。これは、2 つの確率変数が一緒に変動する度合いを測定します。 

2 つの確率変数 $X$ と $Y$ があるとします。まず、確率が $p_{ij}$ で $(x_i, y_j)$ の値を取る離散的であると仮定します。この場合、共分散は次のように定義されます。 

$$\sigma_{XY} = \mathrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

これを直感的に考えるには、次の確率変数のペアを考えてみましょう。$X$ が値 $1$ と $3$ を取り、$Y$ が値 $-1$ と $3$ を取ると仮定します。次の確率があると仮定します。 

$$
\begin{aligned}
P(X = 1 \; \text{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \text{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

$p$は$[0,1]$のパラメータです。$p=1$ の場合、両者は常に同時に最小値または最大値になり、$p=0$ の場合は、反転した値を同時に取ることが保証されます (片方が小さい場合は大きく、その逆も同様です)。$p=1/2$ の場合、4 つの可能性はすべて同等であり、どちらも関連しているはずです。共分散を計算してみましょう。最初に $\mu_X = 2$ と $\mu_Y = 1$ に注意してください。したがって、:eqref:`eq_cov_def` を使用して計算できます。 

$$
\begin{aligned}
\mathrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

$p=1$ (両者が同時に最大に正または負になる場合) の共分散は $2$ です。$p=0$ (反転した場合) の場合、共分散は $-2$ になります。最後に、$p=1/2$ (両者が無関係な場合) の場合、共分散は $0$ になります。したがって、共分散はこれら2つの確率変数がどのように関連しているかを測定することがわかります。 

共分散について簡単に説明すると、これらの線形関係のみが測定されるということです。$\{-2, -1, 0, 1, 2\}$ から $Y$ が等しい確率でランダムに選択される $X = Y^2$ のような、より複雑なリレーションシップは失われる可能性があります。実際、簡単な計算では、一方が他方の決定論的関数であるにもかかわらず、これらの確率変数は共分散がゼロであることがわかります。 

連続確率変数の場合、ほぼ同じ話が成り立ちます。この時点では、離散と連続の間の遷移にかなり慣れているので、導出なしの :eqref:`eq_cov_def` の連続アナログを提供します。 

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

可視化のために、調整可能な共分散をもつ確率変数のコレクションを見てみましょう。

```{.python .input}
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

共分散の性質をいくつか見てみましょう。 

* 任意の確率変数 $X$、$\mathrm{Cov}(X, X) = \mathrm{Var}(X)$ です。
* 任意の確率変数 $X, Y$ と数値 $a$ および $b$ の場合、$\mathrm{Cov}(aX+b, Y) = \mathrm{Cov}(X, aY+b) = a\mathrm{Cov}(X, Y)$
* $X$ と $Y$ が独立している場合は $\mathrm{Cov}(X, Y) = 0$ になります。

さらに、共分散を使用して、前に見た関係を拡張できます。$X$と$Y$は2つの独立確率変数であることを思い出してください。 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$

共分散の知識があれば、この関係を拡大できます。実際、一部の代数では一般的にそれを示すことができます。 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y).
$$

これにより、相関する確率変数の分散和則を一般化することができます。 

### 相関関係

平均と分散の場合と同様に、単位を考えてみましょう。$X$ をある単位 (インチなど) で測定し、$Y$ を別の単位 (たとえばドル) で測定した場合、共分散はこれら 2 つの単位 $\text{inches} \times \text{dollars}$ の積で測定されます。これらの単位は解釈が難しい場合があります。この場合私たちがしばしば望むのは、関連性の単位なしの測定です。実際、正確な量的相関は気にせず、相関が同じ方向にあるかどうか、その関係がどれほど強いかを尋ねることがよくあります。 

何が理にかなっているかを見るために、思考実験を行ってみましょう。インチとドルの確率変数をインチとセントに変換するとします。この場合、確率変数 $Y$ に $100$ が乗算されます。定義を練ると、$\mathrm{Cov}(X, Y)$ に $100$ が乗算されることになります。したがって、この場合、単位を変更すると共分散が $100$ の係数だけ変化することがわかります。したがって、相関の単位不変の尺度を見つけるには、$100$ でスケーリングされる別のもので除算する必要があります。確かに明確な候補、標準偏差があります！実際に、*相関係数*を次のように定義すると 

$$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

これは単位なしの値であることがわかります。ちょっとした数学を見ると、この数値は $-1$ から $1$ の間であり、$1$ は最大正の相関を意味し、$-1$ は最大負の相関があることを意味します。 

上の明示的離散の例に戻ると、$\sigma_X = 1$ と $\sigma_Y = 2$ がわかるので、:eqref:`eq_cor_def` を使用して 2 つの確率変数間の相関を計算し、 

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

現在、この範囲は $-1$ から $1$ までで、予想される動作は $1$ が最も相関していることを示し、$-1$ は最小相関を意味します。 

別の例として、$X$ を任意の確率変数として、$Y=aX+b$ を $X$ の線形決定論的関数と見なします。そうすれば、それを計算できます 

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX+b) = a\mathrm{Cov}(X, X) = a\mathrm{Var}(X),$$

:eqref:`eq_cor_def`までには 

$$
\rho(X, Y) = \frac{a\mathrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

したがって、相関は $a > 0$ の場合は $+1$、$a < 0$ の場合は $-1$ であることがわかります。これは、相関が 2 つの確率変数が関係する度合いと方向性を測定するものであり、変動がとる尺度ではないことを示しています。 

調整可能な相関をもつ確率変数のコレクションをもう一度プロットしてみましょう。

```{.python .input}
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

相関関係のいくつかのプロパティを以下に挙げてみましょう。 

* 任意の確率変数 $X$、$\rho(X, X) = 1$ です。
* 任意の確率変数 $X, Y$ と数値 $a$ および $b$ の場合、$\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$
* $X$ と $Y$ が非ゼロ分散で独立している場合、$\rho(X, Y) = 0$ になります。

最後に、これらの数式のいくつかはおなじみのように感じるかもしれません。確かに、$\mu_X = \mu_Y = 0$と仮定してすべてを拡張すると、これが 

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

これは、項の積を項の和の平方根で割った和のように見えます。これは、$p_{ij}$ で重み付けされた異なる座標をもつ 2 つのベクトル $\mathbf{v}, \mathbf{w}$ 間の角度の余弦の公式です。 

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

実際、ノルムを標準偏差に関連させ、相関を角度の余弦と考えると、幾何学から得た直感の多くは、確率変数の考え方に適用できます。 

## 概要 * 連続確率変数は、連続した値をとることができる確率変数です。離散確率変数に比べると扱いにくい技術的な難しさがあります。* 確率密度関数では、ある区間の曲線の下の領域が次の確率を見つける確率を与える関数を与えることで、連続確率変数を扱うことができます。その区間のサンプル点。* 累積分布関数は、確率変数が所定の閾値より小さいことが観測される確率です。離散変数と連続変数を統合する便利な代替視点を提供します。* 平均は確率変数の平均値です。* 分散は、確率変数とその平均の差の予想される二乗です。* 標準偏差は分散の平方根です。これは、確率変数がとる可能性のある値の範囲を測定することと考えることができます。* チェビシェフの不等式により、確率変数を含む明示的な区間を与えることで、この直感を厳格にすることができます。* 結合密度により、相関する確率変数を扱うことができます。不要な確率変数を積分して、目的の確率変数の分布を求めることで、結合密度を疎外することがあります。* 共分散と相関係数は、相関する2つの確率変数間の線形関係を測定する方法を提供します。 

## 演習 1.$x \ge 1$ には $p(x) = \frac{1}{x^2}$ で与えられる密度の確率変数があり、それ以外の場合は $p(x) = 0$ があるとします。$P(X > 2)$って何ですか？2。ラプラス分布は、密度が $p(x = \frac{1}{2}e^{-|x|}$ で与えられる確率変数です。この関数の平均と標準偏差はどれくらいですか？ヒントとして、$\int_0^\infty xe^{-x} \; dx = 1$ と $\int_0^\infty x^2e^{-x} \; dx = 2$ を挙げて。私は通りであなたに近づき、「平均$1$、標準偏差 $2$ の確率変数があり、25\ %$ of my samples taking a value larger than $9$ を観測した」と言います。信じてくれる？なぜ、なぜそうではないのですか？4。2 つの確率変数 $X, Y$ があり、$x, y \in [0,1]$ には $p_{XY}(x, y) = 4xy$、それ以外の場合は $p_{XY}(x, y) = 0$ で結合密度が与えられるとします。$X$ と $Y$ の共分散は何になりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:

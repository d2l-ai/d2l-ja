# ディストリビューション
:label:`sec_distributions`

離散設定と連続設定の両方で確率を処理する方法を学習したので、よく見られる分布をいくつか見てみましょう。機械学習の分野によっては、これらについて非常に精通している必要がある場合もあれば、ディープラーニングの一部の領域についてはまったく理解していない場合もあります。ただし、これはよく理解しておくべき基本的なリストです。まず、共通ライブラリをいくつかインポートしてみましょう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## ベルヌーイ

これは通常遭遇する最も単純な確率変数です。この確率変数は、確率$p$で$1$、確率$1-p$で$0$が出てくるコインフリップをエンコードします。この分布をもつ確率変数 $X$ があれば、次のように書きます。 

$$
X \sim \mathrm{Bernoulli}(p).
$$

累積分布関数は次のようになります。  

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

確率質量関数を以下にプロットします。

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

ここで、累積分布関数 :eqref:`eq_bernoulli_cdf` をプロットしてみましょう。

```{.python .input}
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

$X \sim \mathrm{Bernoulli}(p)$ の場合、次のようになります。 

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$。

ベルヌーイ確率変数から任意の形状の配列を次のようにサンプリングできます。

```{.python .input}
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 離散均一

次に一般的に遭遇する確率変数は離散一様分布です。ここでの説明では、整数 $\{1, 2, \ldots, n\}$ でサポートされていると仮定しますが、それ以外の値のセットは自由に選択できます。この文脈での「*uniform*」という言葉の意味は、あり得るすべての値が同等であるということです。$i \in \{1, 2, 3, \ldots, n\}$ の各値の確率は $p_i = \frac{1}{n}$ です。この分布をもつ確率変数 $X$ を次のように表します。 

$$
X \sim U(n).
$$

累積分布関数は次のようになります。  

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

まず、確率質量関数をプロットしてみましょう。

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

ここで、累積分布関数 :eqref:`eq_discrete_uniform_cdf` をプロットしてみましょう。

```{.python .input}
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(n)$ の場合、次のようになります。 

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$。

次のように、離散一様確率変数から任意の形状の配列をサンプリングできます。

```{.python .input}
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 連続均一

次に、連続一様分布について考えてみましょう。この確率変数の背後にある考え方は、離散一様分布の $n$ を増やし、区間 $[a, b]$ 内に収まるようにスケーリングすると、$[a, b]$ の任意の値をすべて同じ確率で抽出する連続確率変数に近づくというものです。この分布を次のように表します。 

$$
X \sim U(a, b).
$$

確率密度関数は次のようになります。  

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

累積分布関数は次のようになります。  

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

まず、確率密度関数 :eqref:`eq_cont_uniform_pdf` をプロットしてみましょう。

```{.python .input}
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

ここで、累積分布関数 :eqref:`eq_cont_uniform_cdf` をプロットしてみましょう。

```{.python .input}
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(a, b)$ の場合、次のようになります。 

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$。

次のように、一様確率変数から任意の形状の配列をサンプリングできます。デフォルトで $U(0,1)$ からサンプリングするので、別の範囲が必要な場合はスケーリングする必要があります。

```{.python .input}
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 二項式

もう少し複雑にして、*二項* 確率変数を調べてみましょう。この確率変数は、$n$ の独立した実験のシーケンスを実行し、それぞれが成功する確率が $p$ で、成功すると予想される成功回数を尋ねることから生じます。 

これを数学的に表現しよう。各実験は独立確率変数 $X_i$ で、$1$ を使用して成功をエンコードし、$0$ を使用して失敗をエンコードします。それぞれが独立したコインフリップであり、確率$p$で成功するので、$X_i \sim \mathrm{Bernoulli}(p)$と言えます。その場合、二項確率変数は次のようになります。 

$$
X = \sum_{i=1}^n X_i.
$$

この場合は、次のように書きます。 

$$
X \sim \mathrm{Binomial}(n, p).
$$

累積分布関数を得るには、$\ binom {n} {k} =\ frac {n!} で正確に$k$の成功が得られることに注意する必要があります。{k!(n-k)!}$ ways each of which has a probability of $p^k (1-p) ^ {n-k} $ 発生しています。したがって、累積分布関数は次のようになります。 

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

まず、確率質量関数をプロットしてみましょう。

```{.python .input}
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

ここで、累積分布関数 :eqref:`eq_binomial_cdf` をプロットしてみましょう。

```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

$X \sim \mathrm{Binomial}(n, p)$ の場合、次のようになります。 

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$。

これは、$n$ ベルヌーイ確率変数の合計に対する期待値の線形性、および独立確率変数の和の分散が分散の和であるという事実に由来します。これは以下のようにサンプリングできます。

```{.python .input}
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## ポアソンさっそく思考実験をしてみよう。私たちはバス停に立っていて、次の1分間に何本のバスが到着するか知りたいです。$X^{(1)} \sim \mathrm{Bernoulli}(p)$ について考えてみましょう。$X^{(1)} \sim \mathrm{Bernoulli}(p)$ は、単に 1 分間のウィンドウにバスが到着する確率です。都心から遠く離れたバス停では、これはかなり良い近似値になるかもしれません。一分間に複数のバスを見ることはないかもしれない。 

ただし、混雑している場所にいる場合は、2台のバスが到着する可能性もあれば、到着する可能性もあります。これをモデル化するには、確率変数を最初の 30 秒間、または 2 番目の 30 秒間で 2 つの部分に分割します。この場合、次のように書くことができます。 

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

$X^{(2)}$ は合計の合計であり、$X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$ です。その場合、合計の分布は $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$ になります。 

なぜここで止まるの？その分を $n$ の部分に分割していきましょう。上記と同じ推論で、 

$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

これらの確率変数を考えてみましょう。前のセクションでは、:eqref:`eq_eq_poisson_approx` の平均は $\mu_{X^{(n)}} = n(p/n) = p$ で、分散 $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$ であることがわかっています。$n \rightarrow \infty$ を取ると、これらの数値は $\mu_{X^{(\infty)}} = p$ に安定し、分散 $\sigma_{X^{(\infty)}}^2 = p$ であることがわかります。これは、この無限の細分化限界で定義できる何らかの確率変数が存在する可能性があることを示しています。   

現実の世界ではバスの到着数を数えるだけなので、これはそれほど驚くべきことではありませんが、数学モデルが明確に定義されていることは素晴らしいことです。この議論は、*希少事象の法則*として公式にすることができる。 

この推論を注意深く実行すると、次のモデルに到達できます。確率で$\{0,1,2, \ldots\}$という値をとる確率変数であれば、$X \sim \mathrm{Poisson}(\lambda)$と言います。 

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

$\lambda > 0$ という値は*rate* (または*shape* パラメーター) と呼ばれ、1 単位時間内に予想される平均到着数を表します。   

この確率質量関数を合計して、累積分布関数を求めることができます。 

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

まず、確率質量関数 :eqref:`eq_poisson_mass` をプロットしてみましょう。

```{.python .input}
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

ここで、累積分布関数 :eqref:`eq_poisson_cdf` をプロットしてみましょう。

```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

上で見たように、平均と分散は特に簡潔です。$X \sim \mathrm{Poisson}(\lambda)$ の場合、次のようになります。 

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$。

これは以下のようにサンプリングできます。

```{.python .input}
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gaussian さて、違うけれども関連性のある実験をしてみよう。$n$ 独立した $\mathrm{Bernoulli}(p)$ 測定 $X_i$ をもう一度実行しているとしましょう。これらの和の分布は $X^{(n)} \sim \mathrm{Binomial}(n, p)$ です。$n$ が増えて $p$ が減るという制限を取るのではなく、$p$ を修正して $n \rightarrow \infty$ を送りましょう。この場合 $\mu_{X^{(n)}} = np \rightarrow \infty$ と $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$ なので、この制限を明確に定義する必要があると考える理由はありません。 

しかし、すべての希望が失われるわけではありません！定義して、平均と分散がうまく動作するようにしましょう。 

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

これは平均が0で分散が1であることがわかるので、何らかの限定的な分布に収束すると考えるのもっともらしいです。これらの分布がどのように見えるかをプロットすれば、それが機能することをさらに確信できるようになります。

```{.python .input}
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

注意すべき点の1つは、ポアソンの場合と比較して、標準偏差で割っていることです。これは、起こり得る結果をますます小さな領域に絞り込んでいることを意味します。これは、リミットが離散的ではなく連続的になることを示しています。 

何が起きるかの導出はこの文書の範囲外ですが、*中心極限定理*では $n \rightarrow \infty$ としてガウス分布 (または場合によっては正規分布) が生じるとされています。より明示的に、どの $a, b$ でも次のようになります。 

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

ここで、確率変数は与えられた平均 $\mu$ と分散 $\sigma^2$ をもつ正規分布であるとします。$X$ に密度があれば $X \sim \mathcal{N}(\mu, \sigma^2)$ と書きます。 

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

まず、確率密度関数 :eqref:`eq_gaussian_pdf` をプロットしてみましょう。

```{.python .input}
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

ここで、累積分布関数をプロットしてみましょう。この付録では説明しませんが、Gaussian c.d.f. は、より基本的な関数という点では閉形式の公式を持ちません。この積分を数値的に計算する方法を提供する `erf` を使用します。

```{.python .input}
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

熱心な読者は、これらの用語のいくつかを認識するでしょう。実際、:numref:`sec_integral_calculus`でこの積分に遭遇しました。この $p_X(x)$ の総面積が 1 であり、したがって有効な密度であることを確認するには、まさにその計算が必要です。 

コインフリップを扱うという私たちの選択は計算を短くしましたが、その選択については何も基本的なものではありませんでした。実際に、同一の分布を持つ独立確率変数$X_i$の集合をとると、 

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

それから 

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

はほぼガウス分布になります。これを機能させるには追加の要件が必要ですが、最も一般的には $E[X^4] < \infty$ ですが、その理念は明確です。 

中心極限定理は、ガウス分布が確率、統計、機械学習の基本となる理由です。私たちが測定したものが多くの小さな独立した寄与の合計であると言えるときはいつでも、測定されるものはガウスに近いと仮定できます。   

ガウスにはもっと魅力的な性質がたくさんありますが、ここでもう1つ議論したいと思います。ガウス分布は*最大エントロピー分布*と呼ばれるものです。:numref:`sec_information_theory`ではエントロピーをより深く掘り下げますが、この時点で知る必要があるのは、これがランダム性の尺度であるということだけです。厳密な数学的な意味では、ガウシアンは平均と分散が固定された確率変数の「最もランダムな選択」と考えることができます。したがって、確率変数に平均と分散があることがわかっている場合、ガウス分布はある意味で最も保守的な分布の選択です。 

セクションを閉じるには、$X \sim \mathcal{N}(\mu, \sigma^2)$の場合は思い出してください。 

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$。

以下に示すように、ガウス (または標準正規) 分布からサンプリングできます。

```{.python .input}
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 指数関数的ファミリー
:label:`subsec_exponential_family`

上に挙げたすべての分布の共通の特性の一つは、それらがすべて*指数関数的ファミリー*として知られていることです。指数族は、密度が次の形式で表される一連の分布です。 

$$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

この定義は少し微妙なので、詳しく調べてみましょう。   

第1に、$h(\mathbf{x})$は*基礎となるメジャー*または 
*ベースメジャー*。これは、私たちがしている測定の元の選択と見なすことができます 
指数関数的な重みで修正します   

第二に、\ mathbb {R} ^l$ の中に $\ boldsymbol {\ eta} = (\ eta_1,\ eta_2,...,\ eta_l)\ が*自然パラメータ* または*標準パラメータ*と呼ばれるベクトル$\ boldsymbol {\ eta} = (\ eta_1,\ eta_2,...,\ eta_l)\ があります。ベースメジャーがどのように修正されるかを定義します。\ mathbb {R} ^n$ and exponentiated. The vector $T (\ mathbf {x}) = (x_1, x_2,..., x_n)\ n$ and exponentiated. The vector $T (\ mathbf {x}) = (T_1 (\ mathbf {x}), T_1 (\ mathbf {x}), T_1 (\ mathbf {x}), T_1 (\ mathbf {x}), T_1 (\ mathbf {x}), T_$2 (\ mathbf {x}),..., t_L (\ mathbf {x}) $は$\boldsymbol{\eta}$の*十分な統計*と呼ばれています。$T(\mathbf{x})$ で表される情報は確率密度の計算に十分であり、サンプル $\mathbf{x}$ のその他の情報は必要ないため、この名前が使用されます。 

第3に、$A(\boldsymbol{\eta})$があります。これは*キュムラント関数*と呼ばれ、上記の分布 :eqref:`eq_exp_pdf` が確実に1に積分されます。 

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp}
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

具体的には、ガウス分布を考えてみましょう。$\mathbf{x}$ が一変量変数であると仮定すると、密度が 

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \mathrm{exp} 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

これは、指数ファミリーの定義を次のものと一致させます。 

* *基礎となる測定値*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *自然パラメータ*: $\ ボールド記号 {\ eta} =\ begin {bmatrix}\ eta_1\\\ eta_2
\ end {bmatrix} =\ begin {bmatrix}\ frac {\ mu} {\ sigma^2}\\ frac {1} {2\ sigma^2}\ end {bmatrix} $,
* *十分な統計情報*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$
* *キュムラント関数*: $A ({\ 太字\ eta}) =\ frac {1} {2\ sigma^2}\ mu^2 +\ log (\ sigma)
=\ frac {\ eta_1^2} {4\ eta_2}-\ frac {1} {2}\ log (2\ eta_2) $ 

上記の各用語の正確な選択はいくぶん恣意的であることは注目に値する。実際、重要な特徴は、分布が正確な形式そのものではなく、この形式で表現できることです。 

:numref:`subsec_softmax_and_derivatives` で言及しているように、広く使用されている手法は、最終的な出力 $\mathbf{y}$ が指数ファミリー分布に従うと仮定することです。指数ファミリーは、機械学習で頻繁に見られる一般的で強力な分布ファミリーです。 

## まとめ * ベルヌーイ確率変数は、はい/いいえの結果をもつ事象をモデル化するのに使用できます。* 離散一様分布モデルは、有限の可能性のセットから選択します。* 連続一様分布は区間から選択します。* 二項分布は一連のベルヌーイ確率変数をモデル化し、カウントします。* ポアソン確率変数は、まれな事象の到来をモデル化します。* ガウス確率変数は、多数の独立確率変数を足した結果をモデル化します。* 上記の分布はすべて指数族に属します。 

## 演習

1. 2つの独立した二項確率変数 $X, Y \sim \mathrm{Binomial}(16, 1/2)$ の差 $X-Y$ である確率変数の標準偏差は何ですか。
2. ポアソン確率変数 $X \sim \mathrm{Poisson}(\lambda)$ を取り、$(X - \lambda)/\sqrt{\lambda}$ を $\lambda \rightarrow \infty$ と見なすと、これがほぼガウス分布になることを示すことができます。なぜこれが意味をなすのですか？
3. $n$ 要素の 2 つの離散一様確率変数の和に対する確率質量関数は何ですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab:

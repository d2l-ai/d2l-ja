# ナイーブベイズ分類

複雑な最適化アルゴリズムやGPUについて心配する前に、最初の分類器として、シンプルな統計的指標と条件付き独立を利用した分類器をデプロイしてみましょう。学習というものは、仮定を置くことにほかなりません。見たことのない新しいデータを分類するためには、互いに*類似*しているデータに関して仮定を置く必要があります。

最も利用されている (そして非常にシンプルな) アルゴリズムはナイーブベイズ分類器でしょう。分類のタスクを自然に表すとすれば、「その特徴から、最も可能性のあるラベルはなんですか?」という確率を考えた質問になるでしょう。数式で表すと、予測$\hat{y}$は以下の式で与えられるでしょう。

$$\hat{y} = \text{argmax}_y \> p(y | \mathbf{x})$$

残念ながら、これはすべての値 $\mathbf{x} = x_1, ..., x_d$に対して $p(y | \mathbf{x})$ を予測する必要があります。各特徴が2値のうちの1つをとる場合を考えてみましょう。例えば、特徴$x_1 = 1$が「りんご」がある文書に現れていることを、$x_1 = 0$は現れないことを表すとしましょう。$30$個の2値の特徴があれば、それは$2^{30}$、つまり10億以上の値をとりうる入力ベクトル$\mathbf{x}$を分類する必要があります。

さらに、学習はどうやって行うでしょうか? 対応するラベルを予測するために、すべての起こりうる値を見なければならないのであれば、パターンを学習しているとはいえず、それはデータセットを記憶しているに過ぎません。幸運にも条件付き独立に関するいくつかの仮定を置くことによって、帰納的な性質を設けることができ、学習データからそこまで多くないデータを選んで、一般化するようなモデルを作ることができます。

始めるにあたって、ベイズの定理を利用して、以下のように分類器を表現しましょう。

$$\hat{y} = \text{argmax}_y \> \frac{p( \mathbf{x} | y) p(y)}{p(\mathbf{x})}$$

分母は正規化項$p(\mathbf{x})$でラベル$y$の値に依存しません。結果として、異なる$y$の値に対して、分子を比較することだけ考えれば良いです。たとえ、分母を計算することが非常に難しいことがわかっても、それを無視することができ、分子を評価するだけで良いです。幸運にも、正規化するための定数を計算したい場合は、$\sum_y p(y | \mathbf{x}) = 1$ということがわかっているので、その項をいつでも計算することができます。確率のチェインルールを利用することで、$p( \mathbf{x} | y)$の項は以下のように表現することができます。

$$p(x_1 |y) \cdot p(x_2 | x_1, y) \cdot ... \cdot p( x_d | x_1, ..., x_{d-1} y)$$

この式自体では、まだ何も進んでいません。まだ、大雑把に言って、$2^d$のパラメータを予測しなければならないからです。しかし、*ラベルに対して各特徴が互いに条件付き独立である*仮定を置くことによって、$\prod_i p(x_i | y)$という単純な形式に変換できます。予測式は以下のようになります。

$$ \hat{y} = \text{argmax}_y \> = \prod_i p(x_i | y) p(y)$$

$\prod_i p(x_i | y)$ の各項を予測することは、たった1つのパラメータを予測することと同じになりました。条件付き独立の仮定は、特徴の数に応じた指数的な依存関係を線形な関係へと変換し、パラメータ数の観点から複雑さを解消しました。さらに、以前見たことのないデータに対して予測行うこともできます。なぜなら、$p(x_i | y)$を求めればよいだけで、これをたくさんの異なる文書にもとづいて予測することができるからです。

ラベルが与えられたとき、特徴が互いにすべて独立であるという重要な仮定、つまり$p(\mathbf{x} | y) = \prod_i p(x_i | y)$についてより詳しく見てみましょう。Eメールをスパムかそうでないかに分類することを考えます。`ナイジェリア`、`王子`、`金`、`リッチ`といった単語が現れればは、おそらくスパムの可能性があり、一方`理論`、`ネットワーク`,
`ベイズ`、`統計学`という単語が現れれば、それは口座番号を聞き出す巧みな試みと関係ないものと考えられるでしょう。そこで、それぞれのクラスに対して各単語の起こりうる確率をモデル化し、文章の尤もらしさをスコア化しましょう。実際のところ、これはいわゆる[ベイズのパムフィルタ](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)として、長い間にわたって機能してきました。

## 光学文字認識 (OCR; Optical Character Recognition)

Since images are much easier to deal with, we will illustrate the workings of a Naive Bayes classifier for distinguishing digits on the MNIST dataset. The problem is that we don't actually know $p(y)$ and $p(x_i | y)$. So we need to *estimate* it given some training data first. This is what is called *training* the model. Estimating $p(y)$ is not too hard. Since we are only dealing with 10 classes, this is pretty easy - simply count the number of occurrences $n_y$ for each of the digits and divide it by the total amount of data $n$. For instance, if digit 8 occurs $n_8 = 5,800$ times and we have a total of $n = 60,000$ images, the probability estimate is $p(y=8) = 0.0967$.

Now on to slightly more difficult things—$p(x_i | y)$. Since we picked black and white images, $p(x_i | y)$ denotes the probability that pixel $i$ is switched on for class $y$. Just like before we can go and count the number of times $n_{iy}$ such that an event occurs and divide it by the total number of occurrences of y, i.e. $n_y$. But there's something slightly troubling: certain pixels may never be black (e.g. for very well cropped images the corner pixels might always be white). A convenient way for statisticians to deal with this problem is to add pseudo counts to all occurrences. Hence, rather than $n_{iy}$ we use $n_{iy}+1$ and instead of $n_y$ we use $n_{y} + 1$. This is also called [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing).

```{.python .input  n=1}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import mxnet as mx
from mxnet import nd
import numpy as np

# We go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the counters
xcount = nd.ones((784,10))
ycount = nd.ones((10))

for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))

# using broadcast again for division
py = ycount / ycount.sum()
px = (xcount / ycount.reshape(1,10))
```

```{.python .input  n=9}
for data, label in mnist_train:
    y = int(label)
    ycount[y] += 1
    xcount[:,y] += data.reshape((784))
```

Now that we computed per-pixel counts of occurrence for all pixels, it's time to see how our model behaves. Time to plot it. This is where it is so much more convenient to work with images. Visualizing 28x28x10 probabilities (for each pixel for each class) would typically be an exercise in futility. However, by plotting them as images we get a quick overview. The astute reader probably noticed by now that these are some mean looking digits ...

```{.python .input  n=2}
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print('Class probabilities', py)
```

Now we can compute the likelihoods of an image, given the model. This is statistician speak for $p(x | y)$, i.e. how likely it is to see a particular image under certain conditions (such as the label). Our Naive Bayes model which assumed that all pixels are independent tells us that

$$p(\mathbf{x} | y) = \prod_{i} p(x_i | y)$$

Using Bayes' rule, we can thus compute $p(y | \mathbf{x})$ via

$$p(y | \mathbf{x}) = \frac{p(\mathbf{x} | y) p(y)}{\sum_{y'} p(\mathbf{x} | y')}$$

Let's try this ...

```{.python .input  n=3}
# Get the first test item
data, label = mnist_test[0]
data = data.reshape((784,1))

# Compute the per pixel conditional probabilities
xprob = (px * data + (1-px) * (1-data))
# Take the product
xprob = xprob.prod(0) * py
print('Unnormalized Probabilities', xprob)
# Normalize
xprob = xprob / xprob.sum()
print('Normalized Probabilities', xprob)
```

This went horribly wrong! To find out why, let's look at the per pixel probabilities. They're typically numbers between $0.001$ and $1$. We are multiplying $784$ of them. At this point it is worth mentioning that we are calculating these numbers on a computer, hence with a fixed range for the exponent. What happens is that we experience *numerical underflow*, i.e. multiplying all the small numbers leads to something even smaller until it is rounded down to zero. At that point we get division by zero with `nan` as a result.

To fix this we use the fact that $\log a b = \log a + \log b$, i.e. we switch to summing logarithms. This will get us unnormalized probabilities in log-space. To normalize terms we use the fact that

$$\frac{\exp(a)}{\exp(a) + \exp(b)} = \frac{\exp(a + c)}{\exp(a + c) + \exp(b + c)}$$

In particular, we can pick $c = -\max(a,b)$, which ensures that at least one of the terms in the denominator is $1$.

```{.python .input  n=4}
logpx = nd.log(px)
logpxneg = nd.log(1-px)
logpy = nd.log(py)

def bayespost(data):
    # We need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpost = logpy.copy()
    logpost += (logpx * data + logpxneg * (1-data)).sum(0)
    # Normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpost -= nd.max(logpost)
    # Compute the softmax using logpx
    post = nd.exp(logpost).asnumpy()
    post /= np.sum(post)
    return post

fig, figarr = plt.subplots(2, 10, figsize=(10, 3))

# Show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)

    # Bar chart and image of digit
    figarr[1, ctr].bar(range(10), post)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1

    if ctr == 10:
        break

plt.show()
```

As we can see, this classifier works pretty well in many cases. However, the second last digit shows that it can be both incompetent and overly confident of its incorrect estimates. That is, even if it is horribly wrong, it generates probabilities close to 1 or 0. Not a classifier we should use very much nowadays any longer. To see how well it performs overall, let's compute the overall accuracy of the classifier.

```{.python .input  n=5}
# Initialize counter
ctr = 0
err = 0

for data, label in mnist_test:
    ctr += 1
    x = data.reshape((784,1))
    y = int(label)

    post = bayespost(x)
    if (post[y] < post.max()):
        err += 1

print('Naive Bayes has an error rate of', err/ctr)
```

Modern deep networks achieve error rates of less than 0.01. While Naive Bayes classifiers used to be popular in the 80s and 90s, e.g. for spam filtering, their heydays are over. The poor performance is due to the incorrect statistical assumptions that we made in our model: we assumed that each and every pixel are *independently* generated, depending only on the label. This is clearly not how humans write digits, and this wrong assumption led to the downfall of our overly naive (Bayes) classifier. Time to start building Deep Networks.

## Summary

* Naive Bayes is an easy to use classifier that uses the assumption
  $p(\mathbf{x} | y) = \prod_i p(x_i | y)$.
* The classifier is easy to train but its estimates can be very wrong.
* To address overly confident and nonsensical estimates, the
  probabilities $p(x_i|y)$ are smoothed, e.g. by Laplace
  smoothing. That is, we add a constant to all counts.
* Naive Bayes classifiers don't exploit any correlations between
  observations.

## Exercises

1. Design a Naive Bayes regression estimator where $p(x_i | y)$ is a normal distribution.
1. Under which situations does Naive Bayes work?
1. An eyewitness is sure that he could recognize the perpetrator with 90% accuracy, if he were to encounter him again.
   * Is this a useful statement if there are only 5 suspects?
   * Is it still useful if there are 50?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2320)

![](../img/qr_naive-bayes.svg)

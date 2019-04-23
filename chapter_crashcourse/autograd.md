# 自動微分

機械学習では、私達はモデルを学習させ、次々と更新していきます。モデルは、データを見れば見るほど、良くなっていきます。通常、*良くなる*ということは、モデルがどの程度*悪いか*をスコアとして表す*ロス関数*を最小化することを意味します。ニューラルネットワークとともに、パラメータに関して微分可能なロス関数を選択します。平たく言えば、モデルの各パラメータに対して、ロスをどの程度*増加*または*減少*させるかを決めることができるのです。微分をとるという計算自体は率直で、基本的な計算しか必要しませんが、複雑なモデルに対して人手で行うのは非常に困難で間違いやすいものです。

autogradというパッケージは、自動で微分を計算することによって、この作業を加速させました。多くの他のライブラリが、自動微分を行うためにシンボリックなグラフのコンパイルを必要とするのに対し、`autograd`は通常のコードを書くだけで微分をすることができます。モデルにデータを渡すときはいつでも、グラフが`autograd`によってその都度作成され、どのデータにどの演算が実行されて出力を得られるのかが追跡されます。このグラフによって、`autograd`は実行命令を受けると、勾配を逆伝搬します。
*逆伝搬*は単純に計算グラフを追跡して、各パラメータの偏微分を計算することを意味します。
微分のような数学を見慣れていなければ、Appendixの[“Mathematical Basics”](../chapter_appendix/math.md)を参照してください。

```{.python .input  n=1}
from mxnet import autograd, nd
```

## シンプルな例

単純な例として、$y = 2\mathbf{x}^{\top}\mathbf{x}$を列ベクトル $\mathbf{x}$に関して微分してみましょう。まず、変数`x`を作成して、初期値を与えます。

```{.python .input  n=2}
x = nd.arange(4).reshape((4, 1))
print(x)
```

``x``に関する``y``の勾配を計算したら、それを保存するための場所を用意しましょう。NDArrayに対して、``attach_grad()``のメソッドを利用すると、勾配を保存することができます。


```{.python .input  n=3}
x.attach_grad()
```

ここで``y``を計算するためにMXNetによって計算グラフを作成します。それは、記録デバイスを起動して、各変数を生成する正確なパスを取り込むようなものです。

計算グラフの作成にはそれなりの計算を必要とします。そこで陽に計算グラフを作成するよう指示したときだけ、MXNetは計算グラフを作成します。``with autograd.record():``のブロックの中にコードを記述することによって行うことができます。


```{.python .input  n=4}
with autograd.record():
    y = 2 * nd.dot(x.T, x)
print(y)
```

`x`のshapeは(4, 1)なので`y`はスカラーになります。次に、`backward`の関数を呼ぶことで勾配を自動で取得します。`y`がスカラーでなければ、MXNetはデフォルトで`y`の要素の総和をとって新しい変数とし、その変数に対する`x`の勾配を計算します。

```{.python .input  n=5}
y.backward()
```

関数$y = 2\mathbf{x}^{\top}\mathbf{x}$の$\mathbf{x}$に対する勾配は$4\mathbf{x}$です。実際に計算される勾配が正しいか確かめてみましょう。

```{.python .input  n=6}
print((x.grad - 4 * x).norm().asscalar() == 0)
print(x.grad)
```

## 学習モードと推論モード

上記で確認したように、`record`のか関数を呼ぶと、MXNetは勾配を記録して計算します。また、
`autograd`はデフォルトで推論モードから学習モードへと実行モードを切り替えます。このことは、`is_training`関数を実行すると確認することができます。

```{.python .input  n=7}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

同じモデルであっても、学習と推論の各モードで違った動きをする場合があります（DropoutやBatch normalizationという技術を利用したときなど）。他にも、いくつかのモデルは勾配をより容易に計算するために、補助的な変数を追加で保存する場合もあります。以降の章では、これらの違いについて詳細を説明いたします。この章では、それらについて心配する必要はありません。


## Computing the Gradient of Python Control Flow

One benefit of using automatic differentiation is that even if the computational graph of the function contains Python's control flow (such as conditional and loop control), we may still be able to find the gradient of a variable. Consider the following program:  It should be emphasized that the number of iterations of the loop (while loop) and the execution of the conditional judgment (if statement) depend on the value of the input `b`.

```{.python .input  n=8}
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Note that the number of iterations of the while loop and the execution of the conditional statement (if then else) depend on the value of `a`. To compute gradients, we need to `record` the calculation, and then call the `backward` function to calculate the gradient.

```{.python .input  n=9}
a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

Let's analyze the `f` function defined above. As you can see, it is piecewise linear in its input `a`. In other words, for any `a` there exists some constant such that for a given range `f(a) = g * a`. Consequently `d / a` allows us to verify that the gradient is correct:

```{.python .input  n=10}
print(a.grad == (d / a))
```

## Head gradients and the chain rule

*Caution: This part is tricky and not necessary to understanding subsequent sections. That said, it is needed if you want to build new layers from scratch. You can skip this on a first read.*

Sometimes when we call the backward method, e.g. `y.backward()`, where
`y` is a function of `x` we are just interested in the derivative of
`y` with respect to `x`. Mathematicians write this as
$\frac{dy(x)}{dx}$. At other times, we may be interested in the
gradient of `z` with respect to `x`, where `z` is a function of `y`,
which in turn, is a function of `x`. That is, we are interested in
$\frac{d}{dx} z(y(x))$. Recall that by the chain rule

$$\frac{d}{dx} z(y(x)) = \frac{dz(y)}{dy} \frac{dy(x)}{dx}.$$

So, when ``y`` is part of a larger function ``z`` and we want ``x.grad`` to store $\frac{dz}{dx}$, we can pass in the *head gradient* $\frac{dz}{dy}$ as an input to ``backward()``. The default argument is ``nd.ones_like(y)``. See [Wikipedia](https://en.wikipedia.org/wiki/Chain_rule) for more details.

```{.python .input  n=11}
with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([10, 1., .1, .01])
z.backward(head_gradient)
print(x.grad)
```

## Summary

* MXNet provides an `autograd` package to automate the derivation process.
* MXNet's `autograd` package can be used to derive general imperative programs.
* The running modes of MXNet include the training mode and the prediction mode. We can determine the running mode by `autograd.is_training()`.

## Exercises

1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or matrix. At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Redesign an example of finding the gradient of the control flow. Run and analyze the result.
1. In a second-price auction (such as in eBay or in computational advertising), the winning bidder pays the second-highest price. Compute the gradient of the final price with respect to the winning bidder's bid using `autograd`. What does the result tell you about the mechanism? If you are curious to learn more about second-price auctions, check out this paper by [Edelman, Ostrovski and Schwartz, 2005](https://www.benedelman.org/publications/gsp-060801.pdf).
1. Why is the second derivative much more expensive to compute than the first derivative?
1. Derive the head gradient relationship for the chain rule. If you get stuck, use the ["Chain rule" article on Wikipedia](https://en.wikipedia.org/wiki/Chain_rule).
1. Assume $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$ on a graph, where you computed the latter without any symbolic calculations, i.e. without exploiting that $f'(x) = \cos(x)$.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2318)

![](../img/qr_autograd.svg)

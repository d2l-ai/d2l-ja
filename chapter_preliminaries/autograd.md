# 自動微分
:label:`sec_autograd`

:numref:`sec_calculus` で説明したように、ほぼすべての深層学習アルゴリズムに置いて非常に重要なステップです。これらの微分の計算は単純で基本的な計算しか必要ありませんが、複雑なモデルにおいては、手作業で更新していくことは苦痛を伴います（そして、誤りに繋がりやすいです）。

`autograd` というパッケージは、自動で微分を計算する、つまり*自動微分*によって、この作業を加速させました。多くの他のライブラリが、自動微分を行うためにシンボリックなグラフのコンパイルを必要とするのに対し、`autograd`は通常のコードを書くだけで微分をすることができます。モデルにデータを渡すときはいつでも、グラフが`autograd`によってその都度作成され、どのデータにどの演算が実行されて出力を得られるのかが追跡されます。このグラフによって、`autograd`は実行命令を受けると、勾配を逆伝播します。
*逆伝播*は単純に*計算グラフ*を追跡して、各パラメータに関する偏微分を計算することを意味します。

```{.python .input  n=1}
from mxnet import autograd, np, npx
npx.set_np()
```

## シンプルな例

単純な例として、$y = 2\mathbf{x}^{\top}\mathbf{x}$を列ベクトル $\mathbf{x}$に関して微分してみましょう。まず、変数`x`を作成して、初期値を与えます。


```{.python .input  n=2}
x = np.arange(4)
x
```

 $\mathbf{x}$ に関する $y の勾配を計算したら、それを保存するための場所を用意しましょう。パラメータに関する微分を計算するときはいつでも、新しいメモリを割り当てないことが重要です。なぜなら、同じパラメータを数千回、数百万回と頻繁に更新するので、メモリを使い尽くしてしまうからです。

ベクトル$\mathbf{x}$に関するスカラー値の関数において、その勾配はベクトル値になり$\mathbf{x}$と同じshapeをとります。従って、コードにおいて、`x`に関する勾配にアクセスすることは直感的に理解できるでしょう。`attach_grad` のメソッドを使って、`ndarray`の勾配のためのメモリを確保します。

```{.python .input  n=3}
x.attach_grad()
```

After we calculate a gradient taken with respect to `x`,
we will be able to access it via the `grad` attribute.
As a safe default, `x.grad` is initialized as an array containing all zeros.
That is sensible because our most common use case
for taking gradient in deep learning is to subsequently
update parameters by adding (or subtracting) the gradient
to maximize (or minimize) the differentiated function.
By initializing the gradient to an array of zeros,
we ensure that any update accidentally executed
before a gradient has actually been calculated
will not alter the parameters' value.

```{.python .input  n=4}
x.grad
```

ここで$y$を計算しましょう。次に勾配を計算したいので、MXNetに対して計算グラフを必要なときに (on-the-flyで)作成させます。MXNet は、各変数を生成するパスを確実に捉えるために、記憶装置の電源を入れるような状態を想像してみてください。

計算グラフの作成にはそれなりの計算を必要とします。そこで、陽に計算グラフを作成するよう指示したときだけ、MXNetは計算グラフを作成します。``autograd.record``のスコープの中にコードを記述することによって、この挙動を実装することができます。


```{.python .input  n=5}
with autograd.record():
    y = 2 * np.dot(x.T, x)
y
```

`x` は長さ4の  `ndarray` なので、`np.dot` は `x` と `y` の内積を実行し、`y` に割り当てられるスカラーを出力します。次に、`y`の`backward`の関数を呼ぶことで、`x`の各要素に関する `y` の勾配を自動で計算することができます。

```{.python .input  n=6}
y.backward()
```


If we recheck the value of `x.grad`, we will find its contents overwritten by the newly calculated gradient.

```{.python .input  n=7}
x.grad
```

関数$y = 2\mathbf{x}^{\top}\mathbf{x}$の$\mathbf{x}$に関する勾配は$4\mathbf{x}$です。求める勾配が正しく計算されていることを手短に確かめてみましょう。If the two `ndarray`s are indeed the same, then the equality between them holds at every position.

```{.python .input  n=8}
x.grad == 4 * x
```

If we subsequently compute the gradient of another variable
whose value was calculated as a function of `x`,
the contents of `x.grad` will be overwritten.

```{.python .input  n=9}
with autograd.record():
    y = x.sum()
y.backward()
x.grad
```


## Backward for Non-Scalar Variables

Technically, when `y` is not a scalar,
the most natural interpretation of the gradient of `y` (a vector of length $m$)
with respect to `x` (a vector of length $n$) is the *Jacobian* (an $m\times n$ matrix).
For higher-order and higher-dimensional `y` and `x`,
the Jacobian could be a gnarly high-order tensor.

However, while these more exotic objects do show up
in advanced machine learning (including in deep learning),
more often when we are calling backward on a vector,
we are trying to calculate the derivatives of the loss functions
for each constituent of a *batch* of training examples.
Here, our intent is not to calculate the Jacobian
but rather the sum of the partial derivatives
computed individually for each example in the batch.

Thus when we invoke `backward` on a vector-valued variable `y`,
which is a function of `x`,
MXNet assumes that we want the sum of the gradients.
In short, MXNet will create a new scalar variable
by summing the elements in `y`,
and compute the gradient of that scalar variable with respect to `x`.

```{.python .input  n=10}
with autograd.record():
    y = x * x  # y is a vector
y.backward()

u = x.copy()
u.attach_grad()
with autograd.record():
    v = (u * u).sum()  # v is a scalar
v.backward()

x.grad == u.grad
```

## Detaching Computation

Sometimes, we wish to move some calculations
outside of the recorded computational graph.
For example, say that `y` was calculated as a function of `x`,
and that subsequently `z` was calculated as a function of both `y` and `x`.
Now, imagine that we wanted to calculate
the gradient of `z` with respect to `x`,
but wanted for some reason to treat `y` as a constant,
and only take into account the role
that `x` played after `y` was calculated.

Here, we can call `u = y.detach()` to return a new variable `u`
that has the same value as `y` but discards any information
about how `y` was computed in the computational graph.
In other words, the gradient will not flow backwards through `u` to `x`.
This will provide the same functionality as if we had
calculated `u` as a function of `x` outside of the `autograd.record` scope,
yielding a `u` that will be treated as a constant in any `backward` call.
Thus, the following `backward` function computes
the partial derivative of `z = u * x` with respect to `x` while treating `u` as a constant,
instead of the partial derivative of `z = x * x * x` with respect to `x`.

```{.python .input  n=11}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

Since the computation of `y` was recorded,
we can subsequently call `y.backward()` to get the derivative of `y = x * x` with respect to `x`, which is `2 * x`.

```{.python .input  n=12}
y.backward()
x.grad == 2 * x
```

Note that attaching gradients to a variable `x` implicitly calls `x = x.detach()`.
If `x` is computed based on other variables,
this part of computation will not be used in the `backward` function.

```{.python .input  n=13}
y = np.ones(4) * 2
y.attach_grad()
with autograd.record():
    u = x * y
    u.attach_grad()  # Implicitly run u = u.detach()
    z = 5 * u - x
z.backward()
x.grad, u.grad, y.grad
```

## Pythonの制御フローに対する勾配を計算する

自動微分のメリットとして、たとえ計算グラフが複雑なPythonの制御フロー（条件分岐、ループ、任意の関数呼び出し）を含んでいたとしても、その変数の微分を得られる点があります。次のスニペットでは、`while` ループのイテレーション数や `if` 文の評価回数がが、入力 `a` に依存しています。


```{.python .input  n=16}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

再度、勾配を計算するために、その計算を`record`(保存)する必要があり、また`backward`呼び出す必要があります。

```{.python .input  n=17}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

上で定義された関数`f`を解析してみましょう。関数`f`は入力`a`に対する区分線形関数であることは確認できると思います。言い換えれば、どのような`a`に対しても、`k`の値が入力`a`に依存して、`f(a) = k * a`を満たす定数スカラー`k`が存在します。従って、`d / a`という計算を行うことで、その勾配が正しいかどうかを検証することができます。

```{.python .input  n=10}
print(a.grad == (d / a))
```


## 学習モードと推論モード

上記で確認したように、`autograd.record`の関数を呼ぶと、MXNetは以降のブロックの演算を記録します。また、
`autograd.record`は*推論モード*から*学習モード*へと実行モードを切り替えます。このことは、`is_training`関数を実行すると確認することができます。


```{.python .input  n=19}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```
複雑な深層学習モデルを扱うときは、学習するときと、それを使って推論を行うときで、違った挙動をするモデルのアルゴリズムに直面することもあるでしょう。以降の章では、これらの違いの詳細について説明をします。

## まとめ

* MXNetでは微分の処理を自動化する`autograd`パッケージを提供してます。これを利用するためには、偏微分を計算したい変数に関して勾配をまずアタッチします。そして、対象となる値の計算を記録し、その`backward`関数を実行し、変数の `grad` 属性から最終的な勾配を知ることができます。
* `backward`関数で利用される計算の一部を制御するために、勾配をデタッチすることができます。
* MXNetの実行モードには学習モードと推論モードがあります。`autograd.is_training`を呼ぶと、実行モードを知ることができます。

## 練習

1. なぜ2階微分は、1階微分よりもずっと多くの計算を必要とするのでしょうか。
1. After running `y.backward()`, immediately run it again and see what happens.
1. `a`に関する`d`の微分を計算する制御フローを例としてとりあげましたが、`a`をランダムなベクトルや行列に変更するとどうなるでしょうか。このとき、`f(a)`の計算結果はスカラーではなくなってしまいます。どういった結果になるでしょうか。どのように解析すれば良いでしょうか。
1. その制御フローの勾配を計算する例を変えてみましょう。実行して結果を解析してみましょう。
1. $f(x) = \sin(x)$を考えます。そして、$f(x)$と$\frac{df(x)}{dx}$をグラフ化してください。ただし、$\frac{df(x)}{dx}$については、数式の計算を使わない、つまり $f'(x) = \cos(x)$を使わずにグラフ化しましょう。
1. eBayやComputational advertisingのようなセカンド・プライスオークションにおいては、せりに買った人は二番目に高い入札金額を支払います。`autograd`を使って、せりに買った人の入札金額に関する最終的な価格の勾配を計算してみましょう。その結果から、セカンド・プライスオークションのメカニズムについてわかることがありますか? もしセカンド・プライスオークションについてより深く知りたいと思うのであれば、Edelmanの論文  :cite:`Edelman.Ostrovsky.Schwarz.2007` を参照してください。


## [議論](https://discuss.mxnet.io/t/2318)

![](../img/qr_autograd.svg)

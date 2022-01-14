# 自動微分
:label:`sec_autograd`

:numref:`sec_calculus` で説明したように、微分化はほぼすべてのディープラーニング最適化アルゴリズムにおいて重要なステップです。これらの微分を求める計算は簡単で、必要なのは基本的な微積分だけですが、複雑なモデルの場合、更新を手作業で行うのは面倒です (多くの場合、エラーが起こりやすい)。 

ディープラーニングフレームワークは、微分 (*自動微分) を自動的に計算することで、この作業を迅速化します。実際には、設計したモデルに基づいて、システムは*計算グラフ*を構築し、どのデータをどの操作で組み合わせて出力を生成するかを追跡します。自動微分により、システムは後から勾配を逆伝播できます。ここで、*backpropagate* は単に、計算グラフをトレースし、各パラメーターに関する偏微分を埋めることを意味します。 

## 簡単な例

おもちゃの例として、(**列ベクトル $\mathbf{x}$.に関して関数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ を微分する**) に興味があるとしましょう。まず、変数 `x` を作成して初期値を代入します。

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**$\mathbf{x}$ に対する $y$ の勾配を計算する前に、それを格納する場所が必要です。**] 同じパラメータを何千回または何百万回も更新することが多いので、パラメータに対して微分をとるたびに新しいメモリを割り当てないことが重要です。すぐにメモリが足りなくなる可能性があります。ベクトル $\mathbf{x}$ に対するスカラー値関数の勾配は、それ自体がベクトル値であり、$\mathbf{x}$ と同じ形状であることに注意してください。

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**さあ $y$ を計算してみましょう**)

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

`x` は長さ 4 のベクトルなので、`x` と `x` のドット積が実行され、`y` に代入するスカラー出力が得られます。次に、[**`x` の各成分に対する `y` の勾配を自動的に計算できます**] バックプロパゲーション用の関数を呼び出して勾配を出力します。

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(** $\mathbf{x}$ に対する関数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ の勾配は $4\mathbf{x}$.**) 目的の勾配が正しく計算されたことをすぐに確認してみましょう。

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

[**ここで `x` の別の関数を計算してみましょう**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## 非スカラー変数の場合は逆方向

技術的には、`y` がスカラーでない場合、ベクトル `x` に対するベクトル `y` の微分の最も自然な解釈は行列です。高次の高次元の `y` と `x` では、微分結果が高次のテンソルになる可能性があります。 

しかし、これらのよりエキゾチックなオブジェクトは高度な機械学習 ([**ディープラーニング**] を含む) に現れますが、より頻繁に (**ベクトルを逆方向に呼び出す場合**)、トレーニング例の*バッチ*の各構成要素について、損失関数の導関数を計算しようとしています。ここで、(**私たちの意図は**) 微分行列を計算するのではなく、バッチ内で (**例ごとに個別に計算された偏導関数の和**)。

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## 計算のデタッチ

[**一部の計算を記録された計算グラフの外に移動させたい場合があります。**] たとえば、`y` が `x` の関数として計算され、その後 `z` が `y` と `x` の両方の関数として計算されたとします。ここで、`x` に対する `z` の勾配を計算したいが、何らかの理由で `y` を定数として扱い、`y` が計算された後に `x` が果たした役割のみを考慮に入れたいと想像してください。 

ここで `y` をデタッチすると、`y` と同じ値を持つ新しい変数 `u` が返されますが、`y` が計算グラフでどのように計算されたかに関する情報はすべて破棄されます。つまり、勾配は `u` から `x` まで逆方向に流れません。したがって、次のバックプロパゲーション関数は `x` に対する `z = x * x * x` の偏微分ではなく `u` を定数として扱い、`x` に対する `z = u * x` の偏微分を計算します。

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

`y` の計算が記録されたので、その後 `y` でバックプロパゲーションを呼び出して `x` に対する `y = x * x` の微分 (`2 * x`) を得ることができます。

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Python 制御フローの勾配を計算する

自動微分を使用する利点の 1 つは、(**迷路の Python 制御フローを通過する必要がある関数**) (条件式、ループ、任意の関数呼び出しなど) の計算グラフを [**たとえ**] 構築して、(**結果の変数の勾配を計算できる**)次のスニペットでは、`while` ループの反復回数と `if` ステートメントの評価はどちらも入力 `a` の値に依存することに注意してください。

```{.python .input}
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

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

勾配を計算してみましょう。

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

これで、上で定義した `f` 関数を解析できます。入力 `a` では区分的線形であることに注意してください。つまり、`a` には `f(a) = k * a` のような定数スカラー `k` が存在し、`k` の値は入力 `a` に依存します。したがって `d / a` では、勾配が正しいことを検証できます。

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## [概要

* ディープラーニングフレームワークでは、微分の計算を自動化できます。これを使用するには、まず偏微分を求める変数に勾配を付けます。次に、目標値の計算を記録し、その関数を逆伝播のために実行し、結果の勾配にアクセスします。

## 演習

1. 二次導関数が一次導関数より計算コストがかかるのはなぜですか？
1. バックプロパゲーション用に関数を実行したら、ただちにその関数をもう一度実行して、何が起こるかを確認してください。
1. `a` に対する `d` の微分を計算する制御フローの例では、変数 `a` をランダムなベクトルまたは行列に変更するとどうなるでしょうか。この時点では、`f(a)` の計算結果はスカラーではなくなります。結果はどうなりますか？これをどのように分析するのですか？
1. 制御フローの勾配を求める例を再設計します。結果を実行して解析します。
1. $f(x) = \sin(x)$ にしましょう。$f(x)$ と $\frac{df(x)}{dx}$ をプロットします。後者は $f'(x) = \cos(x)$ を利用せずに計算されます。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

# 単変数微積分
:label:`sec_single_variable_calculus`

:numref:`sec_calculus` では、微分積分の基本要素を見ました。このセクションでは、微積分の基礎と、機械学習のコンテキストで微積分を理解して適用する方法について詳しく説明します。 

## 微分計算微分積分学は基本的に、小さな変化のもとで関数がどのように振る舞うかを調べる学問です。これがディープラーニングの核となる理由を理解するために、例を考えてみましょう。 

重みが便宜上 1 つのベクトル $\mathbf{w} = (w_1, \ldots, w_n)$ に連結されたディープニューラルネットワークがあるとします。トレーニングデータセットを考えると、このデータセットではニューラルネットワークが失われたと考えられます。これを $\mathcal{L}(\mathbf{w})$ と記述します。   

この関数は非常に複雑で、このデータセットで指定されたアーキテクチャの可能なすべてのモデルのパフォーマンスをエンコードするため、どの重みのセット $\mathbf{w}$ が損失を最小化するのかを見分けるのはほぼ不可能です。したがって、実際には、重みを*ランダムに*初期化することから始め、損失をできるだけ速く減少させる方向に小さなステップを繰り返し実行することがよくあります。 

そうすると、表面上は簡単ではないという疑問が生まれます。ウェイトをできるだけ早く減らす方向をどうやって見つければよいのでしょうか？これを掘り下げるために、まず重みが 1 つだけのケースを調べてみましょう。つまり、単一の実数値 $x$ に対する $L(\mathbf{w}) = L(x)$ です。  

$x$を取り、$x + \epsilon$に少量変更するとどうなるかを理解してみましょう。具体的にしたいなら、$\epsilon = 0.0000001$のような数字を考えてみてください。何が起こるかを視覚化するために、$[0, 3]$ に対する関数 $f(x) = \sin(x^x)$ の例をグラフ化してみましょう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

この大きなスケールでは、この関数の振る舞いは単純ではありません。ただし、範囲を $[1.75,2.25]$ のように小さくすると、グラフがはるかに単純になることがわかります。

```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

これを極端に考えると、小さなセグメントにズームインすると、動作がはるかに単純になり、直線になります。

```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

これは、単変数微積分の重要な観察です。使い慣れた関数の振る舞いは、十分に小さい範囲の線でモデル化できます。つまり、ほとんどの関数では、関数の $x$ の値を少しずらすと、出力 $f(x)$ も少しずれると予想するのが妥当です。私たちが答える必要がある唯一の質問は、「出力の変化は入力の変化と比較してどれくらい大きいか？半分の大きさですか？2倍大きい？」 

したがって、関数の入力のわずかな変化に対して、関数の出力の変化の比率を考慮することができます。これを正式には次のように書くことができます。 

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

これはすでにコードで遊んでみるには十分です。たとえば、$L(x) = x^{2} + 1701(x-4)^3$ とわかっているとすると、この値がポイント $x = 4$ でどれくらい大きいかが次のようになります。

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

ここで注意すれば、この数値の出力が疑わしいほど$8$に近いことがわかります。実際、$\epsilon$ を下げると、値は次第に $8$ に近づくようになります。したがって、求める値 (入力の変化が出力を変化させる程度) は $x=4$ 点で $8$ であるべきであると正しく結論づけることができます。数学者がこの事実をコード化する方法は 

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

ちょっと歴史的な余談ですが、ニューラルネットワーク研究の最初の数十年間、科学者はこのアルゴリズム（*有限差分法*）を使用して、小さな摂動下で損失関数がどのように変化するかを評価しました。重みを変更して、損失がどのように変化したかを確認します。これは計算上非効率で、1 つの変数の 1 回の変更が損失にどのように影響するかを確認するには、損失関数を 2 回評価する必要があります。わずか数千個のパラメータでもこれを実行しようとすると、データセット全体で数千回のネットワーク評価が必要になります。:cite:`Rumelhart.Hinton.Williams.ea.1988` で導入された*バックプロパゲーションアルゴリズム* が、データセット上のネットワークの 1 回の予測と同じ計算時間で、重みを一緒に変更すると損失がどのように変化するかを計算する方法が提供されたことは、1986 年まで解決されませんでした。 

この例に戻ると、この値 $8$ は $x$ の値によって異なるため、$x$ の関数として定義するのが理にかなっています。より正式には、この値に依存する変化率は*デリバティブ*と呼ばれ、次のように記述されます。 

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

テキストが異なれば、導関数には異なる表記法が使われます。たとえば、以下の表記はすべて同じことを示しています。 

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

ほとんどの著者は一つの記譜法を選んでそれに固執するでしょうが、それでも保証されるわけではありません。これらすべてに精通していることが最善です。このテキスト全体で $\frac{df}{dx}$ という表記を使います。ただし、複素数式の導関数を取りたくない場合は、$\frac{d}{dx}f$ を使って $$\ frac {d} {dx}\ left [x^4+\ cos\ left (\ frac {x^2+1} {2x-1}\ right)\ right] のような式を書きます。。$$ 

$x$ を少し変更したときに関数がどのように変化するかを確認するために、導関数 :eqref:`eq_der_def` の定義をもう一度解き明かすと、直感的に役立つことがよくあります。 

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

最後の方程式は明示的に呼び出す価値があります。これは、関数を取って入力をわずかに変更すると、微分によってスケーリングされたわずかな量だけ出力が変化することを示しています。 

このようにして、微分は、入力の変化によって出力にどのくらいの変化が生じるかを示すスケーリング係数として理解できます。 

## 微積分の法則
:label:`sec_derivative_table`

次に、陽的関数の導関数を計算する方法を理解する作業に移ります。微積分の完全な形式的な扱いは、第一原理からすべてを導き出すでしょう。ここではこの誘惑にふけるのではなく、遭遇する共通のルールを理解します。 

### 一般的な導関数 :numref:`sec_calculus` で見られたように、微分を計算するときには、計算をいくつかのコア関数に減らすための一連のルールを使用することがよくあります。参照しやすいように、ここで繰り返します。 

* **定数の微分** $\frac{d}{dx}c = 0$。
* **一次関数の導関数** $\frac{d}{dx}(ax) = a$.
* **パワールール** $\frac{d}{dx}x^n = nx^{n-1}$。
* **指数の微分** $\frac{d}{dx}e^x = e^x$。
* **対数の微分** $\frac{d}{dx}\log(x) = \frac{1}{x}$。

### 微分規則すべての導関数を別々に計算してテーブルに格納する必要があるとしたら、微分計算はほぼ不可能でしょう。$f(x) = \log\left(1+(x-1)^{10}\right)$の微分を求めるように、上記の導関数を一般化し、より複雑な微分を計算できるのは数学の賜物です。:numref:`sec_calculus` で述べたように、そのための鍵は、関数を取るとどうなるかを体系化し、さまざまな方法で組み合わせることです。最も重要なのは、合計、積、組成。 

* **合計ルール** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$。
* **プロダクトルール** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$。
* **チェーンルール** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$。

これらのルールを理解するために :eqref:`eq_small_change` をどう使うか見てみましょう。合計ルールでは、推論の連鎖に従うことを検討してください。 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

この結果を $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$ という事実と比較すると、$\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ が希望どおりであることがわかります。ここでの直感は、入力$x$、$g$、$h$を変更すると、$\frac{dg}{dx}(x)$と$\frac{dh}{dx}(x)$による出力の変化に共同で寄与します。 

製品はより微妙で、これらの表現の扱い方について新しい観察が必要です。:eqref:`eq_small_change` を使用する前と同じ方法で開始します。 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$

これは上で行われた計算に似ており、私たちの答え（$\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$）は$\epsilon$の隣にありますが、$\epsilon^{2}$というその項の問題があります。$\epsilon^2$ の累乗は $\epsilon^1$ の累乗よりも高いため、これを*高次の項* と呼びます。後のセクションで、これらを追跡したい場合があることがわかりますが、今のところ $\epsilon = 0.0000001$ の場合、$\epsilon^{2}= 0.0000000000001$ は大幅に小さくなります。$\epsilon \rightarrow 0$ を送付しますので、上位の条件は無視しても問題ありません。この付録では一般的な慣習として、「$\approx$」を使用して、2 つの項が高次の項と等しいことを示します。しかし、もっと形式的になりたいのであれば、差分指数を調べることもできます。 

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

$\epsilon \rightarrow 0$を送ると、右辺の項もゼロになるのが分かります。 

最後に、連鎖ルールでは、:eqref:`eq_small_change`を使って前と同じように進めることができ、 

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

2 行目では、関数 $g$ は、入力 ($h(x)$) が微量 $\epsilon \frac{dh}{dx}(x)$ だけシフトされたものと見なしています。 

これらのルールは、基本的に必要な式を計算するための柔軟なツールセットを提供します。例えば、 

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

各行で次のルールが使用されています。 

1. 連鎖則と対数の導関数。
2. 合計ルール。
3. 定数、連鎖則、べき乗則の導関数。
4. 和則、線形関数の導関数、定数の導関数。

この例を実行すると、次の 2 つのことが明らかになります。 

1. 和、積、定数、べき乗、指数、対数を使って書き留めることができる関数は、これらの規則に従うことで微分を機械的に計算することができます。
2. 人間がこれらのルールに従うのは面倒で間違いが起こりやすいです！

ありがたいことに、これら2つの事実が一緒になって前進への道を示唆しています。これは機械化の完璧な候補です！実際、このセクションの後半で再考するバックプロパゲーションは、まさにそれです。 

### 線形近似微分を扱う場合、上記で使った近似を幾何学的に解釈すると便利なことがよくあります。特に、次の式に注目してください。  

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

は、ポイント $(x, f(x))$ を通り、勾配 $\frac{df}{dx}(x)$ をもつ線で $f$ の値に近似します。このように、以下に示すように、微分は関数 $f$ に線形近似を与えると言います。

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### 高次デリバティブ

表面上は奇妙に思えるかもしれないことをやってみましょう。関数 $f$ を取り、導関数 $\frac{df}{dx}$ を計算します。これにより、任意の時点での変化率が $f$ になります。 

しかし、導関数 $\frac{df}{dx}$ は関数そのものと見なすことができるため、$\frac{df}{dx}$ の微分を計算して $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$ を得ることを妨げるものは何もありません。これを $f$ の 2 次導関数と呼びます。この関数は、$f$ の変化率の変化率、つまり変化率がどのように変化しているかを表します。$n$番目の導関数と呼ばれるものを得るために、この微分を何度でも適用することができます。表記法をクリーンに保つために、$n$ 番目の導関数を次のように表します。  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

これが有用な概念である*理由*を理解しようとしましょう。以下では、$f^{(2)}(x)$、$f^{(1)}(x)$、$f(x)$ を可視化します。   

まず、2 次導関数 $f^{(2)}(x)$ が正の定数である場合を考えます。これは、一次導関数の傾きが正であることを意味します。その結果、一次導関数 $f^{(1)}(x)$ は負から始まり、ある点でゼロになり、最後に正になることがあります。これにより、元の関数 $f$ の傾きがわかるため、関数 $f$ 自体が減少し、平坦化されてから増加します。つまり、関数 $f$ は曲線が上がり、:numref:`fig_positive-second` に示されているように 1 つの最小値をもちます。 

![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
:label:`fig_positive-second`

第2に、二次導関数が負の定数であれば、一次導関数が減少していることを意味します。これは、一次導関数が正から始まり、ある点でゼロになり、その後負になる可能性があることを意味します。したがって、関数 $f$ 自体が増加し、平坦化されてから減少します。つまり、関数 $f$ は下向きにカーブし、:numref:`fig_negative-second` に示されているように 1 つの最大値を持ちます。 

![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
:label:`fig_negative-second`

第三に、二次導関数が常にゼロであれば、一次導関数は決して変わらず、一定です！つまり、$f$ は固定レートで増加 (または減少) し、:numref:`fig_zero-second` に示すように $f$ 自体が直線になります。 

![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

要約すると、2 次導関数は関数 $f$ の曲線を表すものとして解釈できます。正の 2 次導関数は上向きの曲線になり、負の 2 次導関数では $f$ は下向きに曲がり、0 秒導関数は $f$ がまったく湾曲しないことを意味します。 

これをさらに一歩進めましょう。$g(x) = ax^{2}+ bx + c$ という関数を考えてみましょう。そうすれば、それを計算できます。 

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

もし $f(x)$ という元の関数を考えているなら、最初の 2 つの導関数を計算し、$a, b$ と $c$ の値を求めることができます。一次導関数が直線で最良の近似を与えることを確認した前のセクションと同様に、この構造は二次関数による最良の近似を提供します。$f(x) = \sin(x)$ についてこれを視覚化してみましょう。

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

次のセクションでは、この考えを*Taylor Series* のアイデアにまで拡張します。  

### テイラーシリーズ

*テイラー級数* は、$x_0$ の点 $x_0$ で最初の $n$ の導関数に値が与えられた場合、関数 $f(x)$ を近似する方法を提供します。このアイデアは、$x_0$ で与えられたすべての導関数に一致する次数 $n$ の多項式を見つけることです。 

前のセクションで $n=2$ のケースを見てきましたが、ちょっとした代数でこれがわかります。 

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

上でわかるように、$2$ の分母は $x^2$ の 2 つの導関数を取るときに得られる $2$ を相殺するためのもので、他の項はすべてゼロです。一次導関数と値自体にも同じ論理が適用されます。 

ロジックをさらに$n=3$に押し上げれば、結論は次のようになります。 

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

ここで $6 = 3\ 倍 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3 $。 

さらに、次数 $n$ の多項式は次の式で得ることができます。  

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

表記法はどこですか  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

実際、$P_n(x)$ は、関数 $f(x)$ に対する最良の $n$ 次多項式近似と見なすことができます。 

上記の近似の誤差については詳しく説明しませんが、無限限界について言及する価値があります。この場合、$\cos(x)$ や $e^{x}$ などの正しく動作する関数 (実解析関数) では、無限の数の項を書き出して、まったく同じ関数を近似することができます。 

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

$f(x) = e^{x}$ を例に挙げてみましょう。$e^{x}$ は独自の派生物なので、$f^{(n)}(x) = e^{x}$ であることがわかっています。したがって、$e^{x}$ は $x_0 = 0$ のテイラー級数を取ることで再構成できます。つまり、 

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

これがコードでどのように機能するのかを見て、テイラー近似の次数を増やすことで目的の関数 $e^x$ に近づく様子を観察してみましょう。

```{.python .input}
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

テイラーシリーズには、主に次の2つの用途があります。 

1. *理論上の応用*: 複雑すぎる関数を理解しようとするとき、テイラー級数を使うことでそれを直接操作できる多項式に変えることができます。

2. *数値アプリケーション*: $e^{x}$ や $\cos(x)$ などの一部の関数は、マシンでは計算が困難です。値のテーブルは固定精度で格納できますが (これはよく行われます)、「$\cos(1)$ の 1000 桁目は何ですか？」のような未解決の質問が残ります。テイラーシリーズはそのような質問に答えるのにしばしば役立ちます。  

## [概要

* 微分は、入力を少しだけ変化させたときに関数がどのように変化するかを表すのに使えます。
* 初等微分は、微分ルールを使用して組み合わせて、任意の複素微分を作成できます。
* 微分は反復して 2 階以上の微分を得ることができます。順序が増えるたびに、関数の動作に関するより詳細な情報が得られます。
* 単一のデータ例の導関数に含まれる情報を使用して、Taylor 級数から得られた多項式によって行儀の良い関数を近似できます。

## 演習

1. $x^3-4x+1$の導関数は何ですか？
2. $\log(\frac{1}{x})$の導関数は何ですか？
3. 正誤問題:$f'(x) = 0$ の場合、$f$ の最大値または最小値は $x$ になりますか。
4. $x\ge0$ の最小値である $f(x) = x\log(x)$ はどこですか ($f$ は $f(0)$ で制限値の $0$ を取ると仮定します)。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:

# 積分計算
:label:`sec_integral_calculus`

差別化は、従来の微積分教育の内容の半分しか占めていません。もう1つの柱である統合は、「この曲線の下にある領域はどこか」というかなり分断された質問のように思えます。一見無関係に見えますが、積分は「微積分の基本定理」と呼ばれるものを介して微分と密接に絡み合っています。 

本書で説明する機械学習のレベルでは、統合について深く理解する必要はありません。ただし、後で遭遇するその他のアプリケーションの基礎を築くための簡単な紹介を提供します。 

## 幾何学的解釈関数$f(x)$があるとします。わかりやすくするために、$f(x)$ は非負である (ゼロより小さい値を取ることはない) と仮定します。私たちが試して理解したいのは、$f(x)$と$x$軸の間に含まれる領域は何ですか？

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

ほとんどの場合、この領域は無限または未定義です ($f(x) = x^{2}$ より下の領域を考慮してください)。したがって、$a$ と $b$ のように、両端の間の領域についてよく話します。

```{.python .input}
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

このエリアを以下の整数記号で表します。 

$$
\mathrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

内部変数はダミー変数で、$\sum$ の sum のインデックスによく似ています。したがって、これは任意の内部値で等価に記述できます。 

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

このような積分を近似する方法を理解しようとする伝統的な方法があります。$a$ と $b$ の間の領域を取り、$N$ の垂直スライスに切り分けることを想像できます。$N$ が大きい場合は、各スライスの面積を長方形で近似し、その面積を足して曲線の下の総面積を求めることができます。これをコードで行う例を見てみましょう。真の値を取得する方法については、後のセクションで説明します。

```{.python .input}
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

問題は、数値的には可能ですが、次のような最も単純な関数に対してのみこのアプローチを分析的に実行できることです。 

$$
\int_a^b x \;dx.
$$

上のコードの例のように、やや複雑なもの 

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

このような直接的な方法で解ける範囲を超えているのです 

代わりに、別のアプローチをとります。領域の概念を直感的に操作し、積分を求めるための主要な計算ツール、*微積分の基本定理*を学びます。これが我々の統合研究の基礎となる。 

## 微積分の基本定理

積分理論をさらに深く掘り下げるために、関数を紹介しましょう。 

$$
F(x) = \int_0^x f(y) dy.
$$

この関数は $x$ の変更方法に応じて $0$ から $x$ までの面積を測定します。これから必要なのはこれだけだということに注目してください 

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

これは、:numref:`fig_area-subtract` に示されているように、遠端までの面積を測定し、その面積から近端点まで減算できるという事実を数学的にエンコードしたものです。 

![Visualizing why we may reduce the problem of computing the area under a curve between two points to computing the area to the left of a point.](../img/sub-area.svg)
:label:`fig_area-subtract`

したがって、$F(x)$が何であるかを理解することで、任意の区間の積分が何であるかを理解することができます。 

そのために、実験を考えてみましょう。微積分学ではよくあることですが、値を少しずらすとどうなるか想像してみましょう。上記のコメントから、私たちはそれを知っています 

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

これは、関数の小さな断片の下の領域によって関数が変化することを示しています。 

これが近似を行うポイントです。このような小さな領域を見ると、この領域は、高さが $f(x)$、底辺の幅が $\epsilon$ の長方形の領域に近いように見えます。実際、$\epsilon \rightarrow 0$ として、この近似がどんどん良くなっていることが分かります。したがって、結論を下すことができます。 

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

しかし、今気づくことが分かります。$F$ の微分を計算するなら、これはまさに私たちが期待するパターンです！したがって、次のようなかなり驚くべき事実が見られます。 

$$
\frac{dF}{dx}(x) = f(x).
$$

これが微積分の基本定理です。$\frac{d}{dx}\int_{-\infty}^x f(y) \; dy = f(x).$ドル:eqlabel:`eq_ftc`ドルと拡張された形式で書くことができます 

それは領域を見つけるという概念（*先験的に*かなり難しい）をとり、それをステートメントデリバティブ（より完全に理解されるもの）に減らします。私たちがしなければならない最後のコメントは、これは$F(x)$が何であるかを正確に教えてくれないということです。実際、$C$の$F(x) + C$は同じ導関数を持っています。これは統合論の現実です。ありがたいことに、定積分を扱う場合、定数は削除されるため、結果とは無関係です。 

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

これは抽象的な意味のないように思えるかもしれませんが、計算積分に関するまったく新しい視点を与えてくれたことを少し理解しましょう。私たちの目標は、もはや面積を回復するために何らかのチョップアンドサムプロセスを実行することではなく、導関数が私たちの持っている関数である関数を見つけるだけで済みます！:numref:`sec_derivative_table` の表を逆にするだけで、かなり難しい積分を数多く挙げられるようになったので、これは驚くべきことです。たとえば、$x^{n}$ の導関数が $nx^{n-1}$ であることがわかっています。したがって、基本定理:eqref:`eq_ftc`を使うと、 

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

同様に、$e^{x}$ の導関数はそれ自体であることがわかっているので、つまり 

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

このようにして、微分積分の発想を活かして、積分理論全体を自由に発展させることができます。すべての統合ルールは、この 1 つの事実から導き出されます。 

## 変数の変更
:label:`integral_example`

微分と同様に、積分の計算をより扱いやすくするための規則がいくつかあります。実際、微分計算のすべての法則 (積則、和則、連鎖則など) には、それぞれ積分計算 (部品による積分、積分の線形性、変数の変化の公式) に対応する規則があります。このセクションでは、リストの中から間違いなく最も重要なもの、つまり変数の変更式について詳しく説明します。 

まず、それ自体が積分である関数があるとします。 

$$
F(x) = \int_0^x f(y) \; dy.
$$

この関数を別の関数と合成して $F(u(x))$ を取得するときに、この関数がどのように見えるかを知りたいとします。連鎖規則によって、私たちは知っています 

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

上記の基本定理 :eqref:`eq_ftc` を使うことで、これを積分に関する記述に変えることができます。これは与える 

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

$F$ 自体が積分であることを思い出すと、左辺は次のように書き換えられます。 

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

同様に、$F$ が積分であることを思い出すと、基本定理 :eqref:`eq_ftc` を使って $\frac{dF}{dx} = f$ を認識することができるので、結論を出すことができます。 

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

これは*変数の変更*の公式です。 

より直感的に導出するために、$x$ と $x+\epsilon$ の間で $f(u(x))$ の積分を取るとどうなるかを考えてみましょう。小さい $\epsilon$ の場合、この積分はおよそ $\epsilon f(u(x))$ (関連する四角形の面積) になります。ここで、これを $u(x)$ から $u(x+\epsilon)$ までの $f(y)$ の積分と比較してみましょう。$u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$ とわかっているので、この長方形の面積はおよそ $\epsilon \frac{du}{dx}(x)f(u(x))$ です。したがって、これら 2 つの長方形の面積を一致させるには、:numref:`fig_rect-transform` に示すように、最初の四角形に $\frac{du}{dx}(x)$ を掛ける必要があります。 

![Visualizing the transformation of a single thin rectangle under the change of variables.](../img/rect-trans.svg)
:label:`fig_rect-transform`

これは私たちにそれを伝えています 

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

これは、単一の小さな四角形に対して表される変数の変化式です。 

$u(x)$ と $f(x)$ を正しく選択すると、非常に複雑な積分の計算が可能になります。たとえば、$f(y) = 1$ と $u(x) = e^{-x^{2}}$ (つまり $\frac{du}{dx}(x) = -2xe^{-x^{2}}$) を選択したとしても、これはたとえば次のことを示しています。 

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

そしてそれを並べ替えることで 

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## サイン規約に関するコメント

熱心な読者は、上記の計算について何か奇妙なことに気付くでしょう。つまり、次のような計算は 

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

負の数を生成できます。地域について考えるとき、負の値を見るのは奇妙なので、慣例が何であるかを掘り下げる価値があります。 

数学者は署名された領域の概念を取ります。これは2つの方法で現れます。まず、ゼロより小さい場合がある関数 $f(x)$ を考えると、面積も負になります。だから例えば 

$$
\int_0^{1} (-1)\;dx = -1.
$$

同様に、左から右ではなく右から左へと進む積分も負の領域とみなされます。 

$$
\int_0^{-1} 1\; dx = -1.
$$

標準面積 (正関数の左から右へ) は常に正です。これを反転して得られるもの ($x$ 軸を反転して負の数の積分を取得したり、$y$ 軸を反転して積分を間違った順序で取得したりするなど) は、負の領域を生成します。そして実際、2回ひっくり返すと、一対の負の符号が得られ、キャンセルされて正の領域になります 

$$
\int_0^{-1} (-1)\;dx =  1.
$$

この議論がおなじみのように聞こえるなら、それはそうです！:numref:`sec_geometry-linear-algebraic-ops` では、行列式がどのように符号付き領域を表すかについて、ほぼ同じ方法で説明しました。 

## 多重積分場合によっては、より高い次元で作業する必要があります。たとえば、$f(x, y)$ のように 2 つの変数の関数があり、$x$ の範囲が $[a, b]$ を超え、$y$ が $[c, d]$ を超える場合に $f$ 未満の体積を知りたいとします。

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

これを次のように書きます 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

この積分を計算するとします。私の主張は、最初に $x$ の積分を繰り返し計算し、次に $y$ の積分にシフトすることでこれを行うことができるということです。 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

これがなぜなのか見てみましょう。 

上の図で、関数を $\epsilon \times \epsilon$ の正方形に分割し、整数座標 $i, j$ でインデックスを付けます。この場合、積分はおよそ 

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

問題を離散化したら、これらの二乗の値を好きな順序で加算することができ、値の変更について心配する必要はありません。これは :numref:`fig_sum-order` で説明されています。特に、次のように言えます。 

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Illustrating how to decompose a sum over many squares as a sum over first the columns (1), then adding the column sums together (2).](../img/sum-order.svg)
:label:`fig_sum-order`

内側の和は、正確には積分の離散化です。 

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

最後に、これら 2 つの式を組み合わせると、次のようになります。 

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

したがって、すべてをまとめると、 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

離散化したら、数値のリストを追加した順序を並べ替えるだけで済むことに注意してください。これは何もないように見えるかもしれませんが、この結果（*Fubiniの定理*と呼ばれる）は必ずしも真実ではありません！機械学習を行うときに遭遇する数学の種類 (連続関数) については問題ありませんが、失敗する例を作成することは可能です (例えば、四角形 $[0,2]\times[0,1]$ 上の関数 $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$)。 

$x$ で最初に積分を行い、次に $y$ で積分を行うという選択は任意であることに注意してください。最初に$y$を実行し、次に$x$を実行するように選択することもできました。 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

多くの場合、ベクトル表記に凝縮して、$U = [a, b]\times [c, d]$ ではこれは 

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## 多重積分における変数の変化 :eqref:`eq_change_var`の単一変数と同様に、高次元の積分内の変数を変更できることは重要なツールです。導出せずに結果をまとめよう。 

積分領域を再パラメータ化する関数が必要です。これを $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$ とすることができます。これは $n$ の実数変数を受け取り、別の $n$ を返す関数です。式をきれいに保つために、$\phi$ は*injective* であると仮定します。つまり、それ自体ではフォールドされません ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$)。 

この場合、次のように言えます。 

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

$D\phi$ は $\phi$ の*ヤコビアン* で、$\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$ の偏微分の行列です。 

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

よく見ると、$\frac{du}{dx}(x)$ という用語を $\left|\det(D\phi(\mathbf{x}))\right|$ に置き換えた点を除いて、これは単一変数チェーンルール :eqref:`eq_change_var` と似ていることがわかります。この用語をどのように解釈できるか見てみましょう。$\frac{du}{dx}(x)$ という用語は、$u$ を適用して $x$ 軸をどれだけ伸ばしたかを示すために存在していたことを思い出してください。高次元での同じプロセスは、$\boldsymbol{\phi}$を適用して、小さな正方形（または小さな*ハイパーキューブ*）の領域（またはボリューム、またはハイパーボリューム）をどれだけ引き伸ばすかを決定することです。$\boldsymbol{\phi}$ が行列の乗算だった場合、行列式はすでにどのように答えを出しているかがわかります。 

いくつかの研究により、微分と勾配をもつ直線または平面で近似できるのと同じ方法で、*ヤコビアン* が行列による点における多変数関数 $\boldsymbol{\phi}$ に対して最良の近似を提供することを示すことができます。したがって、ヤコビアンの行列式は、一次元で特定したスケーリング係数を正確に反映しています。 

これに詳細を記入するには多少の手間がかかりますので、今はっきりしていなくても心配はいりません。後で利用する例を少なくとも1つ見てみましょう。積分を考えてみましょう 

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

この積分を直接使って遊ぶとどこにも行きませんが、変数を変更すれば、大きな進歩を遂げることができます。$\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$（つまり $x = r \cos(\theta)$、$y = r \sin(\theta)$）をさせると、変数式の変更を適用して、これが次の式と同じであることを確認できます。 

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

どこ 

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

したがって、積分は次のようになります。 

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

最終的な等式は、:numref:`integral_example`節で使用したのと同じ計算が続きます。 

:numref:`sec_random_variables` で連続確率変数を研究するときに、この積分を再び満たします。 

## [概要

* 積分理論により、面積や体積に関する質問に答えることができます。
* 微積分の基本定理により、ある点までの面積の微分は積分される関数の値によって与えられるという観測を通じて、微分に関する知識を活用して面積を計算することができます。
* 高次元の積分は、単変数積分を反復することで計算できます。

## 演習 1.$\int_1^2 \frac{1}{x} \;dx$って何ですか？2。変数変化の公式を使用して $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$ を積分します。$\int_{[0,1]^2} xy \;dx\;dy$って何ですか？4。変数の変化の公式を使用して $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ と $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ を計算し、それらが異なっていることを確認します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:

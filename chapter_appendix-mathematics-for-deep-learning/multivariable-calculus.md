# 多変数微積分
:label:`sec_multivariable_calculus`

これで、単一変数の関数の導関数についてかなり深く理解できたので、数十億の重みの損失関数を検討していた元の質問に戻りましょう。 

## 高次元の差別化 :numref:`sec_single_variable_calculus`が教えてくれるのは、この数十億個のウエイトを1つずつ固定したまま変更すれば、何が起こるかわかるということです！これは単一の変数の関数に過ぎないので、次のように書くことができます。 

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

ある変数では導関数を、他の変数を*偏微分* と呼びます。:eqref:`eq_part_der` では導関数に $\frac{\partial}{\partial w_1}$ という表記を使用します。 

さて、これを取り上げて $w_2$ を少しだけ$w_2 + \epsilon_2$に変更してみましょう。 

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

$\epsilon_1\epsilon_2$ は :eqref:`eq_part_der` で見たものと共に、前のセクションで $\epsilon^{2}$ を破棄できるのと同じ方法で破棄できる高次の項であるという考えをもう一度使用しました。このように続けることで、私たちはそれを書くかもしれません 

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

これはめちゃくちゃに見えるかもしれませんが、右側の合計がドット積とまったく同じように見えることに注目することで、これをより身近なものにすることができます。 

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \text{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

それから 

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

ベクトル $\nabla_{\mathbf{w}} L$ を $L$ の*グラデーション* と呼びます。 

方程式 :eqref:`eq_nabla_use` は、ちょっと考えてみる価値があります。それは私たちが一次元で遭遇した形式とまったく同じです。すべてをベクトルとドット積に変換しただけです。これにより、入力に対する摂動が与えられた場合、関数 $L$ がどのように変化するかをおおよそ知ることができます。次のセクションで説明するように、これはグラデーションに含まれる情報を使用して学習する方法を幾何学的に理解する上で重要なツールとなります。 

しかし、最初に、例を挙げてこの近似が機能しているところを見てみましょう。関数を使って作業していると仮定します。 

$$
f(x, y) = \log(e^x + e^y) \text{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

$(0, \log(2))$のような点を見ると、 

$$
f(x, y) = \log(3) \text{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

したがって、$f$ を $(\epsilon_1, \log(2) + \epsilon_2)$ で近似する場合、:eqref:`eq_nabla_use` という特定のインスタンスが必要であることがわかります。 

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

これをコードでテストして、近似がどれほど優れているかを確認できます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## 勾配の幾何学と勾配降下法 :eqref:`eq_nabla_use` の式をもう一度考えてみましょう。 

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

$L$ の損失を最小限に抑えるためにこれを使用するとします。:numref:`sec_autograd`で最初に記述された勾配降下法のアルゴリズムを幾何学的に理解しよう。私たちがやろうとしていることは次のとおりです。 

1. 初期パラメーター $\mathbf{w}$ を無作為に選択することから始めます。
2. $\mathbf{w}$ で $L$ が最も急速に減少する方向 $\mathbf{v}$ を求めます。
3. その方向に小さな一歩を踏み出してください：$\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$。
4. 繰り返し。

正確な方法がわからないのは、2 番目のステップでベクトル $\mathbf{v}$ を計算することだけです。そのような方向を*最も急な降下方向*と呼びます。:numref:`sec_geometry-linear-algebraic-ops` のドット積の幾何学的理解を利用すると、:eqref:`eq_nabla_use` を次のように書き換えることができることがわかります。 

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

ここでは、便宜上、長さを 1 にするとおり、$\mathbf{v}$ と $\nabla_{\mathbf{w}} L(\mathbf{w})$ の間の角度には $\theta$ を使用していることに注意してください。$L$ の減少する方向をできるだけ早く見つけたい場合は、この式をできるだけ負の値にします。ピックする方向がこの方程式に入る唯一の方法は $\cos(\theta)$ です。したがって、この余弦をできる限り負の値にします。ここで、余弦の形状を思い出すと、$\cos(\theta) = -1$、またはグラデーションと選択した方向の間の角度を $\pi$ ラジアン、つまり $180$ 度にすることで、これを可能な限り負にすることができます。これを実現する唯一の方法は、正反対の方向に向かうことです。$\mathbf{v}$ をピックして $\nabla_{\mathbf{w}} L(\mathbf{w})$ と正反対の方向を指すようにしてください。 

これにより、機械学習における最も重要な数学的概念の1つである、$-\nabla_{\mathbf{w}}L(\mathbf{w})$の方向で最も急勾配な点の方向がわかります。したがって、私たちの非公式アルゴリズムは次のように書き換えることができます。 

1. 初期パラメーター $\mathbf{w}$ を無作為に選択することから始めます。
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$ を計算してください。
3. その方向とは逆の小さな一歩を踏み出して、$\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$。
4. 繰り返し。

この基本的なアルゴリズムは、多くの研究者によって多くの方法で修正され、適応されてきましたが、コアコンセプトはすべての研究者に共通しています。勾配を使用して、損失をできるだけ速く減少させる方向を見つけ、その方向に一歩踏み出すようにパラメーターを更新します。 

## 数学的最適化に関する注意本書では、ディープラーニングの設定で遭遇するすべての関数が複雑すぎて明示的に最小化できないという実際的な理由から、数値最適化手法に真っ向から注目しています。 

しかし、上で得た幾何学的な理解から、関数を直接最適化することについて何を教えてくれるかを考えることは有益な演習です。 

関数 $L(\mathbf{x})$ を最小化する $\mathbf{x}_0$ の値を見つけたいとします。さらに、誰かが私たちに価値を与えて、それが$L$を最小にする値であると私たちに告げたとしましょう。彼らの答えがもっともらしいかどうかを確認するために確認できることはありますか？ 

:eqref:`eq_nabla_use` をもう一度考えてみましょう:$$ L (\ mathbf {x} _0 +\ ボールドシンボル {\ epsilon})\ おおよそ L (\ mathbf {x} _0) +\ ボールドシンボル {\ epsilon}\ cdot\ nabla_ {\ mathbf {x}} L (\ mathbf {x} _0)。$$ 

勾配がゼロでない場合、$-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ の方向に一歩踏み出して $L$ の小さい値を見つけることができます。したがって、私たちが本当に最低限であれば、これは当てはまりません！$\mathbf{x}_0$が最小であれば、$\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$と結論づけることができます。$\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ のポイントをクリティカルポイント*と呼びます。 

まれな設定では、グラデーションがゼロであるすべてのポイントを明示的に見つけ、最小値を持つポイントを見つけることができるので、これは良いことです。 

具体的な例として、$$ f (x) = 3x^4-4x^3 -12x^2 という関数を考えてみましょう。$$ 

この関数は導関数 $$\ フラック {df} {dx} = 12x^3-12x^2 -24x = 12x (x-2) (x+1)。$$ 

最小値の位置は $x = -1, 0, 2$ のみで、関数はそれぞれ $-5,0, -32$ の値を取ります。したがって、$x = 2$ のときに関数を最小化すると結論付けることができます。簡単なプロットでこれが確認されます。

```{.python .input}
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

これは、理論的にも数値的にも作業する際に知っておくべき重要な事実を強調しています。関数を最小化 (または最大化) できる唯一の点は勾配がゼロになりますが、勾配ゼロを持つすべての点が真の*グローバル* 最小 (または最大) であるとは限りません。 

## 多変量連鎖規則 4つの変数 ($w, x, y$, $z$) の関数があるとします。この関数は多くの項を構成することで作れます: 

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

このような連鎖の方程式はニューラルネットワークを扱うときによくあるので、そのような関数の勾配を計算する方法を理解することが鍵となります。:numref:`fig_chain-1` では、どの変数が相互に直接関係しているかを見ると、この接続の視覚的なヒントがわかります。 

![The function relations above where nodes represent values and edges show functional dependence.](../img/chain-net1.svg)
:label:`fig_chain-1`

:eqref:`eq_multi_func_def`のすべてを作曲して書き出すのを止めるものは何もありません 

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

その後、単変数微分を使って導関数を取ることができますが、そうするとすぐに項がいっぱいになり、その多くは繰り返しです！実際、次のようなことが分かります。 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

$\frac{\partial f}{\partial x}$ も計算すると、繰り返し項が多く、2 つの導関数間に多数の*共有* 反復項が含まれる同様の方程式が再び得られます。これは膨大な量の無駄な作業を表しています。この方法で微分を計算する必要があるとしたら、ディープラーニング革命全体が始まる前に行き詰まっていたでしょう。 

問題を解き明かそう。$a$ を変更すると $f$ がどのように変化するかを理解することから始めます。基本的に $w, x, y$ と $z$ はすべて存在しないと仮定します。グラデーションを初めて使用したときと同じように、推論します。$a$を取り、それに少量の$\epsilon$を加えましょう。 

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

最初の行は偏微分の定義に従い、2 行目は勾配の定義に従います。$\frac{\partial f}{\partial u}(u(a, b), v(a, b))$という式のように、すべての導関数を評価する場所を正確に追跡するのは表記上面倒なので、これをより記憶に残るものと略すことがよくあります。 

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

プロセスの意味を考えると便利です。$a$ の変更に伴って $f(u(a, b), v(a, b))$ という形式の関数がどのように値を変更するかを理解しようとしています。これが発生する可能性がある経路は2つあります。$a \rightarrow u \rightarrow f$と$a \rightarrow v \rightarrow f$の経路です。連鎖規則 ($\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ と $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$) によって、これらの寄与分をそれぞれ計算し、加算することができます。 

:numref:`fig_chain-2` に示すように、右側の関数が左側で接続されている機能に依存する別の機能ネットワークがあるとします。 

![Another more subtle example of the chain rule.](../img/chain-net2.svg)
:label:`fig_chain-2`

$\frac{\partial f}{\partial y}$ のようなものを計算するには、$y$ から $f$ までのすべてのパス (この場合は $3$) を合計する必要があります。 

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

このように連鎖律を理解することは、勾配がネットワークをどのように流れるか、また、LSTM (:numref:`sec_lstm`) や残差層 (:numref:`sec_resnet`) などのさまざまなアーキテクチャの選択が勾配フローを制御することで学習プロセスの形成に役立つ理由を理解しようとするときに大きな効果をもたらします。 

## バックプロパゲーションアルゴリズム

前のセクションの :eqref:`eq_multi_func_def` の例に戻りましょう。 

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

$\frac{\partial f}{\partial w}$と計算したい場合は、多変量連鎖則を適用して以下を確認できます。 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

この分解を使って $\frac{\partial f}{\partial w}$ を計算してみましょう。ここで必要なのは、さまざまなシングルステップパーシャルだけであることに注意してください。 

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

これをコードに書き出すと、これはかなり扱いやすい表現になります。

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

ただし、これでも $\frac{\partial f}{\partial x}$ のような計算は簡単にはならないことに注意してください。その理由は、チェーンルールを適用する*方法*です。上記で何をしたかを見ると、可能な場合は常に $\partial w$ を分母に収めました。このようにして、$w$ が 1 つおきの変数をどのように変更するかを確認するために、チェーンルールを適用することにしました。それが私たちが望んでいたことなら、これは良い考えでしょう。しかし、ディープラーニングのモチベーションを思い出してください。すべてのパラメータが*損失*をどのように変化させるかを見たいと思っています。本質的には、可能な限り $\partial f$ を分子に保持するチェーンルールを適用したいと考えています。 

より明確に言うと、以下のように書けることに注意してください。 

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

チェーンルールをこのように適用すると、$\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{and} \; \frac{\partial f}{\partial w}$ が明示的に計算されることに注意してください。方程式も含めることを妨げるものは何もありません。 

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

ネットワーク全体で*any* ノードを変更したときに $f$ がどのように変化するかを追跡します。それを実装しましょう。

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

上記の最初のコードスニペットで行ったように、入力から出力に向かうのではなく、$f$ から入力に向かって導関数を計算して戻すという事実が、このアルゴリズムに*backpropagation* という名前を付けています。次の 2 つのステップがあることに注意してください。
1. 関数の値と、前から後ろへのシングルステップパーシャルを計算します。上記では行っていませんが、これを組み合わせて 1 つの *forward パス* にすることができます。
2. $f$ の勾配を背面から前面へ計算します。これを*バックワードパス*と呼んでいます。

これはまさに、ネットワーク内のすべての重みに関する損失の勾配を 1 回のパスで計算できるように、すべての深層学習アルゴリズムが実装するものです。私たちがそのような分解をしているのは驚くべき事実です。 

これをカプセル化する方法を確認するために、この例をざっと見てみましょう。

```{.python .input}
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

上記で行ったことはすべて、`f.backwards()` を呼び出すことで自動的に実行できます。 

## ヘッシアン (Hessians) 単変数微積分と同様に、勾配を単独で使うよりも関数へのよりよい近似を得る方法を理解するために、高次の微分を考慮すると便利です。 

複数の変数の関数の高階微分を扱うときに直面する直接的な問題が1つあります。それは、多数の変数が存在することです。$n$ 変数の関数 $f(x_1, \ldots, x_n)$ がある場合、$i$ と $j$ の任意の選択に対して $n^{2}$ 多くの二次導関数を取ることができます。 

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

これは従来、*Hessian* と呼ばれる行列に組み立てられます。 

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

この行列のすべてのエントリが独立しているわけではありません。実際に、*混合パーシャル* (複数の変数に関する偏微分) が両方とも存在し、連続している限り、$i$ と $j$、 

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

これに続いて、最初に $x_i$ の方向に関数を摂動させ、次に $x_j$ で摂動し、その結果を $x_j$、次に $x_i$ に摂動するとどうなるかを比較します。これらの次数が両方とも同じ最終変化につながるという知識をもとに、$f$の出力です 

単一変数の場合と同様に、これらの微分を使用して、関数が点付近でどのように動作するかをよりよく理解できます。特に、単一の変数で見たように、点 $\mathbf{x}_0$ の近くで最適近似の 2 次を求めるために使用できます。 

例を見てみましょう。$f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$ と仮定します。これは、2 つの変数の 2 次関数の一般的な形式です。関数の値、その勾配、ヘッシアン :eqref:`eq_hess_def` をすべてゼロ点で見ると、次のようになります。 

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

こう言うことで元の多項式を取り戻すことができます 

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

一般に、$\mathbf{x}_0$の任意の点でこの展開を計算すると、 

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

これはあらゆる次元の入力に対して機能し、ある点でどの関数に対しても最適な近似二次関数を提供します。例を挙げて、関数をプロットしてみましょう。 

$$
f(x, y) = xe^{-x^2-y^2}.
$$

勾配とヘッシアンは$$\ nabla f (x, y) = e^ {-x^2-y^2}\ begin {pmatrix} 1-2x^2\\ -2xy\ 終了 {pmatrix}\;\ text {と}\;\ mathbf {H} f (x, y) = e^ {-x^2-y^2}\ begin {pマトリックス} 4x^3-6x & 4x^2y-2y\\ 4x^2y-2y &4xy^2-2x\ end {pmatrix}。$$ 

したがって、少し代数を使うと、$[-1,0]^\top$ の近似二次関数は次のようになります。 

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x, y, w, **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

これは :numref:`sec_gd` で説明したニュートンアルゴリズムの基礎を形成します。ニュートンアルゴリズムでは、数値最適化を繰り返し実行して、最適近似の 2 次を求めて、その 2 次関数を正確に最小化します。 

## 小さな行列微積分行列に関わる関数の微分は特に素晴らしいと判明しました。このセクションは表記法的に重くなることがあるため、最初の読みでは読み飛ばしてもかまいませんが、一般的な行列演算を含む関数の導関数が、特に深層学習アプリケーションにとって中心的な行列演算がどれほど重要であるかを考えると、当初の予想よりもはるかにクリーンであることが多いことを知っておくと便利です。 

例から始めましょう。固定列ベクトル $\boldsymbol{\beta}$ があり、積関数 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ を取り、$\mathbf{x}$ を変更すると内積がどのように変化するかを理解するとします。 

ML で行列導関数を扱うときに便利な表記法は*分母レイアウト行列微分* と呼ばれ、微分の分母にあるベクトル、行列、テンソルの形に偏導関数を組み立てます。この場合は、次のように書きます。 

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

ここで、列ベクトル $\mathbf{x}$ の形状を一致させました。 

関数をコンポーネントに書き出すと、これは次のようになります。 

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

ここで $\beta_1$ という偏微分をとると、最初の項を除いてすべてがゼロであることに注目してください。$x_1$ に $\beta_1$ を掛けたものです。 

$$
\frac{df}{dx_1} = \beta_1,
$$

より一般的には 

$$
\frac{df}{dx_i} = \beta_i.
$$

これをマトリックスに再構成して確認することができます 

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

これは、このセクションでよく取り上げる、行列計算に関するいくつかの要因を示しています。 

* まず、計算はかなり複雑になります。
* 第2に、最終結果は中間プロセスよりもはるかにクリーンで、常に単一変数の場合と似ています。この場合、$\frac{d}{dx}(bx) = b$ と $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ はどちらも類似していることに注意してください。
* 第三に、移調はどこからともなく見えることがよくあります。この主な理由は、分母の形状を一致させるという慣習です。したがって、行列を乗算するときは、元の項の形状に合わせて転置する必要があります。

直感を築き続けるために、もう少し難しい計算を試してみましょう。列ベクトル $\mathbf{x}$ と正方行列 $A$ があり、計算したいとします。 

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

楽譜法を操作しやすくするために、Einstein 記譜法を使用してこの問題について考えてみましょう。この場合、関数は次のように記述できます。 

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

導関数を計算するには、$k$ ごとに、次の値が何であるかを理解する必要があります。 

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

積のルールでは、これは 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

$\frac{dx_i}{dx_k}$ のような用語では、$i=k$ の場合はこれが 1 であり、それ以外の場合は 0 であることがわかりにくいです。つまり、$i$ と $k$ が異なるすべての項がこの和から消滅するので、最初の和に残っているのは $i=k$ の項だけです。$j=k$が必要な第2学期についても同じ推論が成り立ちます。これは与える 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

これで、アインシュタイン表記法のインデックスの名前は任意です。$i$ と $j$ が異なるという事実は、この時点ではこの計算には重要ではないため、インデックスを再作成して、両方とも $i$ を使用して確認することができます。 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

さて、ここから先に進むためにいくつかの練習が必要になり始めます。この結果を行列演算で特定してみましょう。$a_{ki} + a_{ik}$ は $\mathbf{A} + \mathbf{A}^\top$ の $k, i$ 番目の成分です。これは与える 

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

同様に、この項は行列 $\mathbf{A} + \mathbf{A}^\top$ とベクトル $\mathbf{x}$ の積になったので、次のことがわかります。 

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

したがって、:eqref:`eq_mat_goal_1` からの目的の導関数の $k$ 番目のエントリは、右側のベクトルの $k$ 番目のエントリに過ぎず、したがって 2 つは同じであることがわかります。したがって、利回り 

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

これには前回の作業よりもはるかに多くの作業が必要でしたが、最終的な結果はわずかです。それ以上に、従来の単変数微分については次のような計算を考えてみましょう。 

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

同等です $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$。ここでも、単一の変数 result に似た結果が得られますが、転置が投げ込まれています。 

この時点で、パターンはかなり疑わしいように見えるはずなので、その理由を理解してみましょう。このような行列導関数を取るとき、最初に得られる式が別の行列式、つまり積と行列の和とそれらの転置の観点から書くことができる式であると仮定しましょう。そのような式が存在する場合、すべての行列に対して真である必要があります。特に、$1 \times 1$ 行列に当てはまる必要があります。その場合、行列の積は数値の積に過ぎず、行列の合計は和であり、転置は何もしません。言い換えれば、*得られる式はどれも、単一の変数式に一致する必要があります。つまり、ある程度の練習をすれば、関連する単一変数式がどのように見えるかを知るだけで、行列微分を推測できることがよくあります。 

これを試してみよう。$\mathbf{X}$ が $n \times m$ 行列で、$\mathbf{U}$ が $n \times r$、$\mathbf{V}$ が $r \times m$ であるとします。計算してみよう 

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

この計算は、行列因数分解と呼ばれる領域で重要です。しかし、私たちにとっては、計算するのは微分にすぎません。$1\times1$ 行列でこれがどうなるかをイメージしてみよう。その場合、次の式が得られます。 

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

ここで、微分はかなり標準的です。これを行列式に変換し直そうとすると、 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

しかし、これを見てみると、あまりうまくいきません。$\mathbf{X}$ は $n \times m$ であり、$\mathbf{U}\mathbf{V}$ であるため、マトリックス $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ は $n \times m$ であることを思い出してください。一方、$\mathbf{U}$ は $n \times r$ であり、$n \times m$ と $n \times r$ の行列は次元が一致しないため乗算できません。 

$r \times m$ である $\mathbf{V}$ と同じ形状の $\frac{d}{d\mathbf{V}}$ を取得したいと考えています。したがって、$r \times m$を得るには、$n \times m$行列と$n \times r$行列をとり、それらを（おそらく転置して）掛け合わせる必要があります。$U^\top$ に $(\mathbf{X} - \mathbf{U}\mathbf{V})$ を掛けることでこれを行うことができます。したがって、:eqref:`eq_mat_goal_2` の解は次のようになります。 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

これが機能することを示すために、詳細な計算を提供しないのは怠慢です。この経験則が機能するとすでに信じている場合は、この派生をスキップしてください。計算するには 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

$a$、$b$ごとに見つけなければなりません 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

$\mathbf{X}$ と $\mathbf{U}$ のすべてのエントリは $\frac{d}{dv_{ab}}$ に関する限り定数であることを想起して、微分を和の中に押し込み、連鎖則を二乗に適用して、 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

前述の導出と同様に、$k=a$ と $j=b$ の場合、$\frac{dv_{kj}}{dv_{ab}}$ はゼロ以外であることに注意してください。これらの条件のいずれかが成立しない場合、合計の項はゼロになり、自由に破棄することができます。私たちはそれを見る 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

ここで重要な微妙な点は、$k=a$ が内部和の内側では発生しないという要件です。これは $k$ がダミー変数であり、内部項内で合計しているためです。表記法的にクリーンな例として、その理由を考えてみましょう。 

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

この時点から、和の成分を特定し始めるかもしれません。まず、 

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

つまり、和の内側にある式全体は次のようになります。 

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

これは、派生物を次のように書くことができることを意味します。 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

これを行列の $a, b$ 要素のようにして、前の例と同様の手法を使用して行列式を得ることができます。つまり、$u_{ia}$ のインデックスの順序を交換する必要があります。$u_{ia} = [\mathbf{U}^\top]_{ai}$に気付いたら、次のように書くことができます 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

これはマトリックス積であるため、次のように結論付けることができます。 

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

こうして解を:eqref:`eq_mat_goal_2`に書くことができます 

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

これは上で推測した解に一致します！ 

この時点で、「私が学んだすべての微積分則の行列バージョンを書き留めておけないのはなぜですか？これはまだ機械的なものであることは明らかです。なぜ私たちはそれを乗り越えないのですか！」そして確かにそのような規則があり、:cite:`Petersen.Pedersen.ea.2008`は優れた要約を提供します。ただし、単一の値と比較して行列演算を組み合わせる方法は多いため、単一変数よりも行列微分規則の方がはるかに多くなります。多くの場合、インデックスを操作するか、適切な場合は自動微分に任せるのがベストです。 

## [概要

* 高次元では、一次元で微分と同じ目的を果たす勾配を定義できます。これにより、入力に任意の小さな変更を加えたときに、多変数関数がどのように変化するかを確認できます。
* バックプロパゲーションアルゴリズムは、多変数連鎖則を編成して、多くの偏微分を効率的に計算できるようにする方法であると考えることができます。
* 行列計算により、行列式の導関数を簡潔に書くことができます。

## 演習 1.列ベクトル $\boldsymbol{\beta}$ を指定して、$f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ と $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$ の両方の導関数を計算します。なぜ同じ答えが出るのですか？2。$\mathbf{v}$ を $n$ 次元のベクトルとします。$\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$って何ですか？3。$L(x, y) = \log(e^x + e^y)$ にしましょう。勾配を計算します。グラデーションの成分の合計はどれくらいですか？4。$f(x, y) = x^2y + xy^2$ にしましょう。唯一の臨界点が $(0,0)$ であることを示してください。$f(x, x)$ を考慮して、$(0,0)$ が最大値か最小値か、どちらでもないかを判断します。関数 $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$ を最小化するとします。$g$ と $h$ の観点から $\nabla f = 0$ の条件を幾何学的に解釈するにはどうすればよいのでしょうか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab:

# 最尤法
:label:`sec_maximum_likelihood`

機械学習で最もよく見られる考え方の1つは、最尤法の視点です。これは、パラメータが不明な確率モデルを扱う場合、データの確率を最も高くするパラメータが最も可能性が高いという概念です。 

## 最尤法則

これにはベイズ解釈があり、考えると役に立ちます。パラメーター $\boldsymbol{\theta}$ とデータ例 $X$ のコレクションをもつモデルがあるとします。具体的には、$\boldsymbol{\theta}$はコインが反転したときに頭が上がる確率を表す単一の値であり、$X$は一連の独立したコインフリップであると想像できます。この例は後で詳しく見ていきます。 

モデルのパラメータについて最も可能性の高い値を見つけたい場合は、 

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

ベイズの法則では、これは 

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

パラメータに依存しないデータ生成確率である $P(X)$ という式は $\boldsymbol{\theta}$ にまったく依存しないため、$\boldsymbol{\theta}$ という最良の選択を変更することなく削除できます。同様に、どのパラメータセットが他のどのパラメータよりも優れているかについて事前に仮定していないと仮定することができるので、$P(\boldsymbol{\theta})$ も theta に依存しないと宣言することができます！これは、例えば、コインフリッピングの例では、$[0,1]$の確率が、公平であるかどうかを事前に信じることなく（しばしば*非情報事前*と呼ばれる）、任意の値になる可能性があるという意味があります。したがって、ベイズの法則を適用すると、$\boldsymbol{\theta}$ の最善の選択が $\boldsymbol{\theta}$ の最尤推定値であることがわかります。 

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

一般的な用語として、パラメーター ($P(X \mid \boldsymbol{\theta})$) が与えられたデータの確率を*尤度*と呼びます。 

### 具体的な例

具体的な例で、これがどのように機能するかを見てみましょう。コインフリップがヘッドになる確率を表す単一のパラメータ $\theta$ があるとします。その場合、尾を得る確率は $1-\theta$ です。したがって、観測データ $X$ が $n_H$ 頭、尾が $n_T$ の系列である場合、独立確率が乗算されるという事実を利用して、次のことがわかります。  

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

$13$コインをひっくり返して、$n_H = 9$と$n_T = 4$をもつ「HHHTHTTHTHHHHHT」というシーケンスを取得すると、これが 

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

この例の良い点の1つは、答えがわかっていることです。確かに、私たちが口頭で言ったら、「私は13枚のコインをひっくり返し、9枚が頭を上げました。コインが私たちの頭に来る確率についての最善の推測は何ですか？、"誰もが$9/13$を正しく推測するだろう。この最尤法がもたらすのは、非常に複雑な状況に一般化される方法で、第1のプリンシパルからその数を得る方法です。 

この例では、$P(X \mid \theta)$ のプロットは次のようになります。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

これは、予想される $9/13 \approx 0.7\ldots$ に近いところで最大値になります。それが正確にそこにあるかどうかを確認するには、微積分学に目を向けることができます。最大値では、関数の勾配は平坦であることに注意してください。したがって、導関数がゼロの場合に $\theta$ の値を求め、確率が最も高い値を求めることで、最尤推定値 :eqref:`eq_max_like` を求めることができます。私たちは以下を計算します。 

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

これには、$0$、$1$、$9/13$ の 3 つのソリューションがあります。最初の2つは、確率$0$をシーケンスに割り当てるため、明らかに最大値ではなく最小値です。最終的な値は、シーケンスにゼロ確率を割り当てないため、最尤推定 $\hat \theta = 9/13$ でなければなりません。 

## 数値最適化と負の対数尤度

前の例はいいですが、何十億ものパラメータとデータ例があるとしたらどうでしょうか？ 

まず、すべてのデータ例が独立していると仮定すると、尤度は多数の確率の積であるため、尤度自体を実質的に考えることができなくなることに注意してください。実際、各確率は $[0,1]$ で、通常は $1/2$ 程度の値であり、$(1/2)^{1000000000}$ の積は機械精度をはるかに下回ります。それを直接扱うことはできません。   

ただし、対数は積を合計に変換することを思い出してください。この場合、  

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

この数値は、単精度の $32$ ビット浮動小数点数にも完全に収まります。したがって、*対数尤度*を考慮する必要があります。 

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

関数 $x \mapsto \log(x)$ は増加しているので、尤度の最大化は対数尤度を最大化することと同じです。実際に :numref:`sec_naive_bayes` では、単純ベイズ分類器の特定の例を扱うときに、この推論が適用されることがわかります。 

多くの場合、損失を最小化したい損失関数を使用します。*負の対数尤度* である $-\log(P(X \mid \boldsymbol{\theta}))$ を取ることで、最尤法を損失の最小化に変えることができます。 

これを説明するために、以前のコインフリッピング問題を考えて、閉じた形の解を知らないふりをしてください。それを計算するかもしれない 

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

これはコードに書き込むことができ、何十億ものコインフリップに対しても自由に最適化できます。

```{.python .input}
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

人々が負の対数尤度を好む理由は、数値的な利便性だけではありません。それが好ましい理由は他にもいくつかあります。 

対数尤度を考慮する2つ目の理由は、微積分則の適用が単純化されていることです。前述のとおり、独立性の仮定により、機械学習で遭遇する確率のほとんどは個々の確率の積です。 

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

つまり、積法則を直接適用して微分を計算すると、次のようになります。 

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

これには $n(n-1)$ の乗算と $(n-1)$ の加算が必要なため、入力の 2 次時間に比例します。グループ化の用語を十分に賢くすれば、これは線形時間になりますが、ある程度の考慮が必要です。負の対数尤度については、代わりに 

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

それはそれから与える 

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

これには $n$ の除算と $n-1$ の合計しか必要ないため、入力では線形時間になります。 

負の対数尤度を考慮する3つ目の理由は、情報理論との関係です。これについては、:numref:`sec_information_theory`で詳しく説明します。これは厳密な数学的理論であり、確率変数の情報の程度またはランダム性を測定する方法を提供します。その分野での重要な研究対象はエントロピーであり、  

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

ソースのランダム性を測定します。これは平均の $-\log$ 確率に過ぎないことに注意してください。したがって、負の対数尤度を取り、データ例の数で割ると、クロスエントロピーと呼ばれるエントロピーの相対値が得られます。この理論的な解釈だけでも、モデルのパフォーマンスを測定する方法として、データセットに対する平均負の対数尤度を報告する動機付けとなるほど説得力があります。 

## 連続変数の最尤法

ここまで行ってきたことはすべて、離散確率変数を扱うことを前提としていますが、連続確率変数を扱う場合はどうでしょうか？ 

簡単にまとめると、確率のすべてのインスタンスを確率密度に置き換える以外は何も変化しないということです。密度を小文字の $p$ で書くことを思い出してください。これは、例えば今こう言うことを意味します 

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

問題は、「なぜこれでいいの？」結局のところ、密度を導入した理由は、特定の結果を得る確率がゼロだったからであり、したがって、任意のパラメータセットに対してデータを生成する確率がゼロではないからです。 

実際、これは事実であり、なぜ私たちが密度にシフトできるのかを理解することは、イプシロンに何が起こるかを追跡する上での練習です。 

まず、目標を再定義しましょう。連続確率変数について、正確な値を得る確率を計算するのではなく、$\epsilon$の範囲内で一致させることを考えたとします。わかりやすくするために、データは同一分布の確率変数 $X_1, \ldots, X_N$ の反復観測値 $x_1, \ldots, x_N$ であると仮定します。前に見てきたように、これは次のように書くことができます。 

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

したがって、これの負の対数をとると、 

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

この式を調べると、$\epsilon$ が出現する場所は加法定数 $-N\log(\epsilon)$ だけです。これは$\boldsymbol{\theta}$のパラメータにはまったく依存しないため、$\boldsymbol{\theta}$の最適な選択は$\epsilon$の選択には依存しません。4桁または400桁が必要な場合、$\boldsymbol{\theta}$の最良の選択は変わりません。したがって、イプシロンを自由に削除して、最適化したいものが 

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

したがって、確率を確率密度に置き換えることで、最尤の視点が連続確率変数でも離散確率変数と同じように簡単に操作できることがわかります。 

## まとめ * 最尤法により、特定のデータセットに最も適合するモデルは、最も高い確率でデータを生成するモデルであることがわかります。* 多くの場合、数値の安定性、積の和への変換（および* 離散的な設定では動機付けが最も簡単ですが、データポイントに割り当てられた確率密度を最大化することで、連続設定に自由に一般化することもできます。 

## 演習 1.確率変数がある値 $\alpha$ に対して密度が $\frac{1}{\alpha}e^{-\alpha x}$ であることがわかっているとします。$3$ という確率変数から 1 つの観測値が得られます。$\alpha$ の最尤推定値はどれくらいですか？2。平均が不明で分散 $1$ をもつガウス分布から抽出された標本 $\{x_i\}_{i=1}^N$ のデータセットがあるとします。平均の最尤推定値はどれくらいですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:

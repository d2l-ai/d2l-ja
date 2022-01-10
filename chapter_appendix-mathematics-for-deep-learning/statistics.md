# 統計学
:label:`sec_statistics`

ディープラーニングのトッププラクティショナーであるためには、最先端かつ高精度のモデルをトレーニングする能力が不可欠であることは間違いありません。ただし、改善が有意なのはいつか、またはトレーニングプロセスのランダムな変動の結果のみであるのかは不明な場合がよくあります。推定値の不確実性を論じるためには、いくつかの統計学を学ばなければならない。 

*統計*の最初の参照は、$9^{\mathrm{th}}$世紀のアラブの学者Al-Kindiが統計と頻度分析を使用して暗号化されたメッセージを解読する方法について詳細に説明したことにさかのぼることができます。800年後、研究者が人口統計学的および経済データの収集と分析に焦点を合わせた1700年代にドイツから現代の統計が生まれました。今日、統計学は、データの収集、処理、分析、解釈、視覚化に関係する科学的な主題です。さらに、統計学の核となる理論は、学界、産学、政府の研究において広く用いられてきた。 

具体的には、統計を*記述統計*と*統計的推論*に分けることができます。前者は、*サンプル*と呼ばれる観測データの集合の特徴を要約し、図解することに重点を置いています。サンプルは*母集団*から抽出され、実験の対象となる類似の個人、項目、または事象の合計セットを示します。記述統計とは対照的に、*統計的推論* は、標本分布がある程度母集団分布を再現できるという仮定に基づいて、与えられた*標本*から母集団の特性をさらに推定します。 

「機械学習と統計の本質的な違いは何ですか？」基本的に、統計学は推論問題に焦点を合わせています。この種の問題には、因果推論などの変数間の関係のモデル化や、A/B 検定などのモデルパラメーターの統計的有意性の検定が含まれます。一方、機械学習では、各パラメーターの機能を明示的にプログラミングして理解することなく、正確な予測を行うことに重点が置かれています。 

ここでは、推定量の評価と比較、仮説検定の実施、信頼区間の構築の3種類の統計推論方法を紹介します。これらのメソッドは、特定の母集団の特性、つまり真のパラメーター $\theta$ を推測するのに役立ちます。簡潔にするために、与えられた母集団の真のパラメーター $\theta$ はスカラー値であると仮定します。$\theta$ がベクトルまたはテンソルの場合まで拡張するのは簡単なので、ここでは省略します。 

## 推定量の評価と比較

統計学では、*estimator* は与えられた標本の関数で、真のパラメーター $\theta$ を推定するために使用されます。サンプル {$x_1, x_2, \ldots, x_n$} を観察した後に $\theta$ の推定値として $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ と書きます。 

:numref:`sec_maximum_likelihood`節で、推定量の簡単な例を見てきました。ベルヌーイ確率変数から多数の標本がある場合、確率変数が1である確率の最尤推定は、観測された標本の数を数え、標本の総数で割ることによって求めることができます。同様に、ある演習では、サンプル数が与えられた場合のガウス平均の最尤推定値は、すべてのサンプルの平均値によって与えられることを示すように求められました。これらの推定量ではパラメーターの真の値が得られることはほとんどありませんが、サンプル数が多い場合は推定値が近くなるのが理想的です。 

例として、平均が0、分散が1のガウス確率変数の真の密度と、そのガウス分布からのコレクションサンプルを以下に示します。すべての点が見えるようになり、元の密度との関係がより明確になるように $y$ 座標を作成しました。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

パラメーター $\hat{\theta}_n$ の推定量を計算するには、さまざまな方法があります。このセクションでは、推定量を評価および比較するための 3 つの一般的な方法 (平均二乗誤差、標準偏差、統計的偏差) を紹介します。 

### 平均二乗誤差

推定量の評価に使用される最も単純なメトリックは、推定量の*平均二乗誤差 (MSE) * (または $l_2$ 損失) であり、次のように定義できます。 

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

これにより、真の値からの平均二乗偏差を定量化できます。MSE は常に非負です。:numref:`sec_linear_regression` を読めば、最も一般的に使用される回帰損失関数として認識されるでしょう。推定量を評価するための尺度として、その値がゼロに近いほど、推定器は真のパラメーター $\theta$ に近くなります。 

### 統計的偏り

MSE は自然なメトリックを提供しますが、MSE が大きくなる可能性のある複数の異なる現象を簡単に想像できます。基本的に重要な2つは、データセットのランダム性による推定量の変動と、推定手順による推定量の系統的誤差です。 

まず、系統誤差を測定してみましょう。推定器 $\hat{\theta}_n$ の場合、*統計的バイアス* の数学的図解は次のように定義できます。 

$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\mathrm{bias}(\hat{\theta}_n) = 0$ の場合、推定器 $\hat{\theta}_n$ の期待値はパラメーターの真の値と等しくなることに注意してください。この場合、$\hat{\theta}_n$ は不偏推定器であると言います。一般に、期待値が真のパラメーターと同じであるため、不偏推定器の方がバイアス推定器よりも優れています。 

ただし、実際には偏った推定量が頻繁に使用されることに注意する価値があります。不偏推定量が、さらなる仮定なしには存在しない場合や、計算が困難になる場合があります。これは推定器の重大な欠陥のように思えるかもしれませんが、実際に遭遇する推定量の大部分は、使用可能なサンプル数が無限大になる傾向があるためバイアスがゼロになる傾向があるという意味で、少なくとも漸近的に偏りがない ($\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$)。 

### 分散と標準偏差

次に、推定器でランダム性を測定します。:numref:`sec_random_variables` から、*標準偏差* (または*標準誤差*) は分散の平方根として定義されることを思い出してください。推定器の標準偏差または分散を測定することで、推定量の変動の程度を測定できます。 

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

:eqref:`eq_var_est`と:eqref:`eq_mse_est`を比較することは重要です。この方程式では、真の母集団値 $\theta$ と比較するのではなく、期待される標本平均である $E(\hat{\theta}_n)$ と比較します。したがって、推定器が真の値からどれだけ離れているかを測定するのではなく、推定器自体の変動を測定します。 

### バイアスと分散のトレードオフ

これら 2 つの主要成分が平均二乗誤差に寄与していることは直感的に明らかです。少し衝撃的なのは、これが実際には平均二乗誤差をこれら 2 つの寄与分に 3 つ目の寄与分を加えたものの「分解」であることを示すことができるということです。つまり、平均二乗誤差は、偏り、分散、既約誤差の二乗の和として書くことができます。 

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \mathrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \mathrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (\mathrm{bias} [\hat{\theta}_n])^2 + \mathrm{Var} (\hat{\theta}_n) + \mathrm{Var} [\theta].\\
\end{aligned}
$$

上記の式を*バイアスと分散のトレードオフ*と呼びます。平均二乗誤差は 3 つの誤差の原因 (: the error from high bias, the error from high variance and the irreducible error. The bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. If a model suffers from high bias error, we often say it is *underfitting* or lack of *flexibilty* as introduced in (:numref:`sec_model_selection`) に分けることができます。通常、分散が大きいのは、モデルが複雑すぎて、トレーニングデータに過適合することが原因です。その結果、*過適合* モデルはデータ内の小さな変動の影響を受けやすくなります。モデルに高分散がある場合、(:numref:`sec_model_selection`) で紹介したように、モデルは*過適合* で*一般化*がないとよく言われます。既約誤差は $\theta$ 自体のノイズに起因します。 

### コード内での推定量の評価

推定器の標準偏差は、テンソル `a` に対して `a.std()` を呼び出すだけで実装されているため、省略しますが、統計的バイアスと平均二乗誤差を実装します。

```{.python .input}
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

偏りと分散のトレードオフの方程式を説明するために、$10,000$ 標本をもつ正規分布 $\mathcal{N}(\theta, \sigma^2)$ をシミュレートします。ここでは $\theta = 1$ と $\sigma = 4$ を使います。推定器は与えられた標本の関数なので、ここでは標本の平均をこの正規分布 $\mathcal{N}(\theta, \sigma^2)$ における真の $\theta$ の推定量として使用します。

```{.python .input}
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

バイアスの二乗と推定器の分散の和を計算して、トレードオフ方程式を検証してみましょう。まず、推定量の MSE を計算します。

```{.python .input}
#@tab all
mse(samples, theta_true)
```

次に、$\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$を以下のように計算します。ご覧のとおり、この 2 つの値は数値の精度に一致しています。

```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 仮説検定の実施

統計的推論で最もよく遭遇するトピックは仮説検定です。仮説検定は20世紀初頭に普及しましたが、最初に使用されたのは1700年代のジョン・アーバスノットにまでさかのぼることができます。ジョンはロンドンで80年の出生記録を追跡し、毎年女性よりも男性の方が多いと結論付けました。それに続いて、$p$ 値とピアソンのカイ二乗検定を発明したカール・ピアソン、スチューデントの t 分布の父であるウィリアム・ゴセット、帰無仮説と有意性検定を始めたロナルド・フィッシャーによる知能遺産が現代の有意性検定です。 

*仮説検定* は、母集団に関する既定の記述に対して何らかの証拠を評価する方法です。既定のステートメントを*帰無仮説* $H_0$ と呼び、観測されたデータを使用して棄却しようとします。ここでは、$H_0$ を統計的有意性検定の開始点として使用します。*対立仮説* $H_A$ (または $H_1$) は、帰無仮説に反するステートメントです。帰無仮説は、変数間の関係を仮定する宣言形式で記述されることがよくあります。それは可能な限り明示的な要約を反映し、統計理論によって検証可能であるべきです。 

あなたが化学者だと想像してください。研究室で何千時間も過ごした後、数学を理解する能力を劇的に向上させる新しい薬を開発します。その魔法の力を示すには、それをテストする必要があります。当然のことながら、薬を服用して数学をよりよく学ぶのに役立つかどうかを確認するには、ボランティアが必要かもしれません。どうやって始めますか？ 

まず、いくつかのメトリックによって測定された数学の理解能力に違いがないように、ボランティアの2つのグループを慎重にランダムに選択する必要があります。この2つのグループは、一般にテストグループとコントロールグループと呼ばれます。*検査グループ*（または*治療グループ*）は、薬を経験する個人のグループであり、*コントロールグループ*は、ベンチマークとして確保されているユーザーのグループ、つまりこの薬を服用する以外は同一の環境設定を表します。このようにして、処理における独立変数の影響を除いて、すべての変数の影響が最小化されます。 

第二に、薬を服用した後、新しい数式を学んだ後、ボランティアに同じテストをさせるなど、2つのグループの数学の理解度を同じ指標で測定する必要があります。その後、それらのパフォーマンスを収集し、結果を比較できます。この場合、帰無仮説は 2 つのグループ間に差がないという仮説であり、代替は存在するという仮説になります。 

これはまだ完全に正式ではありません。あなたが慎重に考えなければならない多くの詳細があります。たとえば、数学の理解力をテストするのに適した測定基準は何ですか？あなたの薬の有効性を自信を持って主張できるように、あなたの検査には何人のボランティアがいますか？テストはどのくらいの期間実行すべきですか？2つのグループに違いがあるかどうかはどうやって判断しますか？平均的な成績だけに関心がありますか、それともスコアの変動範囲も気になりますか？などです。 

このように、仮説検定は、実験計画と観測結果の確実性に関する推論の枠組みとなります。帰無仮説が真である可能性が非常に低いことを示すことができれば、自信を持って棄却することができます。 

仮説検定をどのように扱うかというストーリーを完成させるには、ここでいくつかの追加の用語を導入し、上記の概念のいくつかを正式なものにする必要があります。 

### 統計的有意性

*統計的有意性*は、帰無仮説 $H_0$ を棄却すべきでない場合に誤って棄却する確率を測定します。つまり、 

$$ \text{statistical significance }= 1 - \alpha = 1 - P(\text{reject } H_0 \mid H_0 \text{ is true} ).$$

*Type I エラー* または*誤検出*とも呼ばれます。$\alpha$ は*有意水準*と呼ばれ、一般的に使用される値は $5\ %$, i.e., $1-\ alpha = 95\ %$ です。有意水準は、真の帰無仮説を棄却する場合に、我々が取るべきリスクの水準として説明できます。 

:numref:`fig_statistical_significance` は、2 標本仮説検定における特定の正規分布の観測値と確率を示します。観測データの例が $95\ %$ の閾値外にある場合、帰無仮説の仮定の下では非常にありそうもない観測になります。したがって、帰無仮説に誤りがある可能性があり、棄却します。 

![Statistical significance.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`

### 統計的検出力

*統計的検出力*（または*感度*）は、帰無仮説 $H_0$ を棄却すべきときに棄却する確率を測定します。つまり、 

$$ \text{statistical power }= 1 - \beta = 1 - P(\text{ fail to reject } H_0  \mid H_0 \text{ is false} ).$$

*第1種の過誤* は帰無仮説が真である場合に棄却することによって生じる誤りであり、*第II種の過誤*は帰無仮説が偽であるときに棄却できなかったために生じる誤りであることを思い出してください。タイプIIの誤差は通常 $\beta$ と表されるため、対応する統計的検出力は $1-\beta$ です。 

直感的に言うと、統計的検出力は、目的の統計的有意水準で最小の大きさの実際の不一致が検定される可能性として解釈できます。80\ %$は一般的に使用される統計的検出力のしきい値です。統計的検出力が高いほど、真の差を検出する可能性が高くなります。 

統計的検出力の最も一般的な用途の 1 つは、必要なサンプル数の決定です。帰無仮説が偽である場合に棄却する確率は、その帰無仮説が偽である程度 (*効果の大きさ* と呼ばれる) とサンプル数によって決まります。ご想像のとおり、効果サイズが小さいと、高い確率で検出できるサンプル数が非常に多くなります。この簡単な付録の範囲を超えて詳細に導き出すために, 例として, サンプルが平均ゼロ分散1ガウスから来たという帰無仮説を棄却できるようにしたい, サンプルの平均は実際には1に近いと考えています, サンプルサイズが$8$だけだただし、サンプル母集団の真の平均が $0.01$ に近いと考える場合、差を検出するには $80000$ に近いサンプルサイズが必要です。 

そのパワーは水フィルターとして想像できます。この例えでは、ハイパワー仮説検定は、水中の有害物質を可能な限り削減する高品質の水ろ過システムのようなものです。一方、より小さな不一致は、比較的小さな物質が隙間から容易に逃げる可能性がある、低品質の水フィルターのようなものです。同様に、統計的検出力の検出力が十分でない場合、検定では小さい方の不一致が捕捉されないことがあります。 

### 検定統計量

*検定統計量* $T(x)$ は、標本データの特徴を要約するスカラーです。このような統計量を定義する目的は、異なる分布を区別して仮説検定を実行できるようにすることです。化学者の例を振り返ってみると、ある母集団が他の母集団よりも優れていることを示したいのであれば、平均を検定統計量としてとるのが妥当かもしれません。検定統計量の選択が異なると、統計的検出力が大幅に異なる統計的検定につながる可能性があります。 

多くの場合、$T(X)$ (帰無仮説に基づく検定統計量の分布) は、帰無仮説で考えると、正規分布などの一般的な確率分布に少なくともほぼ従います。このような分布を明示的に導き出し、データセットで検定統計量を測定できれば、統計量が予想される範囲をはるかに超えていれば、帰無仮説を安全に棄却できます。これを定量的にすると、$p$ 値の概念が生まれます。 

### $p$

$p$ (または*確率値*) は、帰無仮説が*真*であると仮定して、$T(X)$ が観測された検定統計量 $T(x)$ と少なくとも同じ極値になる確率です。つまり、 

$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$

$p$ の値が、定義済みの固定された統計的有意水準 $\alpha$ 以下の場合、帰無仮説を棄却できます。そうでなければ、帰無仮説を棄却する証拠がないと結論づけます。特定の人口分布について、*棄却地域* は、統計的有意水準 $\alpha$ より小さい値の $p$ を持つすべてのポイントに含まれる区間になります。 

### 片側テストと両面テスト

通常、有意検定には片側検定と両側検定の2種類があります。*片側検定* (または*片側検定*) は、帰無仮説と対立仮説に一方向しかない場合に適用できます。たとえば、帰無仮説では、真のパラメーター $\theta$ は値 $c$ 以下であると仮定できます。対立仮説は $\theta$ は $c$ より大きいというものです。つまり、棄却範囲はサンプリング分布の片側のみにあります。片側検定とは異なり、*両側検定* (または*両側検定*) は、棄却領域が標本分布の両側にある場合に適用できます。この場合の例には、真のパラメーター $\theta$ が値 $c$ と等しいという帰無仮説があります。対立仮説は $\theta$ は $c$ と等しくないというものです。 

### 仮説検定の一般的な手順

上記の概念に慣れたら、仮説検定の一般的な手順を見ていきましょう。 

1. 問題を述べ、帰無仮説 $H_0$ を立てます。
2. 統計的有意水準 $\alpha$ と統計的検出力 ($1 - \beta$) を設定します。
3. 実験を通してサンプルを入手する。必要なサンプル数は、統計的検出力と予想される効果の大きさによって異なります。
4. 検定統計量と $p$ 値を計算します。
5. $p$ 値と統計的有意水準 $\alpha$ に基づいて、帰無仮説を保持するか棄却するかを決定します。

仮説検定を行うには、帰無仮説と取るべきリスクのレベルを定義することから始めます。次に、帰無仮説に対する証拠として検定統計量の極値を取って、標本の検定統計量を計算します。検定統計量が棄却領域内にある場合、帰無仮説を棄却して対立仮説を優先する場合があります。 

仮説検定は、臨床試験やA/B検査など、さまざまなシナリオに適用できます。 

## 信頼区間の構築

パラメーター $\theta$ の値を推定する場合、$\hat \theta$ などの点推定器には不確実性の概念が含まれていないため、有用性は限定的です。むしろ、真のパラメーター $\theta$ を含む区間を高い確率で生成できれば、はるかに良いでしょう。一世紀前にそのような考えに興味を持っていたら、1937年に信頼区間の概念を初めて導入したJerzy Neyman :cite:`Neyman.1937`の「古典的確率論に基づく統計的推定理論の概要」を読むとワクワクするでしょう。 

信頼区間は、特定の確実度でできる限り小さくすることが有用です。それを導き出す方法を見てみましょう。 

### 定義

数学的には、真のパラメーター $\theta$ の*信頼区間* は、標本データから以下のように計算された区間 $C_n$ です。 

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

ここで $\alpha \in (0, 1)$ と $1 - \alpha$ は、区間の*信頼水準*または*カバレッジ*と呼ばれます。これは、上で説明した有意水準と同じ $\alpha$ です。 

:eqref:`eq_confidence` は変数 $C_n$ に関するものであり、固定された $\theta$ に関するものではないことに注意してください。これを強調するために、$P_{\theta} (\theta \in C_n)$ ではなく $P_{\theta} (C_n \ni \theta)$ と書いています。 

### 通訳

生成された間隔の 95\ %$ confidence interval as an interval where you can be $95\ %$ sure the true parameter lies, however this is sadly not true.  The true parameter is fixed, and it is the interval that is random.  Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, $95\ %$ に真のパラメータが含まれると解釈するのはとても魅力的です。 

これは賢明なことのように思えるかもしれませんが、結果の解釈に実際の意味を持つ可能性があります。特に、真値を含まないことを「ほぼ確実」する区間を構築することで :eqref:`eq_confidence` を満たすことができます。ただし、そうすることがほとんどない限りは。このセクションは、魅力的でありながら虚偽の陳述を3つ提供することで締めくくります。これらの点についての詳細な説明は :cite:`Morey.Hoekstra.Rouder.ea.2016` にあります。 

* **ファラシー 1**。信頼区間が狭いということは、パラメーターを正確に推定できることを意味します。
* **ファラシー 2**。信頼区間内の値は、区間外の値よりも真の値である可能性が高くなります。
* **ファラシー 3**。特定の人が95\ %$ confidence interval contains the true value is $95\ %$を観測した確率。

信頼区間は微妙なオブジェクトだと言っても過言ではありません。しかし、解釈を明確にしておけば、強力なツールになり得ます。 

### ガウス分布の例

最も古典的な例として、未知の平均と分散をもつガウス分布の平均に対する信頼区間について考えてみましょう。ガウス分布の $\mathcal{N}(\mu, \sigma^2)$ から $n$ サンプル $\{x_i\}_{i=1}^n$ を収集するとします。平均と標準偏差の推定量は次のように計算できます。 

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

ここで、確率変数を考えてみると 

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

* $n-1$ *自由度* の*スチューデントの t 分布と呼ばれるよく知られた分布に従う確率変数が得られます。 

この分布は非常によく研究されており、例えば $n\rightarrow \infty$ としてはほぼ標準ガウス分布であることが知られています。したがって、ガウスc.d.f. の値をテーブルで調べると、$T$ の値は区間 $[-1.96, 1.96]$ に少なくとも $95\ %$ of the time.  For finite values of $n$ であると結論付けることができます。多少大きくなりますが、よく知られており、表では事前計算されています。 

したがって、大規模な$n$については、 

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

両辺に$\hat\sigma_n/\sqrt{n}$を掛けて$\hat\mu_n$を足すことでこれを並べ替えると、 

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

したがって、95\ %$の信頼区間が見つかったことがわかります。$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$ドル :eqlabel:`eq_gauss_confidence` 

:eqref:`eq_gauss_confidence`は統計学で最もよく使われる数式の一つと言っても過言ではありません。それを実装して、統計についての議論を締めくくりましょう。簡単にするために、ここでは漸近的な体制にあると仮定します。$N$ の小さい値には、プログラムまたは $t$ テーブルから取得した正しい `t_star` の値を含める必要があります。

```{.python .input}
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## [概要

* 統計学は推論問題に重点を置いていますが、ディープラーニングは明示的なプログラミングと理解なしに正確な予測を行うことに重点を置いています。
* 一般的な統計推論方法には、推定量の評価と比較、仮説検定の実施、信頼区間の構築の 3 つがあります。
* 最も一般的な推定量には、統計的偏差、標準偏差、平均二乗誤差の 3 つがあります。
* 信頼区間は、与えられた標本によって構築できる、真の母集団パラメータの推定範囲です。
* 仮説検定は、母集団に関するデフォルトステートメントに対していくつかの証拠を評価する方法です。

## 演習

1. $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$ とします。ここで、「iid」は*独立で同一分布* を表します。$\theta$ の次の推定量を考えてみます。
$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$ドル$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$ドル
    * $\hat{\theta}.$ の統計的偏り、標準偏差、平均二乗誤差を求めます。
    * $\tilde{\theta}.$ の統計的偏り、標準偏差、平均二乗誤差を求めます。
    * どの推定器が優れていますか？
1. 紹介する化学者の例では、両側仮説検定を行うための5つのステップを導き出すことができますか？統計的有意水準 $\alpha = 0.05$ と統計的検出力 $1 - \beta = 0.8$ を考えます。
1. $100$ で個別に生成されたデータセットに対して $N=2$ と $\alpha = 0.5$ を使用して信頼区間コードを実行し、結果の間隔 (この例では `t_star = 1.0`) をプロットします。真の平均 $0$ を含むには程遠い、非常に短い区間がいくつか表示されます。これは信頼区間の解釈と矛盾しますか？精度の高い推定値を示すために短い間隔を使用することに抵抗がないと感じますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1103)
:end_tab:

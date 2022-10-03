```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 自動微分
:label:`sec_autograd`

:numref:`sec_calculus` から、微分を計算することは、ディープネットワークの学習に使用するすべての最適化アルゴリズムにおいて重要なステップであることを思い出してください。計算は簡単ですが、手作業で計算するのは面倒でエラーを起こしやすく、この問題はモデルがより複雑になるにつれて大きくなります。 

幸いなことに、最新のディープラーニングフレームワークはすべて、*自動微分*（*autograd* と短縮されることが多い）を提供することで、この作業を私たちのプレートから取り除きます。連続する各関数にデータを渡すと、フレームワークは各値が他の値にどのように依存するかを追跡する*計算グラフ*を構築します。微分を計算するために、自動微分パッケージは連鎖則を適用してこのグラフを逆方向に処理します。この方法で連鎖則を適用する計算アルゴリズムは、*バックプロパゲーション*と呼ばれます。 

オートグラード図書館は過去10年間で注目を集めていますが、長い歴史があります。実際、オートグラードに関する最も初期の言及は、半世紀以上前にさかのぼります。:cite:`Wengert.1964`.現代のバックプロパゲーションの背後にある核となるアイデアは、1980年の:cite:`Speelpenning.1980`の博士論文にまでさかのぼり、1980年代後半の:cite:`Griewank.1989`でさらに発展しました。バックプロパゲーションは勾配を計算する既定の方法になりましたが、唯一の選択肢ではありません。たとえば、Juliaプログラミング言語は前方伝播:cite:`Revels.Lubin.Papamarkou.2016`を採用しています。方法を探る前に、まず autograd パッケージをマスターしましょう。 

## シンプルな機能

興味があると仮定しましょう (**列ベクトル$\mathbf{x}$に関して関数$y = 2\mathbf{x}^{\top}\mathbf{x}$を微分する**) まず、`x`に初期値を割り当てます。

```{.python .input  n=1}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**$\mathbf{x}$に対する$y$の勾配を計算する前に、それを保存する場所が必要です。**] ディープラーニングでは、数千または数百万の同じパラメータに関して導関数を連続的に計算する必要があるため、通常、微分を取るたびに新しいメモリを割り当てることは避けます。時間が経つと、メモリ不足の危険があります。ベクトル $\mathbf{x}$ に対するスカラー値関数の勾配はベクトル値であり、$\mathbf{x}$ と同じ形状であることに注意してください。

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
x.requires_grad_(True)  # Better create `x = torch.arange(4.0, requires_grad=True)`
x.grad                  # The default value is None
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**ここで `x` の関数を計算し、その結果を `y` に代入します**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

:begin_tab:`mxnet`
[**`x`に対する`y`の勾配を取ることができるようになりました**] `backward`メソッドを呼び出します。次に、`x`の`grad`属性を介してグラデーションにアクセスできます。
:end_tab:

:begin_tab:`pytorch`
[**`x`に対する`y`の勾配を取ることができるようになりました**] `backward`メソッドを呼び出します。次に、`x`の`grad`属性を介してグラデーションにアクセスできます。
:end_tab:

:begin_tab:`tensorflow`
[**`x`に対する`y`の勾配を計算できるようになりました**] `gradient`関数を呼び出します。
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**$\mathbf{x}$に対する関数$y = 2\mathbf{x}^{\top}\mathbf{x}$の勾配は$4\mathbf{x}$であることがすでにわかっています**) 自動勾配計算と期待される結果が同一であることを検証できます。

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**次に、`x` の別の関数を計算し、そのグラデーションを取得しましょう。**] MXNet は、新しいグラデーションを記録するたびにグラデーションバッファーをリセットすることに注意してください。
:end_tab:

:begin_tab:`pytorch`
[**それでは、`x`の別の関数を計算し、そのグラデーションを取得しましょう。**] PyTorchは、新しいグラデーションを記録するときにグラデーションバッファを自動的にリセットしないことに注意してください。代わりに、新しいグラデーションが既に保存されているグラデーションに追加されます。この動作は、複数の目的関数の合計を最適化する場合に便利です。勾配バッファをリセットするには、`x.grad.zero()` を次のように呼び出します。
:end_tab:

:begin_tab:`tensorflow`
[**次に、`x`の別の関数を計算し、その勾配を取得しましょう。**] TensorFlowは、新しいグラデーションを記録するたびにグラデーションバッファをリセットすることに注意してください。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## 非スカラー変数の逆方向

`y` がベクトルの場合、ベクトル `x` に関する `y` の導関数の最も自然な解釈は、`x` の各成分に対する `y` の各成分の偏微分を含む*ヤコビアン* と呼ばれる行列です。同様に、高次の`y`と`x`の場合、微分結果はさらに高次のテンソルになる可能性があります。 

ヤコビアンは、いくつかの高度な機械学習技術で現れますが、より一般的には、`y`の各成分の勾配をフルベクトル`x`に関して合計し、`x`と同じ形状のベクトルを生成します。たとえば、トレーニング例の*バッチ*のそれぞれについて個別に計算された損失関数の値を表すベクトルがよくあります。ここでは、(**例ごとに個別に計算された勾配を合計**) したいだけです。

:begin_tab:`mxnet`
MXNet は、勾配を計算する前に合計によってすべてのテンソルをスカラーに減らすことで、この問題を処理します。つまり、ヤコビアン $\partial_{\mathbf{x}} \mathbf{y}$ を返すのではなく、合計 $\partial_{\mathbf{x}} \sum_i y_i$ の勾配を返します。
:end_tab:

:begin_tab:`pytorch`
ディープラーニングフレームワークは、非スカラーテンソルの勾配を解釈する方法が異なるため、PyTorch は混乱を避けるためにいくつかの手順を実行します。非スカラーで `backward` を呼び出すと、オブジェクトをスカラーに減らす方法を PyTorch に指示しない限り、エラーが発生します。より正式には、`backward` が $\partial_{\mathbf{x}} \mathbf{y}$ ではなく $\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ を計算するように、いくつかのベクトル $\mathbf{v}$ を提供する必要があります。この次の部分は混乱するかもしれませんが、後で明らかになる理由から、この引数（$\mathbf{v}$を表す）は`gradient`という名前になっています。より詳細な説明については、Yang Zhangの[Medium post](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29)を参照してください。
:end_tab:

:begin_tab:`tensorflow`
デフォルトでは、TensorFlow は合計の勾配を返します。つまり、ヤコビアン $\partial_{\mathbf{x}} \mathbf{y}$ を返すのではなく、合計 $\partial_{\mathbf{x}} \sum_i y_i$ の勾配を返します。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## 計算をデタッチする

時々、[**記録された計算グラフの外に計算を移動する**] したい場合があります。たとえば、入力を使用して、勾配を計算したくない補助中間項を作成するとします。この場合、それぞれの計算影響グラフを最終結果から*切り離す*必要があります。次のおもちゃの例はこれをより明確にしています。`z = x * y`と`y = x * x`があるが、`y`を介して伝えられる影響ではなく、`z`に対する`x`の*直接的な*影響に焦点を当てたいとします。この場合、新しい変数 `u` を作成できます。この変数は、`y` と同じ値を取りますが、その*出所* (作成方法) が消去されています。したがって、`u`にはグラフに祖先がなく、`u`から`x`まで流れない勾配があります。たとえば、`z = x * u`の勾配を取ると、`x`という結果が得られます（`z = x * x * x`以降に予想していたような`3 * x * x`ではありません）。

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set `persistent=True` to preserve the compute graph. 
# This lets us run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

この手順は`y`につながるグラフから`y`の祖先を切り離しますが、`y`につながる計算グラフは存続するため、`y`に対する`y`の勾配を計算できることに注意してください。

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

## グラデーションと Python コントロールフロー

ここまで、`z = x * x * x`などの関数を使用して、入力から出力へのパスが明確に定義されているケースを確認しました。プログラミングは、結果の計算方法により多くの自由度を提供します。たとえば、補助変数や中間結果の条件選択に依存させることができます。自動微分を使用する利点の1つは、[**たとえ**]（**Pythonの制御フローの迷路を通過する必要がある関数**）（例えば、条件文、ループ、任意の関数呼び出し）の計算グラフを作成することです（**結果の変数の勾配を計算することはできます**）これを説明するために、`while`ループの反復回数と`if`ステートメントの評価の両方が入力`a`の値に依存する次のコードスニペットを考えてみましょう。

```{.python .input}
%%tab mxnet
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
%%tab pytorch
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
%%tab tensorflow
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

以下では、この関数を呼び出し、ランダムな値を入力として渡します。入力は確率変数なので、計算グラフがどのような形式になるかはわかりません。ただし、特定の入力に対して`f(a)`を実行するたびに、特定の計算グラフが認識され、その後`backward`を実行できます。

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

私たちの関数`f`はデモンストレーション目的で少し工夫されていますが、入力への依存は非常に単純です。これは、区分的に定義されたスケールを持つ`a`の*線形*関数です。したがって、`f(a) / a`は定数エントリのベクトルであり、さらに、`f(a) / a`は、`a`に対する`f(a)`の勾配と一致する必要があります。

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

動的制御フローは、ディープラーニングでは非常に一般的です。たとえば、テキストを処理する場合、計算グラフは入力の長さに依存します。このような場合、勾配を事前に計算することは不可能であるため、自動微分は統計的モデリングにとって不可欠になります。  

## ディスカッション

これで、自動微分のパワーを味わうことができました。デリバティブを自動的かつ効率的に計算するためのライブラリの開発は、ディープラーニングの実践者にとって生産性を大幅に向上させ、より高い懸念に集中できるようにしました。さらに、autogradを使用すると、ペンと紙のグラデーションの計算に非常に時間がかかる大規模なモデルを設計できます。興味深いことに、（統計的な意味で）autogradを使用してモデルを「最適化」しますが、autogradライブラリ自体の*最適化*（計算上の意味で）は、フレームワーク設計者にとって非常に興味深い豊富なテーマです。ここでは、コンパイラとグラフ操作のツールを活用して、最も便利でメモリ効率の良い方法で結果を計算します。  

とりあえず、次の基本を覚えておきましょう。(i) 微分を求める変数に勾配を付ける、(ii) 目標値の計算を記録する、(iii) バックプロパゲーション関数を実行する、(iv) 結果の勾配にアクセスする。 

## 演習

1. 二階微分は一次導関数よりも計算コストがはるかに高いのはなぜですか？
1. バックプロパゲーションの関数を実行したら、すぐに再度実行して、何が起こるかを確認します。なぜ？
1. `a` に対する `d` の微分を計算する制御フローの例では、変数 `a` をランダムなベクトルまたは行列に変更するとどうなるでしょうか。この時点で、`f(a)` の計算結果はスカラーではなくなりました。結果はどうなりますか？これをどのように分析しますか？
1. $f(x) = \sin(x)$としましょう。$f$ とその導関数 $f'$ のグラフをプロットします。$f'(x) = \cos(x)$ という事実を悪用するのではなく、結果を得るために自動微分を使用してください。 
1. $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$ としましょう。$x$ から $f(x)$ までの依存関係グラフのトレース結果を書き出します。 
1. 連鎖則を使用して前述の関数の微分 $\frac{df}{dx}$ を計算し、各項を前に作成した依存グラフに配置します。 
1. グラフと中間導関数の結果を考えると、勾配を計算するときにいくつかの選択肢があります。$x$から$f$まで開始し、$f$から$x$までトレースして結果を1回評価します。$x$ から $f$ へのパスは一般に *順微分* として知られていますが、$f$ から $x$ へのパスは後方微分として知られています。 
1. 前方微分と後方微分を使うのはいつですか？ヒント:必要な中間データの量、ステップを並列化する能力、関連する行列とベクトルのサイズを考慮してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

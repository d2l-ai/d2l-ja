# データ操作
:label:`sec_ndarray`

何かを成し遂げるためには、データを保存し操作する何らかの方法が必要です。一般に、データを扱うには、(i) データを取得することと、(ii) データがコンピューター内に収まった後に処理するという2つの重要なことがあります。なんらかの保存方法がないとデータを取得しても意味がないので、まずは合成データをいじって手を汚しましょう。まず、*テンソル* とも呼ばれる $n$ 次元配列を紹介します。 

Python で最も広く使われている科学計算パッケージである NumPy を使ったことがあるなら、この節はよく知っていることでしょう。どのフレームワークを使用するかにかかわらず、その*テンソルクラス* (MXNet では `ndarray`、PyTorch と TensorFlow の両方で `Tensor`) は NumPy の `ndarray` と似ていますが、いくつかのキラーな機能があります。まず、GPU は計算を高速化するために十分にサポートされていますが、NumPy は CPU 計算しかサポートしていません。第2に、テンソルクラスは自動微分をサポートしている。これらの特性により、テンソルクラスはディープラーニングに適しています。本書全体を通して、テンソルと言うとき、特に明記されていない限り、テンソルクラスのインスタンスを指しています。 

## はじめに

このセクションでは、本書を読み進めていくにつれて構築する基本的な数学および数値計算ツールを身に付けて、使い始めることを目指しています。数学的な概念やライブラリ関数を掘り起こすのに苦労しても心配しないでください。次のセクションでは、この資料を実際的な例のコンテキストで再検討します。一方、すでに何らかの経歴があり、数学的な内容をより深く掘り下げたい場合は、このセクションをスキップしてください。

:begin_tab:`mxnet`
まず、`np` (`numpy`) と `npx` (`numpy_extension`) のモジュールを MXnet からインポートします。ここで、`np` モジュールには NumPy でサポートされる関数が含まれ、`npx` モジュールには Numpy ライクな環境でディープラーニングを強化するために開発された一連の拡張機能が含まれています。テンソルを使用する場合、ほとんどの場合 `set_np` 関数を呼び出します。これは、MXNet の他のコンポーネントによるテンソル処理の互換性のためです。
:end_tab:

:begin_tab:`pytorch`
(**まず、`torch` をインポートします。PyTorch という名前ですが、`pytorch` ではなく `torch` をインポートする必要があることに注意してください **)
:end_tab:

:begin_tab:`tensorflow`
まず、`tensorflow` をインポートします。名前が少し長いため、短いエイリアス `tf` を付けてインポートすることがよくあります。
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**テンソルは数値の (多次元の) 配列を表す。**] 1つの軸で、テンソルを*vector* と呼びます。2 つの軸を持つテンソルを*matrix* と呼びます。$k > 2$ 軸では、特殊な名前を削除し、オブジェクトを $k^\mathrm{th}$ *次数テンソル* として参照します。

:begin_tab:`mxnet`
MXNet には、値が事前設定された新しいテンソルを作成するためのさまざまな関数が用意されています。たとえば、`arange(n)` を呼び出すと、0 (含まれる) から始まって `n` (含まれていない) で終わる等間隔の値のベクトルを作成できます。デフォルトのインターバルサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

:begin_tab:`pytorch`
PyTorch には、値があらかじめ入力された新しいテンソルを作成するためのさまざまな関数が用意されています。たとえば、`arange(n)` を呼び出すと、0 (含まれる) から始まって `n` (含まれていない) で終わる等間隔の値のベクトルを作成できます。デフォルトのインターバルサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow には、値が事前設定された新しいテンソルを作成するためのさまざまな関数が用意されています。たとえば、`range(n)` を呼び出すと、0 (含む) から始まって `n` (含まれていない) で終わる等間隔の値のベクトルを作成できます。デフォルトのインターバルサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

(**テンソルの*shape***) (~~と要素の総数~~) (各軸に沿った長さ) は `shape` プロパティを調べることでアクセスできます。

```{.python .input}
#@tab all
x.shape
```

テンソルの要素の総数、つまりすべての形状要素の積を知りたいだけなら、その大きさを調べることができます。ここではベクトルを扱っているので、その `shape` の 1 つの要素はそのサイズと同じです。

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

[**要素数も値も変えずにテンソルの形を変える**] には、`reshape` 関数を呼び出します。たとえば、テンソル `x` を形状 (12,) の行ベクトルから形状 (3, 4) の行列に変換できます。この新しいテンソルにはまったく同じ値が含まれていますが、3 行 4 列で構成された行列として表示されます。繰り返しますが、形状は変更されましたが、要素は変更されていません。形状を変更してもサイズは変更されないことに注意してください。

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

すべてのディメンションを手動で指定して形状を変更する必要はありません。ターゲットの形状が形状 (高さ、幅) をもつ行列の場合、幅がわかると、高さが暗黙的に与えられます。なぜ自分たちで除算をしなければならないのですか？上の例では、3 行の行列を得るために、3 行と 4 列の両方を指定しました。幸いなことに、テンソルは残りの次元を指定して 1 つの次元を自動的に計算できます。この機能を呼び出すには、テンソルで自動的に推論する次元に `-1` を配置します。私たちの場合、`x.reshape(3, 4)` を呼び出す代わりに、`x.reshape(-1, 4)` または `x.reshape(3, -1)` を同等に呼び出すことができます。 

通常、行列はゼロ、1、その他の定数、または特定の分布からランダムにサンプリングされた数値のいずれかで初期化されます。[**すべての要素を0に設定したテンソルを表すテンソルを作成できます**](~~or 1~~)、(2, 3, 4) の形状は次のようになります。

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

同様に、次のように、各要素を 1 に設定したテンソルを作成できます。

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

多くの場合、何らかの確率分布から [**テンソルの各要素の値をランダムにサンプリング**] します。たとえば、ニューラルネットワークでパラメーターとして機能する配列を作成する場合、通常、配列の値をランダムに初期化します。次のスニペットは、形状 (3, 4) を持つテンソルを作成します。各要素は、平均 0、標準偏差 1 の標準ガウス (正規) 分布からランダムにサンプリングされます。

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

また、数値を含む Python リスト (またはリストのリスト) を提供することで、目的のテンソルで [**各要素の正確な値を指定**] することもできます。ここで、最も外側のリストは軸 0 に対応し、内側のリストは軸 1 に対応しています。

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## オペレーション

この本はソフトウェア工学に関するものではありません。私たちの関心は、単に配列からデータを読み書きすることに限定されません。これらの配列に対して数学演算を実行したいと考えています。最も単純で有用な操作には、*elementwise* 演算があります。これらは、配列の各要素に標準のスカラー演算を適用します。2 つの配列を入力として取る関数の場合、要素単位の演算では、2 つの配列の対応する要素のペアごとに、何らかの標準二項演算子が適用されます。スカラーからスカラーにマッピングする任意の関数から要素単位の関数を作成できます。 

数学的表記法では、このような*単項* スカラー演算子 (入力を 1 つ取る) を $f: \mathbb{R} \rightarrow \mathbb{R}$ というシグネチャで表します。これは、関数が任意の実数 ($\mathbb{R}$) から別の実数にマッピングしていることを意味します。同様に、シグネチャ $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$ によって、*binary* スカラー演算子 (2 つの実数入力を取り、1 つの出力を生成する) を表します。同じ形状* の 2 つのベクトル $\mathbf{u}$ と $\mathbf{v}$ と二項演算子 $f$ を指定すると、$i$ すべてに対して $c_i \gets f(u_i, v_i)$ を設定することでベクトル $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ を生成できます。$c_i, u_i$ と $v_i$ は $\mathbf{c}, \mathbf{u}$ および $\mathbf{v}$ の $i^\mathrm{th}$ 要素です。ここでは、スカラー関数を要素単位のベクトル演算に*リフト* して、ベクトル値 $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ を生成しました。 

一般的な標準算術演算子 (`+`、`-`、`*`、`/`、および `**`) はすべて、任意の形状の同じ形状のテンソルに対して要素単位の演算に*解除* されています。要素単位の演算は、同じ形状の任意の 2 つのテンソルに対して呼び出すことができます。次の例では、カンマを使用して 5 要素のタプルを生成します。各要素は要素ごとの演算の結果です。 

### オペレーション

[**一般的な標準算術演算子 (`+`、`-`、`*`、`/`、`**`) は、すべて要素単位の演算に*解除* されました。**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

べき乗のような単項演算子を含む、多くの (**より多くの演算を要素単位に適用できる**)。

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

要素単位の計算に加えて、ベクトルドット積や行列乗算などの線形代数演算も実行できます。:numref:`sec_linear-algebra`では、線形代数の重要な部分（事前知識は想定されていない）について説明します。 

また、複数のテンソルを端から端まで積み重ねて [***連結*、**] してより大きなテンソルを形成することもできます。テンソルのリストを提供し、どの軸に沿って連結するかをシステムに指示するだけです。以下の例は、行 (図形の最初の要素である軸 0) と列 (軸1、図形の 2 番目の要素) に沿って 2 つの行列を連結した場合の動作を示しています。1 番目の出力テンソルの軸 0 の長さ ($6$) は、2 つの入力テンソルの軸 0 の長さ ($3 + 3$) の和であり、2 番目の出力テンソルの軸 1 の長さ ($8$) は、2 つの入力テンソルの軸 1 の長さ ($4 + 4$) の合計であることがわかります。

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

時々、[**論理文で二項テンソルを構築する*。**] `X == Y` を例に挙げてみましょう。位置ごとに `X` と `Y` がその位置で等しい場合、新しいテンソルの対応するエントリは値 1 を取ります。これは、論理ステートメント `X == Y` がその位置で真であることを意味します。

```{.python .input}
#@tab all
X == Y
```

[**テンソルのすべての要素を合計すると**] は要素が 1 つだけのテンソルになります。

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## ブロードキャストメカニズム
:label:`subsec_broadcasting`

上のセクションでは、同じ形状の 2 つのテンソルに対して要素単位の演算を実行する方法を説明しました。特定の条件下では、形状が異なっていても [**ブロードキャストメカニズム*を呼び出すことで要素単位の演算を実行できる。**] このメカニズムは次のように機能します。まず、要素を適切にコピーして一方または両方の配列を展開し、この変換後、2 つのテンソルが同じ形。次に、結果の配列に対して要素ごとの演算を実行します。 

ほとんどの場合、次の例のように、配列が最初は長さが 1 しかない軸に沿ってブロードキャストします。

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

`a` と `b` はそれぞれ $3\times1$ と $1\times2$ の行列なので、加算してもそれらの形状は一致しません。以下のように、両方の行列のエントリをより大きな $3\times2$ 行列に「ブロードキャスト」します。行列 `a` では列を複製し、行列 `b` では行を複製してから両方を要素ごとに加算します。

```{.python .input}
#@tab all
a + b
```

## インデックス作成とスライシング

他の Python 配列と同様に、テンソル内の要素にはインデックスでアクセスできます。Python の配列と同様に、最初の要素のインデックスは 0 で、範囲は最初で最後の要素の*前* を含むように指定されます。標準の Python リストと同様に、負のインデックスを使って、リストの最後に対する相対的な位置に従って要素にアクセスできます。 

したがって、[**`[-1]` は最後の要素を選択し、`[1:3]` は 2 番目と 3 番目の要素**] を選択します。

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
reading以外にも (**インデックスを指定して行列の要素を書くこともできます**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow の `Tensors` は不変であり、割り当てることはできません。TensorFlow の `Variables` は、割り当てをサポートする状態の可変コンテナです。TensorFlow のグラデーションは `Variable` の割り当てでは逆流しないことに注意してください。 

`Variable` 全体に値を代入するだけでなく、インデックスを指定することで `Variable` の要素を書くことができます。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

[**複数の要素に同じ値を割り当てるには、すべての要素にインデックスを付けてから値を割り当てます。**] たとえば、`[0:2, :]` は 1 行目と 2 行目にアクセスし、`:` は軸 1 (列) に沿ってすべての要素を取得します。行列の索引付けについて説明しましたが、これは明らかにベクトルや2次元以上のテンソルでも機能します。

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## メモリーの節約

[**操作を実行すると、新しいメモリがホストの結果に割り当てられる場合があります**] たとえば、`Y = X + Y` と書くと、`Y` が指していたテンソルを逆参照し、代わりに新しく割り当てられたメモリで `Y` をポイントします。次の例では、Python の `id()` 関数でこれを実証しています。この関数は、メモリ内の参照先オブジェクトの正確なアドレスを与えます。`Y = Y + X` を実行すると、`id(Y)` が別の場所を指していることがわかります。これは、Python が最初に `Y + X` を評価し、結果に対して新しいメモリを割り当て、`Y` がメモリ内のこの新しい位置を指すようにするためです。

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

これは、2 つの理由から望ましくない場合があります。まず、常に不必要にメモリを割り当てることを回避したくありません。機械学習では、数百メガバイトのパラメータがあり、それらすべてを毎秒複数回更新することがあります。通常は、これらの更新を「その場で」* 実行します。2つ目は、複数の変数から同じパラメータを指す場合です。インプレースで更新しなければ、他の参照は古いメモリ位置を指すため、コードの一部が誤って古いパラメータを参照する可能性があります。

:begin_tab:`mxnet, pytorch`
幸い、(**インプレース操作の実行**) は簡単です。操作の結果は、`Y[:] = <expression>` のようにスライス表記を使用して、前に割り当てた配列に代入できます。この概念を説明するために、`zeros_like` を使用して $0$ エントリのブロックを割り当てて、別の `Y` と同じ形状の新しい行列 `Z` を最初に作成します。
:end_tab:

:begin_tab:`tensorflow`
`Variables` は、TensorFlow の可変状態のコンテナーです。モデルパラメータを保存する方法を提供します。`assign` を使用して `Variable` に操作の結果を割り当てることができます。この概念を説明するために、`zeros_like` を使用して $0$ エントリのブロックを割り当てて、別のテンソル `Y` と同じ形状の `Variable` `Z` を作成します。
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**`X` の値が以降の計算で再利用されない場合、`X[:] = X + Y` または `X += Y` を使用して操作のメモリオーバーヘッドを減らすこともできます。**]
:end_tab:

:begin_tab:`tensorflow`
`Variable` に状態を永続的に保存した後でも、モデルパラメーターではないテンソルに対する過剰な割り当てを避けることで、メモリ使用量をさらに削減できます。 

TensorFlow `Tensors` は不変であり、勾配は `Variable` の割り当てを通過しないため、TensorFlow では個々の操作をインプレースで実行する明示的な方法を提供していません。 

ただし、TensorFlow には `tf.function` デコレータが用意されており、実行前にコンパイルおよび最適化される TensorFlow グラフ内に計算をラップします。これにより、TensorFlow は未使用の値をプルーニングし、不要になった以前の割り当てを再利用できます。これにより、TensorFlow 計算のメモリオーバーヘッドが最小限に抑えられます。
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 他の Python オブジェクトへの変換

:begin_tab:`mxnet, tensorflow`
[**NumPy テンソル (`ndarray`) への変換 (`ndarray`) **]、またはその逆は簡単です。変換された結果はメモリを共有しません。この小さな不便さは、実際には非常に重要です。CPU や GPU で操作を実行するときに、Python の NumPy パッケージが同じメモリチャンクで何か他の処理を行いたいかどうかを待って、計算を中断したくありません。
:end_tab:

:begin_tab:`pytorch`
[**NumPy テンソル (`ndarray`) への変換 (`ndarray`) **]、またはその逆は簡単です。Tensor と numpy 配列は、基になるメモリ位置を共有し、インプレース操作で一方を変更すると、もう一方も変更されます。
:end_tab:

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

(**サイズ 1 のテンソルを Python スカラーに変換**) するには、`item` 関数または Python の組み込み関数を呼び出すことができます。

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## [概要

* ディープラーニング用のデータを格納および操作するための主要なインターフェイスは、テンソル ($n$ 次元配列) です。基本的な数学演算、ブロードキャスト、インデックス作成、スライス、メモリ節約、他の Python オブジェクトへの変換など、さまざまな機能を提供します。

## 演習

1. このセクションのコードを実行します。このセクションの条件文 `X == Y` を `X < Y` または `X > Y` に変更し、取得できるテンソルの種類を確認します。
1. ブロードキャストメカニズムの要素によって動作する 2 つのテンソルを、3 次元テンソルなどの他の形状に置き換えます。結果は期待したとおりですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:

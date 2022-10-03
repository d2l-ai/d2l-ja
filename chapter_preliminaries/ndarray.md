```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# データ操作
:label:`sec_ndarray`

何かを成し遂げるためには、データを保存し操作する何らかの方法が必要です。一般的に、データに関して重要なことは2つあります。(i) データを取得することと、(ii) コンピューター内にいったんデータを処理することです。データを格納する方法がないとデータを取得しても意味がありません。まず、$n$ 次元配列で手を汚しましょう。これを*テンソル* とも呼びます。NumPy の科学計算パッケージを既に知っているなら、これは簡単です。すべての最新のディープラーニングフレームワークでは、*テンソルクラス*（MXNetでは`ndarray`、PyTorchおよびTensorFlowでは`Tensor`）は、NumPyの`ndarray`に似ており、いくつかのキラー機能が追加されています。まず、テンソルクラスは自動微分をサポートします。第二に、数値計算を高速化するためにGPUを活用しますが、NumPyはCPUでのみ動作します。これらの特性により、ニューラルネットワークはコーディングが容易で、実行も高速になります。 

## はじめに

:begin_tab:`mxnet`
まず、`np` (`numpy`) および `npx` (`numpy_extension`) モジュールを MXNet からインポートします。ここで、`np` モジュールには NumPy でサポートされる関数が含まれていますが、`npx` モジュールには Numpy ライクな環境でディープラーニングを強化するために開発された一連の拡張が含まれています。テンソルを使用する場合、ほとんどの場合、`set_np` 関数を呼び出します。これは、MXNet の他のコンポーネントによるテンソル処理の互換性のためです。
:end_tab:

:begin_tab:`pytorch`
(**まず、PyTorch ライブラリをインポートします。パッケージ名は `torch`.** であることに注意してください)
:end_tab:

:begin_tab:`tensorflow`
まず、`tensorflow`をインポートします。簡潔にするために、開業医はしばしば `tf` というエイリアスを割り当てます。
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

[**テンソルは数値の (場合によっては多次元の) 配列を表します。**] 1つの軸では、テンソルは*ベクトル*と呼ばれます。2 つの軸では、テンソルは*マトリックス* と呼ばれます。$k > 2$ 軸では、特殊な名前を削除し、オブジェクトを $k^\mathrm{th}$ *次数テンソル* と呼びます。

:begin_tab:`mxnet`
MXNet は、値があらかじめ入力された新しいテンソルを作成するためのさまざまな関数を提供します。たとえば、`arange(n)` を呼び出すと、0（含まれる）から始まり `n`（含まれていない）で終わる等間隔の値のベクトルを作成できます。デフォルトでは、間隔のサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

:begin_tab:`pytorch`
PyTorch は、値があらかじめ入力された新しいテンソルを作成するためのさまざまな関数を提供します。たとえば、`arange(n)` を呼び出すと、0（含まれる）から始まり `n`（含まれていない）で終わる等間隔の値のベクトルを作成できます。デフォルトでは、間隔のサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow は、値があらかじめ入力された新しいテンソルを作成するためのさまざまな関数を提供します。たとえば、`range(n)` を呼び出すと、0（含まれる）から始まり `n`（含まれていない）で終わる等間隔の値のベクトルを作成できます。デフォルトでは、間隔のサイズは $1$ です。特に指定しない限り、新しいテンソルはメインメモリに格納され、CPU ベースの計算用に指定されます。
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

:begin_tab:`mxnet`
これらの値はそれぞれ、テンソルの*要素*と呼ばれます。テンソル `x` には 12 の要素が含まれています。テンソルの要素の総数は、`size` 属性を介して調べることができます。
:end_tab:

:begin_tab:`pytorch`
これらの値はそれぞれ、テンソルの*要素*と呼ばれます。テンソル `x` には 12 の要素が含まれています。テンソルの要素の総数は、`numel` メソッドで調べることができます。
:end_tab:

:begin_tab:`tensorflow`
これらの値はそれぞれ、テンソルの*要素*と呼ばれます。テンソル `x` には 12 の要素が含まれています。テンソルの要素の総数は、関数 `size` で調べることができます。
:end_tab:

```{.python .input}
%%tab mxnet
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

（**テンソルの*形状***）（各軸に沿った長さ）には、`shape`属性を調べることでアクセスできます。ここではベクトルを扱っているので、`shape` は 1 つの要素だけを含み、サイズと同じです。

```{.python .input}
%%tab all
x.shape
```

`reshape` を呼び出すことで、[**サイズや値を変更せずにテンソルの形状を変更**] できます。たとえば、形状が (12,) のベクトル `x` を (3, 4) の形をした行列 `X` に変換できます。この新しいテンソルはすべての要素を保持しますが、それらをマトリックスに再構成します。ベクトルの要素は一度に 1 行ずつ、つまり `x[3] == X[0, 3]` にレイアウトされていることに注目してください。

```{.python .input}
%%tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

すべての形状コンポーネントを `reshape` に指定するのは冗長であることに注意してください。テンソルのサイズはすでにわかっているので、残りを考えれば、形状の1つのコンポーネントを計算できます。たとえば、サイズが$n$のテンソルとターゲット形状（$h$、$w$）を考えると、$w = n/h$であることがわかります。シェイプの 1 つのコンポーネントを自動的に推測するには、自動的に推測されるシェイプコンポーネントに `-1` を配置します。私たちの場合、`x.reshape(3, 4)`を呼び出す代わりに、`x.reshape(-1, 4)`または`x.reshape(3, -1)`を同等に呼び出すことができました。 

実務家は、多くの場合、すべて0または1を含むように初期化されたテンソルを扱う必要があります。[**すべての要素をゼロに設定したテンソルを構築できます**](~~または one~~) と (2, 3, 4) の形状は `zeros` 関数を使用します。

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

同様に、`ones` を呼び出すと、すべて 1 のテンソルを作成できます。

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

私たちはしばしば、与えられた確率分布から [**各要素を無作為に（そして独立して）**] サンプリングしたいと考えています。たとえば、ニューラルネットワークのパラメーターはランダムに初期化されることがよくあります。次のスニペットは、平均 0、標準偏差 1 の標準ガウス (正規) 分布から抽出された要素でテンソルを作成します。

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

最後に、数値リテラルを含む (ネストされている可能性もある) Python リストを提供することで [**各要素の正確な値を提供する**]、テンソルを構築できます。ここでは、リストのリストを持つ行列を作成します。最も外側のリストは軸0に対応し、内側のリストは軸1に対応します。

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## インデックス作成とスライス

Python のリストと同様に、インデックス (0 から開始) することでテンソル要素にアクセスできます。リストの末尾からの相対的な位置に基づいて要素にアクセスするには、負のインデックスを使用できます。最後に、スライス（例：`X[start:stop]`）を介してインデックスの全範囲にアクセスできます。戻り値には最初のインデックス（`start`）*が含まれますが、最後の*（`stop`）は含まれません。最後に、$k^\mathrm{th}$ 次数テンソルにインデックス (またはスライス) を 1 つだけ指定すると、軸 0 に沿って適用されます。したがって、次のコードでは、[**`[-1]` は最後の行を選択し、`[1:3]` は2番目と3番目の行を選択します**]。

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
読むだけでなく、(**インデックスを指定して行列の要素を記述することもできます。**)
:end_tab:

:begin_tab:`tensorflow`
TensorFlow の `Tensors` は不変であり、割り当てることはできません。TensorFlow の `Variables` は、割り当てをサポートする状態の可変コンテナです。TensorFlow のグラデーションは `Variable` の割り当てを通して逆流しないことに注意してください。 

`Variable` 全体に値を割り当てる以外に、インデックスを指定して `Variable` の要素を記述できます。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

[**複数の要素に同じ値を割り当てるには、割り当て操作の左側にインデックスを適用します。**] たとえば、`[:2, :]` は 1 行目と 2 行目にアクセスします。ここで、`:` は軸 1 (列) に沿ったすべての要素を取得します。行列の索引付けについて説明しましたが、これはベクトルと 2 次元以上のテンソルに対しても機能します。

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

## オペレーション

テンソルの構築方法と、テンソルの要素の読み書き方法がわかったので、さまざまな数学演算でテンソルを操作することができます。最も有用なツールには、*要素単位*の操作があります。これらは、テンソルの各要素に標準のスカラー演算を適用します。入力として 2 つのテンソルを取る関数の場合、要素単位の演算は、対応する要素の各ペアに標準の二項演算子を適用します。スカラーからスカラーにマップする任意の関数から要素単位の関数を作成できます。 

数学的表記法では、そのようなことを表します
*単項* スカラー演算子 (1 つの入力を取る)
署名$f: \mathbb{R} \rightarrow \mathbb{R}$によります。これは単に、関数が任意の実数から他の実数にマップされることを意味します。$e^x$ のような単項演算子を含め、ほとんどの標準演算子は要素単位に適用できます。

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

同様に、シグネチャ$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$を介して実数のペアを（単一の）実数にマップする*バイナリ*スカラー演算子を示します。同じ形状* の任意の 2 つのベクトル $\mathbf{u}$ と $\mathbf{v}$ * と二項演算子 $f$ が与えられた場合、すべての $i$ に $c_i \gets f(u_i, v_i)$ を設定することにより、ベクトル $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ を生成できます。ここで、$c_i, u_i$ と $v_i$ は、ベクトル $\mathbf{c}, \mathbf{u}$ および $\mathbf{v}$ の $i^\mathrm{th}$ 要素です。ここでは、スカラー関数を要素単位のベクトル演算に*持ち上げ* して、ベクトル値の $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ を生成しました。加算（`+`）、減算（`-`）、乗算（`*`）、除算（`/`）、およびべき乗（`**`）の一般的な標準算術演算子はすべて、任意の形状の同じ形状のテンソルに対して要素単位の演算に*持ち上げられました。

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

要素単位の計算に加えて、ドット積や行列の乗算などの線形代数演算も実行できます。これらについては、:numref:`sec_linear-algebra`ですぐに詳しく説明します。 

また、複数のテンソルをまとめて [**連結**] し、それらを端から端まで積み重ねて、より大きなテンソルを形成することもできます。テンソルのリストを提供し、連結する軸をシステムに伝えるだけです。以下の例は、行 (軸 0) と列 (軸 1) に沿って 2 つの行列を連結するとどうなるかを示しています。最初の出力の軸0の長さ（$6$）は、2つの入力テンソルの軸0の長さ（$3 + 3$）の合計であることがわかります。一方、2番目の出力の軸1の長さ（$8$）は、2つの入力テンソルの軸1の長さ（$4 + 4$）の合計です。

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

時々、[***論理文*を介してバイナリテンソルを構築する。**] `X == Y`を例にとります。各位置`i, j`について、`X[i, j]`と`Y[i, j]`が等しい場合、結果の対応するエントリは値`1`をとり、そうでない場合は値`0`を取ります。

```{.python .input}
%%tab all
X == Y
```

[**テンソルのすべての要素を合計する**] は、要素が 1 つだけのテンソルになります。

```{.python .input}
%%tab mxnet, pytorch
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## 放送
:label:`subsec_broadcasting`

これで、同じ形状の 2 つのテンソルに対して要素単位の二項演算を実行する方法がわかりました。特定の条件下では、形状が異なる場合でも、[***ブロードキャストメカニズム*を呼び出すことで要素単位のバイナリ演算を実行できます**] ブロードキャストは、次の2段階の手順に従って動作します。（i）長さ1の軸に沿って要素をコピーして一方または両方の配列を拡張し、この後変換すると、2つのテンソルは同じ形状になります。（ii）結果の配列に対して要素単位の演算を実行します。

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

`a`と`b`はそれぞれ$3\times1$と$1\times2$の行列であるため、それらの形状は一致しません。ブロードキャストでは、行列 `a` を列に沿って複製し、行に沿って行列 `b` を要素ごとに加算する前に、より大きな $3\times2$ 行列を生成します。

```{.python .input}
%%tab all
a + b
```

## メモリを節約する

[**操作を実行すると、新しいメモリがホスト結果に割り当てられる可能性があります。**] たとえば、`Y = X + Y`と記述すると、`Y`が指していたテンソルを逆参照し、代わりに新しく割り当てられたメモリを指す`Y`を指します。この問題を Python の `id()` 関数で実証できます。この関数は、メモリ内の参照オブジェクトの正確なアドレスを提供します。`Y = Y + X` を実行した後、`id(Y)` は別の場所を指していることに注意してください。これは、Python が最初に`Y + X`を評価し、結果のために新しいメモリを割り当ててから、`Y`をメモリ内のこの新しい場所を指すためです。

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

これは、2 つの理由から望ましくない場合があります。まず、不必要にメモリを割り当てて回り回りたくありません。機械学習では、数百メガバイトのパラメータがあり、それらすべてを毎秒複数回更新することがよくあります。可能な限り、これらの更新を*その場*で実行したいと考えています。次に、複数の変数から同じパラメータを指す場合があります。その場で更新しない場合、メモリリークが発生したり、誤って古いパラメータを参照したりしないように、これらの参照をすべて更新するように注意する必要があります。

:begin_tab:`mxnet, pytorch`
幸い、(**インプレース操作の実行**) は簡単です。操作の結果は、スライス表記法 `Y[:] = <expression>` を使用して、以前に割り当てられた配列 `Y` に割り当てることができます。この概念を説明するために、`zeros_like`を使用してテンソル`Z`の値を初期化した後に上書きし、`Y`と同じ形状にします。
:end_tab:

:begin_tab:`tensorflow`
`Variables` は TensorFlow の可変状態のコンテナです。これらは、モデルパラメータを保存する方法を提供します。操作の結果を `assign` で `Variable` に割り当てることができます。この概念を説明するために、`Variable` `Z` の値を初期化した後に `zeros_like` を使用して上書きし、`Y` と同じ形状にします。
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**`X`の値が以降の計算で再利用されない場合、`X[:] = X + Y`または`X += Y`を使用して操作のメモリオーバーヘッドを減らすこともできます。**]
:end_tab:

:begin_tab:`tensorflow`
`Variable` に状態を永続的に保存した後でも、モデルパラメーターではないテンソルへの過剰な割り当てを回避して、メモリ使用量をさらに削減したい場合があります。TensorFlow `Tensors` は不変であり、グラデーションは `Variable` の割り当てを通過しないため、TensorFlow には個々の操作をインプレースで実行する明示的な方法はありません。 

ただし、TensorFlow は `tf.function` デコレータを提供し、実行前にコンパイルおよび最適化される TensorFlow グラフ内に計算をラップします。これにより、TensorFlow は未使用の値をプルーニングし、不要になった以前の割り当てを再利用できます。これにより、TensorFlow 計算のメモリオーバーヘッドが最小限に抑えられます。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 他の Python オブジェクトへの変換

:begin_tab:`mxnet, tensorflow`
[**NumPy テンソルへの変換 (`ndarray`) **]、またはその逆は簡単です。変換された結果はメモリを共有しません。この小さな不便さは実際には非常に重要です。CPU や GPU で操作を実行するとき、Python の NumPy パッケージが同じメモリチャンクで何か他のことをしたいかどうかを確認するのを待って、計算を停止したくありません。
:end_tab:

:begin_tab:`pytorch`
[**NumPy テンソルへの変換 (`ndarray`) **]、またはその逆は簡単です。トーチテンソルとnumpy配列は基礎となるメモリを共有し、インプレース操作で一方を変更するともう一方も変更されます。
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

(**サイズ1のテンソルをPythonスカラーに変換する**) には、`item`関数またはPythonの組み込み関数を呼び出すことができます。

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## まとめ

 * テンソルクラスは、ディープラーニングライブラリのデータを格納および操作するための主要なインターフェイスです。
 * テンソルは、構築ルーチン、索引付けとスライス、基本的な数学演算、ブロードキャスト、メモリ効率の良い代入、他の Python オブジェクトとの変換など、さまざまな機能を提供します。

## 演習

1. このセクションのコードを実行します。条件ステートメント `X == Y` を `X < Y` または `X > Y` に変更し、取得できるテンソルの種類を確認します。
1. 放送機構の要素で動作する2つのテンソルを他の形状、たとえば3次元テンソルに置き換えます。結果は期待したとおりですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:

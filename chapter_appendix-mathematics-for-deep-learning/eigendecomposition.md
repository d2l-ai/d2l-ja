# 固有分解
:label:`sec_eigendecompositions`

固有値は、線形代数を研究するときに遭遇する最も有用な概念の1つですが、初心者の場合、その重要性を見落としがちです。以下では固有分解を紹介し、なぜそんなに重要なのかという感覚を伝えようとする。  

次のエントリをもつ行列 $A$ があるとします。 

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

$A$ を任意のベクトル $\mathbf{v} = [x, y]^\top$ に適用すると、ベクトル $\mathbf{A}\mathbf{v} = [2x, -y]^\top$ が得られます。これは直感的に解釈できます。ベクトルを $x$ 方向に 2 倍の幅になるように引き伸ばし、$y$ 方向に反転します。 

しかし、何かが変わらない*いくつかの*ベクトルがあります。つまり、$[1, 0]^\top$ は $[2, 0]^\top$ に送信され、$[0, 1]^\top$ は $[0, -1]^\top$ に送信されます。これらのベクトルはまだ同じ線上にあり、唯一の変更点は、マトリックスがそれぞれ $2$ と $-1$ の係数で引き伸ばされることです。このようなベクトルを*固有ベクトル*と呼び、*固有値*で引き伸ばされる係数を呼びます。 

一般に、数値 $\lambda$ とベクトル $\mathbf{v}$ を見つけることができれば、  

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

$\mathbf{v}$ は $A$ の固有ベクトルであり、$\lambda$ は固有値であると言います。 

## 固有値の求め方を考えてみましょう。$\lambda \mathbf{v}$ を両辺から減算し、ベクトルを因数分解すると、上記は次のものと同等であることがわかります。 

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der` が起こるためには、$(\mathbf{A} - \lambda \mathbf{I})$ はある方向をゼロに圧縮しなければならないことがわかります。したがって、この方向は反転できず、行列式はゼロになります。したがって、$\lambda$ が何であるか $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ を見つけることによって、*固有値* を求めることができます。固有値が見つかると、$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ を解いて関連する*固有ベクトル*を見つけることができます。 

### 一例もっと難しいマトリックスでこれを見てみましょう 

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ を考えてみると、これは多項式の $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$ と等価であることがわかります。したがって、2 つの固有値は $4$ と $1$ になります。関連するベクトルを見つけるには、解く必要があります。 

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

これをそれぞれベクトル $[1, -1]^\top$ と $[1, 2]^\top$ で解くことができます。 

これは、組み込みの `numpy.linalg.eig` ルーチンを使用してコードで確認できます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy` は固有ベクトルを長さが 1 になるように正規化しますが、ここでは任意の長さであるとしました。さらに、符号の選択は任意です。しかし、計算されたベクトルは、同じ固有値で手作業で見つけたベクトルと平行です。 

## 行列の分解前の例をもう一歩続けてみましょう。させて 

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

列が行列 $\mathbf{A}$ の固有ベクトルである行列である。させて 

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

対角線上の固有値が関連付けられている行列である。そして、固有値と固有ベクトルの定義から、次のことが分かります。 

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

行列 $W$ は可逆なので、両辺に右側の $W^{-1}$ を掛けると、以下のように書けます。 

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

次のセクションでは、これによるいくつかの素晴らしい結果を見ていきますが、今のところ、線形独立な固有ベクトルの完全な集合を見つけることができれば ($W$ が可逆になるように)、そのような分解が存在することを知る必要があるだけです。 

## 固有分解の操作固有分解 :eqref:`eq_eig_decomp` の良い点の一つは、私たちが普段遭遇する多くの演算を固有分解という観点からきれいに書けることです。最初の例として、以下について考えてみましょう。 

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

これは、行列の正のべき乗に対して、固有値を同じべき乗に上げることで固有分解が得られることを示しています。負のべき乗についても同じことが言えるので、行列を反転させたい場合は、 

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

言い換えれば、各固有値を反転させるだけです。これは、各固有値がゼロでない限り機能します。そのため、可逆性はゼロの固有値を持たないことと同じであることがわかります。   

実際に、$\lambda_1, \ldots, \lambda_n$ が行列の固有値である場合、その行列の行列式は次のようになります。 

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

またはすべての固有値の積です。$\mathbf{W}$ がどのようなストレッチをしても $W^{-1}$ はそれを元に戻すので、最終的に起こる唯一のストレッチは、対角行列 $\boldsymbol{\Sigma}$ を乗算することであり、これは対角要素の積でボリュームをストレッチします。 

最後に、ランクは行列の線形独立列の最大数であったことを思い出してください。固有分解を詳しく調べると、ランクは $\mathbf{A}$ の非ゼロ固有値の数と同じであることがわかります。 

例は続くかもしれませんが、要点は明らかです。固有分解は多くの線形代数計算を単純化することができ、多くの数値アルゴリズムと線形代数で行う多くの解析の基礎となる基本的な操作です。  

## 対称行列の固有分解上記のプロセスが動作するのに十分な線形独立固有ベクトルを見つけることは必ずしも可能ではない.例えば、マトリックス 

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

固有ベクトルは 1 つ、つまり $(1, 0)^\top$ しかありません。このような行列を処理するには、我々がカバーできる以上の高度な手法 (ヨルダン正規形や特異値分解など) が必要です。多くの場合、固有ベクトルの完全なセットの存在を保証できる行列に注意を制限する必要があります。 

最もよく見られるファミリーは、$\mathbf{A} = \mathbf{A}^\top$ の行列である*対称行列* です。この場合、$W$ を*直交行列* (列の長さがすべて 1 つのベクトルで互いに直角な行列 ($\mathbf{W}^\top = \mathbf{W}^{-1}$)) とし、すべての固有値が実数になります。したがって、この特殊なケースでは :eqref:`eq_eig_decomp` を次のように書くことができます。 

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## ゲルシュゴリンの円定理固有値は直感的に推論するのが難しいことが多いです。任意の行列が提示された場合、固有値を計算せずに固有値が何であるかについて言えることはほとんどありません。ただし、最大値が対角線上にある場合、よく近似しやすくなる定理が1つあります。 

$\mathbf{A} = (a_{ij})$ を任意の正方行列 ($n\times n$) とします。$r_i = \sum_{j \neq i} |a_{ij}|$ を定義します。$\mathcal{D}_i$ は、中心が $a_{ii}$ の半径 $r_i$ をもつ複素平面上の円盤を表すとします。すると、$\mathbf{A}$ のすべての固有値が $\mathcal{D}_i$ の 1 つに含まれます。 

これは開梱するのが少し難しいかもしれないので、例を見てみましょう。行列を考えてみましょう。 

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

$r_1 = 0.3$、$r_2 = 0.6$、$r_3 = 0.8$、$r_4 = 0.9$ があります。行列は対称なので、固有値はすべて実数です。これは、すべての固有値が次の範囲の 1 つになることを意味します。  

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$

数値計算を実行すると、固有値がおよそ $0.99$、$2.97$、$4.95$、$9.08$ になり、すべて指定された範囲内であることがわかります。

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

このようにして、固有値を近似することができ、対角が他のすべての要素よりも著しく大きい場合、近似はかなり正確になります。   

それは小さなことですが、固有分解のような複雑で微妙な話題があるので、できる限り直感的に把握できるのは良いことです。 

## 役に立つアプリケーション:反復マップの発展

ここで、固有ベクトルが原理的に何であるかを理解したところで、ニューラルネットワークの振る舞いの中心となる問題、つまり適切な重みの初期化を深く理解するために、固有ベクトルがどのように使用されるかを見てみましょう。  

### 長期挙動としての固有ベクトル

ディープニューラルネットワークの初期化の完全な数学的調査は本文の範囲外ですが、固有値がこれらのモデルがどのように機能するかを理解するために、ここではおもちゃのバージョンを見ることができます。ご存知のように、ニューラルネットワークは、非線形演算による線形変換の層を散在させることによって動作します。ここでは簡単にするために、非線形性はなく、変換は単一の反復行列演算 $A$ であると仮定します。したがって、モデルの出力は次のようになります。 

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

これらのモデルが初期化されると、$A$ はガウスエントリをもつ乱数行列とみなされるので、そのうちの 1 つを作ってみましょう。具体的には、平均 0、分散 1 のガウス分布 $5 \times 5$ 行列から始めます。

```{.python .input}
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### ランダムデータに対する振る舞いトイモデルを簡単にするために、$\mathbf{v}_{in}$ で送るデータベクトルはランダムな 5 次元ガウスベクトルであると仮定します。何が起きて欲しいのか考えてみよう。コンテキストとして、一般的な機械学習問題を考えてみましょう。ここでは、画像のような入力データを、画像が猫の絵である確率などの予測に変換しようとしています。$\mathbf{A}$ を繰り返し適用すると乱数ベクトルが非常に長くなると、入力の小さな変化は増幅されて出力が大きく変化します。入力イメージにわずかな変更を加えると、予測が大きく異なります。これは正しくないようです！ 

反対に、$\mathbf{A}$ がランダムベクトルを短くすると、多数のレイヤーを通過した後、ベクトルは実質的に何も縮小されず、出力は入力に依存しません。これも明らかに正しくありません！ 

インプットに応じてアウトプットが変化することを確認するために、成長と減衰の間の狭い線を歩く必要がありますが、それほど多くはありません！ 

行列 $\mathbf{A}$ をランダムな入力ベクトルに対して繰り返し乗算し、ノルムを追跡するとどうなるか見てみましょう。

```{.python .input}
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

規範は手に負えないほど高まっています！実際、商のリストを取ると、パターンが見えます。

```{.python .input}
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

上記の計算の最後の部分を見ると、乱数ベクトルが `1.974459321485[...]` の係数でストレッチされていることがわかります。この場合、最後の部分は少しシフトしますが、ストレッチ係数は安定しています。   

### 固有ベクトルに関連して戻る

固有ベクトルと固有値は何かが引き伸ばされる量に対応することを見てきましたが、それは特定のベクトルと特定の伸張に対するものでした。$\mathbf{A}$の用途を見てみましょう。ここで少し注意点があります。それらをすべて見るには、複素数に行く必要があることがわかります。これらはストレッチと回転と考えることができます。複素数のノルム (実部と虚数部の二乗和の平方根) をとることで、その伸張係数を測定することができます。それらも並べ替えてみましょう。

```{.python .input}
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### ある観察

ここでは、予期せぬことが起きています。乱数ベクトルに適用された行列 $\mathbf{A}$ の長期伸張について以前に特定した数値は、*正確に* (小数点以下 13 桁まで正確です) です。$\mathbf{A}$ の最大固有値です。これは明らかに偶然ではありません！ 

しかし、幾何学的に何が起こっているのかを考えると、これは理にかなっています。乱数ベクトルを考えてみましょう。この乱数ベクトルはあらゆる方向に少しずつ向いているため、特に、最大固有値に関連する $\mathbf{A}$ の固有ベクトルと少なくとも少しは同じ方向を指しています。これは*原理固有値*、*原理固有ベクトル*と呼ばれるほど重要です。$\mathbf{A}$ を適用すると、ランダムベクトルは可能なすべての固有ベクトルに関連付けられているすべての方向に伸長されますが、とりわけこの原理固有ベクトルに関連付けられた方向に伸長されます。つまり、$A$ で適用すると、乱数ベクトルは長くなり、主固有ベクトルと整列する方向に近い方向を向きます。行列を何度も適用した後、すべての実用的な目的のために、ランダムベクトルが原理固有ベクトルに変換されるまで、原理固有ベクトルとの整列はますます近くなります。実際、このアルゴリズムは、行列の最大の固有値と固有ベクトルを見つけるための「累乗反復」と呼ばれるものの基礎となります。詳細については、:cite:`Van-Loan.Golub.1983` などを参照してください。 

### 正規化を修正する

さて、上記の議論から、ランダムベクトルを引き伸ばしたり押しつぶしたりするのではなく、プロセス全体を通してランダムベクトルをほぼ同じサイズにしておきたいと結論付けました。そのために、行列をこの原理固有値で再スケーリングし、最大の固有値が代わりに一になるようにします。この場合に何が起こるか見てみましょう。

```{.python .input}
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

また、前のように連続するノルム間の比率をプロットすると、実際に安定することがわかります。

```{.python .input}
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 結論

私たちは今、私たちが望んでいたことを正確に見ています！行列を主固有値で正規化すると、ランダムデータは以前のように爆発せず、最終的に特定の値に平衡化することがわかります。第一原理からこういうことができるのはいいことですが、その数学を深く見ると、独立平均ゼロ、分散 1 ガウス要素をもつ大きなランダム行列の最大固有値は平均で約$\sqrt{n}$、私たちの場合は $\sqrt{5} \approx 2.2$、*循環法* :cite:`Ginibre.1965`として知られている興味深い事実のためです。ランダム行列の固有値 (および特異値と呼ばれる関連オブジェクト) の関係は、:cite:`Pennington.Schoenholz.Ganguli.2017` 以降の研究で論じられたように、ニューラルネットワークの適切な初期化と深いつながりがあることが示されている。 

## 概要 * 固有ベクトルは、方向を変えずに行列によって引き伸ばされるベクトルです。* 固有値とは、行列を適用することによって固有ベクトルが引き伸ばされる量です。* 行列の固有分解により、多くの演算を固有値に対する演算に還元できます。ゲルシュゴリン円定理は行列の固有値の近似値を得ることができる。* 反復された行列のべき乗の振る舞いは、主に最大固有値の大きさに依存する。この理解は、ニューラルネットワークの初期化の理論に多くの応用があります。 

## Exercises
1. What are the eigenvalues and eigenvectors of
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  What are the eigenvalues and eigenvectors of the following matrix, and what is strange about this example compared to the previous one?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Without computing the eigenvalues, is it possible that the smallest eigenvalue of the following matrix is less that $0.5$? *Note*: this problem can be done in your head.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:

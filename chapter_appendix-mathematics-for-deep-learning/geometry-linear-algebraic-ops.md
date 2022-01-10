# 幾何学と線形代数演算
:label:`sec_geometry-linear-algebraic-ops`

:numref:`sec_linear-algebra` では、線形代数の基礎を学び、それを使ってデータを変換する一般的な演算を表現する方法を見ました。線形代数は、ディープラーニングや機械学習の分野で私たちが行っている多くの作業の根底にある重要な数学の柱の 1 つです。:numref:`sec_linear-algebra` には、最新のディープラーニングモデルの仕組みを伝えるのに十分な機械が含まれていましたが、このテーマにはさらに多くの機能があります。このセクションでは、線形代数演算の幾何学的解釈を強調し、固有値と固有ベクトルを含むいくつかの基本的な概念を紹介します。 

## ベクトルの幾何学まず、ベクトルの2つの一般的な幾何学的解釈、つまり空間の点または方向について議論する必要があります。基本的に、ベクトルは以下の Python リストのような数字のリストです。

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

数学者はほとんどの場合、これを*column* または*row* ベクトルとして記述します。つまり、 

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

または 

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

データ例は列ベクトルで、加重合計を形成するために使用される重みは行ベクトルであり、これらの解釈は異なることがよくあります。ただし、柔軟性があることは有益です。:numref:`sec_linear-algebra` で説明したように、単一のベクトルの既定の方向は列ベクトルですが、表形式のデータセットを表す行列では、各データ例を行列の行ベクトルとして扱うのがより一般的です。 

ベクトルを考えると、最初に与えるべき解釈は空間上の点です。2 次元または 3 次元では、ベクトルの成分を使用して、*origin* と呼ばれる固定参照と比較した空間内の点の位置を定義することで、これらの点を視覚化できます。これは :numref:`fig_grid` で見ることができます。 

![An illustration of visualizing vectors as points in the plane.  The first component of the vector gives the $x$-coordinate, the second component gives the $y$-coordinate.  Higher dimensions are analogous, although much harder to visualize.](../img/grid-points.svg)
:label:`fig_grid`

この幾何学的な観点から、問題をより抽象的なレベルで考えることができます。絵を猫か犬に分類するような乗り越えられないように見える問題に直面しなくなった, タスクを空間内のポイントの集まりとして抽象的に検討し、タスクを2つの異なるポイントのクラスターを分離する方法を発見することとしてタスクを描くことができます。. 

並行して、人々がしばしばベクトルをとる第2の視点があります。それは空間の方向です。ベクトル $\mathbf{v} = [3,2]^\top$ を右に $3$ 単位、原点から $2$ 単位上に位置すると考えることができるだけでなく、$3$ ステップを右に、$2$ ステップアップする方向そのものと考えることもできます。このように、図 :numref:`fig_arrow` のすべてのベクトルを同じものと見なします。 

![Any vector can be visualized as an arrow in the plane.  In this case, every vector drawn is a representation of the vector $(3,2)^\top$.](../img/par-vec.svg)
:label:`fig_arrow`

このシフトの利点の1つは、ベクトル加算の行為を視覚的に理解できることです。特に、:numref:`fig_add-vec` に見られるように、一方のベクトルによって与えられる方向に従い、次にもう一方のベクトルによって与えられる方向に従います。 

![We can visualize vector addition by first following one vector, and then another.](../img/vec-add.svg)
:label:`fig_add-vec`

ベクトル減算も同様の解釈をしています。$\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$ というアイデンティティを考慮すると、ベクトル $\mathbf{u}-\mathbf{v}$ がポイント $\mathbf{v}$ からポイント $\mathbf{u}$ に向かう方向であることがわかります。 

## 内積と角度 :numref:`sec_linear-algebra` で見たように、2 つの列ベクトル $\mathbf{u}$ と $\mathbf{v}$ をとると、計算によってドット積を形成できます。 

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

:eqref:`eq_dot_def` は対称であるため、古典的な乗算の表記法をミラーリングして、次のように記述します。 

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

ベクトルの順序を交換しても同じ答えが得られるという事実を強調しています。 

ドット積 :eqref:`eq_dot_def` も幾何学的解釈を認めている : it is closely related to the angle between two vectors.  Consider the angle shown in :numref:`fig_angle`。 

![Between any two vectors in the plane there is a well defined angle $\theta$.  We will see this angle is intimately tied to the dot product.](../img/vec-angle.svg)
:label:`fig_angle`

まず、2つの特定のベクトルを考えてみましょう。 

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

ベクトル $\mathbf{v}$ は長さ $r$ で $x$ 軸に平行に走ります。ベクトル $\mathbf{w}$ は長さが $s$ で、$x$ 軸に対して角度が $\theta$ です。これら 2 つのベクトルの内積を計算すると、 

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

いくつかの単純な代数的操作で、項を並べ替えて得ることができます 

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

つまり、これら2つの特定のベクトルについて、ノルムと組み合わされた内積は、2つのベクトルの間の角度を教えてくれます。これと同じ事実が一般的に当てはまります。ここでは式を導き出さないが、$\|\mathbf{v} - \mathbf{w}\|^2$ をドット積で、もう一方は余弦の法則を幾何学的に使う 2 つの方法で書くことを考えれば、完全な関係を得ることができる。実際、任意の 2 つのベクトル $\mathbf{v}$ と $\mathbf{w}$ について、2 つのベクトルの間の角度は次のようになります。 

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

計算では二次元を参照するものがないので、これは良い結果です。実際、これを300万から300万の次元で問題なく使用できます。 

簡単な例として、一対のベクトル間の角度を計算する方法を見てみましょう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

ここでは使用しませんが、角度が $\pi/2$ (または $90^{\circ}$) のベクトルを*直交*として参照することを知っておくと便利です。上の方程式を調べると、$\cos(\theta) = 0$ と同じである $\theta = \pi/2$ のときにこの現象が発生することがわかります。これが発生する唯一の方法は、内積自体がゼロで、$\mathbf{v}\cdot\mathbf{w} = 0$ の場合にのみ 2 つのベクトルが直交する場合です。これは、オブジェクトを幾何学的に理解する場合に役立つ公式です。 

角度の計算がなぜ役立つのか、と尋ねるのは合理的です。その答えは、データに期待される不変性の種類にあります。イメージと複製イメージについて考えてみましょう。すべてのピクセル値は同じですが、明るさは $10\ %$ です。通常、個々のピクセルの値は元の値とはかけ離れています。したがって、元のイメージと暗いイメージの間の距離を計算すると、距離が大きくなる可能性があります。ただし、ほとんどの ML アプリケーションでは、*content* は同じです。猫/犬の分類子に関する限り、これはまだ猫のイメージです。しかし、角度を考慮すると、$\mathbf{v}$ のベクトル $\mathbf{v}$ と $0.1\cdot\mathbf{v}$ の間の角度がゼロであることは分かりにくいです。これは、スケーリングベクトルが同じ方向を保ち、長さが変わるだけであるという事実に相当します。この角度は、暗い方のイメージを同一とみなします。 

このような例はいたるところにあります。テキストでは、同じことを言っている文書を2倍長く書けば、議論されているトピックが変わらないようにしたいと思うかもしれません。一部のエンコーディング (ボキャブラリ内の単語の出現回数を数えるなど) では、これはドキュメントをエンコードするベクトルの 2 倍に相当するので、角度も使用できます。 

### コサイン類似度 MLの文脈では、角度が2つのベクトルの近さを測定するために用いられる場合、実践者は*コサイン類似度*という用語を採用して $$\ cos (\ theta) =\ frac {\ mathbf {v}\ cdot\ mathbf {w}} {\ |\ mathbf {v}\ |\ mathbf {w}\ |}。$$ 

余弦は、2 つのベクトルが同じ方向を指している場合は最大値 $1$、反対方向を指す場合は最小値 $-1$、2 つのベクトルが直交している場合は値 $0$ を取ります。高次元ベクトルの成分が平均 $0$ でランダムにサンプリングされた場合、その余弦はほぼ常に $0$ に近くなることに注意してください。 

## ハイパープレーン

ベクトルの操作に加えて、線形代数で理解しなければならないもう1つの重要なオブジェクトは、線（2次元）または平面（3次元）の高次元への一般化である*超平面*です。$d$ 次元のベクトル空間では、超平面は $d-1$ 次元をもち、空間を 2 つの半空間に分割します。 

例から始めましょう。列ベクトル $\mathbf{w}=[2,1]^\top$ があるとします。「$\mathbf{w}\cdot\mathbf{v} = 1$で$\mathbf{v}$のポイントは何ですか？」ドット積と:eqref:`eq_angle_forumla`より上の角度の関係を思い出すと、これは$$\ |\ mathbf {v}\ |\ mathbf {w}\ |\ cos (\ theta) = 1\;\ iff\;\ |\ mathbf {v}\ |\ cos (\ theta) =\ frac {1} {\ |\ mathbf {w}\ |} =\ frac {1} {\ sqrt {5}}。$$ 

![Recalling trigonometry, we see the formula $\|\mathbf{v}\|\cos(\theta)$ is the length of the projection of the vector $\mathbf{v}$ onto the direction of $\mathbf{w}$](../img/proj-vec.svg)
:label:`fig_vector-project`

この式の幾何学的意味を考慮すると、:numref:`fig_vector-project` に示すように $\mathbf{v}$ の $\mathbf{w}$ 方向への投影の長さが正確に $1/\|\mathbf{w}\|$ であると言うことと同等であることがわかります。これが当てはまるすべての点の集合は、ベクトル $\mathbf{w}$ に対して直角の直線です。必要であれば、この直線の方程式を見つけることができ、$2x + y = 1$ または同等の $y = 1 - 2x$ であることがわかります。 

$\mathbf{w}\cdot\mathbf{v} > 1$または$\mathbf{w}\cdot\mathbf{v} < 1$の点セットについて尋ねたときに何が起こるかを見ると、これらはそれぞれ予測が$1/\|\mathbf{w}\|$より長い場合または短い場合であることがわかります。したがって、これらの 2 つの不等式は直線の両側を定義します。このようにして、:numref:`fig_space-division`で見られるように、片側のすべての点のドット積がスレッショルドを下回り、もう片方のポイントが上の半分にスペースを分割する方法を見つけました。 

![If we now consider the inequality version of the expression, we see that our hyperplane (in this case: just a line) separates the space into two halves.](../img/space-division.svg)
:label:`fig_space-division`

高次元の話はほぼ同じです。$\mathbf{w} = [1,2,3]^\top$ を取り、$\mathbf{w}\cdot\mathbf{v} = 1$ で三次元の点について尋ねると、与えられたベクトル $\mathbf{w}$ に対して直角の平面が得られます。:numref:`fig_higher-division` に示すように、2 つの不等式によって平面の両側が定義されます。 

![Hyperplanes in any dimension separate the space into two halves.](../img/space-division-3d.svg)
:label:`fig_higher-division`

この時点では視覚化能力は尽きますが、数十、数百、数十億の次元でこれを行うことを妨げるものは何もありません。これは、機械学習モデルについて考えるときによく発生します。たとえば、:numref:`sec_softmax` のような線形分類モデルは、異なるターゲットクラスを分離する超平面を見つける方法として理解できます。この文脈では、このような超平面はしばしば「ディシジョンプレーン」と呼ばれます。ディープラーニングされた分類モデルの大半は、ソフトマックスに入力される線形層で終わるため、ディープニューラルネットワークの役割は、ターゲットクラスを超平面できれいに分離できるような非線形埋め込みを見つけることであると解釈できます。 

手作業で作成した例を挙げると、Fashion MNIST データセット (:numref:`sec_fashion_mnist` を参照) から T シャツとズボンの小さな画像を分類するための妥当なモデルを作成できることに注目してください。決定平面と眼球の平均値の間のベクトルを取り出して粗いしきい値を定義するだけです。まず、データをロードして平均を計算します。

```{.python .input}
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

これらの平均を詳細に調べることは有益なので、それらがどのように見えるかをプロットしてみましょう。この場合、平均はTシャツのぼやけた画像に似ていることがわかります。

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

2番目のケースでは、平均がズボンのぼやけたイメージに似ていることが再びわかります。

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

完全に機械学習されたソリューションでは、データセットからしきい値を学習します。今回は、トレーニングデータ上で見栄えの良い閾値を手作業で確認しただけです。

```{.python .input}
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean(predictions.type(y_test.dtype) == y_test, dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## 線形変換の幾何

:numref:`sec_linear-algebra` と上記の議論を通して、ベクトル、長さ、角度の幾何学をしっかりと理解できました。ただし、ここで説明を省略した重要なオブジェクトが1つあります。それは、行列で表される線形変換の幾何学的理解です。異なる可能性がある 2 つの高次元空間の間でデータを変換するために行列ができることを完全に内部化するには、かなりの実践が必要であり、この付録の範囲外です。しかし、私たちは二次元で直感を築き始めることができます。 

行列があるとします。 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

これを任意のベクトル $\mathbf{v} = [x, y]^\top$ に適用したい場合、乗算すると次のようになります。 

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

これは奇妙な計算のように思えるかもしれませんが、明確な何かがいくらか突き通せなくなった。しかし、それは $[1,0]^\top$ と $[0,1]^\top$ の 2 つの特定のベクトル* をどのように変換するかという点で、行列が*任意の*ベクトルを変換する方法を記述できることを示しています。これはちょっと考えてみる価値がある。無限問題（実数のペアに何が起こるか）を有限の問題（これらの特定のベクトルに何が起こるか）に本質的に減らしました。これらのベクトルは*basis* の例であり、空間内の任意のベクトルをこれらの*基底ベクトル*の加重和として書き込むことができます。 

特定の行列を使うとどうなるかを描きましょう 

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

特定のベクトル $\mathbf{v} = [2, -1]^\top$ を見ると、これは $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$ であることがわかります。したがって、行列 $A$ はこれを $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$ に送ります。このロジックを注意深く実行すると、たとえば、すべての整数ペアの点のグリッドを考慮すると、行列の乗算によってグリッドが歪んだり、回転したり、スケーリングされたりすることがありますが、グリッド構造は :numref:`fig_grid-transform` のとおりのままでなければなりません。 

![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/grid-transform.svg)
:label:`fig_grid-transform`

これは、行列で表される線形変換を内面化する上で最も重要な直感的なポイントです。行列は、空間のある部分を他の部分と異なる方法で歪めることができません。彼らができることは、空間上の元の座標を取り、それらをゆがめ、回転、スケーリングすることだけです。 

一部の歪みは深刻な場合があります。例えば、マトリックス 

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

は、2 次元平面全体を 1 本の線に圧縮します。このような変換の特定と操作は後のセクションのトピックですが、幾何学的には、これは上で見た変換のタイプとは根本的に異なることがわかります。たとえば、行列 $\mathbf{A}$ の結果を元のグリッドに「折り返す」ことができます。行列 $\mathbf{B}$ からの結果が得られないのは、ベクトル $[1,2]^\top$ がどこから来たのか分からないからです。$[1,1]^\top$ か $[0, -1]^\top$ ですか？ 

この図は$2\times2$行列のためのものですが、学んだ教訓をより高い次元へと導くことを妨げるものは何もありません。$[1,0, \ldots,0]$ のような同様の基底ベクトルを取り、行列がそれらをどこに送るかを見ると、行列の乗算がどのような次元空間で空間全体を歪ませているかを感じ始めることができます。 

## 線形依存性

行列をもう一度考えてみましょう。 

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

これにより、プレーン全体が圧縮され、1 本のライン $y = 2x$ に収まるようになります。今、疑問が生じます。マトリックス自体を見るだけでこれを検出できる方法はありますか？答えは、確かにできるということです。$\mathbf{b}_1 = [2,4]^\top$と$\mathbf{b}_2 = [-1, -2]^\top$を$\mathbf{B}$の2つの列としましょう。行列 $\mathbf{B}$ で変換されたものはすべて、$a_1\mathbf{b}_1 + a_2\mathbf{b}_2$ のように、行列の列の加重和として記述できることを思い出してください。これを*線形の組み合わせ*と呼びます。$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ ということは、これら 2 つの列の任意の線形結合を $\mathbf{b}_2$ のように完全に記述できることを意味します。 

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

これは、空間で一意の方向を定義していないため、ある意味では柱の 1 つが冗長であることを意味します。このマトリックスが平面全体を単一の線に折りたたむことはすでにわかっているので、これはあまり驚かないはずです。さらに、線形依存 $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ がこれを捉えていることがわかります。これを 2 つのベクトル間でより対称にするために、これを次のように記述します。 

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

一般に、係数 $a_1, \ldots, a_k$ *すべてがゼロではない* が存在する場合、ベクトル $\mathbf{v}_1, \ldots, \mathbf{v}_k$ の集合は*線形依存* であると言うので、 

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

この場合、ベクトルの 1 つを他のベクトルの組み合わせで解き、効果的に冗長化することができます。したがって、行列の列の線形依存性は、行列が空間をある程度低い次元に圧縮しているという事実を証明しています。線形依存性がなければ、ベクトルは*線形独立*であるとします。行列の列が線形独立している場合、圧縮は行われず、操作を元に戻すことができます。 

## ランク

一般的な $n\times m$ 行列がある場合、行列がどの次元空間にマッピングされるかを尋ねるのが妥当です。*rank*と呼ばれる概念が私たちの答えになります。前のセクションでは、線形依存は空間のより低い次元への圧縮を証明するものであり、これを使用してランクの概念を定義できることを指摘しました。特に、行列 $\mathbf{A}$ のランクは、すべての列のサブセットの中で線形独立列の最大数です。たとえば、マトリックスは次のようになります。 

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

は $\mathrm{rank}(B)=1$ です。2 つの列は線形従属ですが、どちらの列も線形従属ではないためです。もっと難しい例として、 

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

とは、$\mathbf{C}$ が 2 位であることを示しています。たとえば、最初の 2 つの列は線形独立していますが、3 つの列の 4 つのコレクションはいずれも従属しているためです。 

説明したように、この手順は非常に非効率的です。これは、与えられた行列の列のサブセットをすべて調べる必要があるため、列数が指数関数的になる可能性があります。後で、行列のランクを計算するためのより効率的な計算方法を見ていきますが、今のところ、概念が明確に定義されていることを確認し、意味を理解するにはこれで十分です。 

## 可逆性

上記で、線形に依存する列をもつ行列による乗算は元に戻せない、つまり常に入力を回復できる逆演算がないことを見てきました。ただし、フルランク行列（つまり、$n$ のランク $n$ の行列 $\mathbf{A}$）による乗算は、いつでも元に戻せるはずです。行列を考えてみましょう。 

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

対角線に沿って 1、それ以外はゼロをもつ行列ですこれを*identity* マトリックスと呼びます。これは、適用時にデータを変更しないままにするマトリックスです。行列 $\mathbf{A}$ が行ったことを元に戻す行列を見つけるには、行列 $\mathbf{A}^{-1}$ を求めます。 

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

これをシステムとして見ると、$n \times n$ の未知数 ($\mathbf{A}^{-1}$ のエントリ) と $n \times n$ の方程式 (積 $\mathbf{A}^{-1}\mathbf{A}$ のすべてのエントリと $\mathbf{I}$ のすべてのエントリの間で保持する必要のある等価性) があるので、一般的に解が存在することを期待する必要があります。実際、次のセクションでは、行列式がゼロでない限り解を見つけることができるという性質を持つ*行列式*と呼ばれる量を確認します。このような行列 $\mathbf{A}^{-1}$ を*逆行列と呼びます。たとえば、$\mathbf{A}$ が一般的な $2 \times 2$ 行列であるとします。 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

すると、逆が次のようになっていることがわかります。 

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

上記の式で与えられた逆数で乗算すると実際に機能することを確認することで、これを確認することができます。

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### 数学的問題行列の逆行列は理論上は有用ですが、実際には問題を解くために行列の逆行列を「使う」ことはほとんど望んでいないと言わざるを得ません。一般に、次のような一次方程式を解くための数値的に安定したアルゴリズムがあります。 

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

逆数を計算して乗算して得るよりも 

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

小さい数で除算すると数値が不安定になるのと同様に、低いランクに近い行列の逆転も起こります。 

また、行列 $\mathbf{A}$ は*sparse* であることが一般的です。つまり、行列にはゼロ以外の値が少数しか含まれていません。例を調べてみると、逆がスパースであるという意味ではないことがわかります。$\mathbf{A}$ が $1$ 百万行列 x $1$ 百万の行列で、ゼロ以外のエントリが $5$ 百万個しかない場合でも (したがって $5$ 百万個だけ格納する必要があります)、逆行列には通常、ほとんどすべてのエントリが負でないため、$1\text{M}^2$ エントリ、つまり $1$ 兆をすべて格納する必要があります。エントリー！ 

線形代数を扱うときに頻繁に遭遇する厄介な数値問題に深く掘り下げる時間はありませんが、注意して進めるタイミングについてある程度の直感を提供したいと考えています。一般に、実際には逆転を避けることは経験則です。 

## 行列式線形代数の幾何学的ビューは、*行列式* と呼ばれる基本量を直感的に解釈する方法を提供します。前のグリッドイメージで、領域がハイライト表示されているとします (:numref:`fig_grid-filled`)。 

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

ハイライトされた四角形を見てください。これは $(0, 1)$ と $(1, 0)$ で指定されたエッジをもつ正方形で、面積が 1 になります。$\mathbf{A}$ がこの正方形を変換すると、平行四辺形になることがわかります。この平行四辺形は、私たちが始めたのと同じ面積を持つべき理由はありません。実際、ここで示した特定のケースでは 

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

この平行四辺形の面積を計算し、その面積が$5$であることを求めるのは座標幾何学の練習です。 

一般に、マトリックスがあれば 

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

計算してみると、結果として得られる平行四辺形の面積は $ad-bc$ であることがわかります。この領域を*行列式*と呼びます。 

これをいくつかのサンプルコードですばやく確認してみましょう。

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

私たちの間でワシの目は、この表現がゼロになることも負になることもあることに気付くでしょう。負の項の場合、これは一般に数学で採用されている慣例の問題です。行列が図形を反転させると、面積は否定されます。行列式がゼロのとき、もっと学ぶことを見てみましょう。 

考えてみよう 

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

この行列の行列式を計算すると、$2\cdot(-2 ) - 4\cdot(-1) = 0$ が得られます。上記の理解を考えると、これは理にかなっています。$\mathbf{B}$ は、正方形を元のイメージから面積がゼロの線分に圧縮します。実際、低次元の空間に圧縮されることが、変換後に面積をゼロにする唯一の方法です。したがって、次の結果が真であることがわかります。行列 $A$ は、行列式がゼロに等しくない場合にのみ反転可能です。 

最後のコメントとして、飛行機に描かれた人物がいると想像してください。コンピューター科学者のように考えると、その図を小さな正方形の集まりに分解して、図の面積が本質的に分解の正方形の数だけになるようにすることができます。ここで、その図形を行列で変換すると、これらの各二乗を平行四辺形に送ります。平行四辺形は行列式によって与えられる面積を持ちます。どの図形についても、行列式は行列が任意の Figure の面積をスケーリングする (符号付き) 数を与えることがわかります。 

大きな行列の行列式の計算は手間がかかる場合がありますが、直感は同じです。行列式は $n\times n$ 行列が $n$ 次元の体積をスケーリングする因子のままです。 

## テンソルと一般的な線形代数演算

:numref:`sec_linear-algebra` では、テンソルの概念が導入されました。このセクションでは、テンソル収縮 (テンソルは行列の乗算に相当) についてさらに深く掘り下げ、行列とベクトル演算の数について統一されたビューを提供する方法を見ていきます。 

行列とベクトルでは、データを変換するためにそれらを乗算する方法を知っていました。テンソルが役に立つためには、テンソルについても同様の定義が必要です。行列の乗算について考えてみましょう。 

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

または同等に 

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

このパターンはテンソルに対して繰り返すことができます。テンソルの場合、普遍的に選択できるものを合計するケースは1つもないので、合計したいインデックスを正確に指定する必要があります。例えば、我々は考えることができる 

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

このような変換を*テンソル収縮* と呼びます。これは、乗算だけを行列化するはるかに柔軟な変換ファミリーを表すことができます。 

表記法の簡略化としてよく使われるように、式の中で複数回出現するインデックスの和が正確に超えていることがわかります。したがって、人々はしばしば、*アインシュタイン記法*を使って作業します。この場合、合計はすべての繰り返されるインデックスに対して暗黙的に引き継がれます。これにより、コンパクトな式が得られます。 

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### 線形代数の一般的な例

これまでに見た線形代数的定義のうち、この圧縮テンソル表記法で表現できる線形代数の定義がいくつあるか見てみましょう。 

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

このようにして、無数の特殊記法を短いテンソル式に置き換えることができます。 

### コードテンソルでの表現は，コード内でも柔軟に操作できます。:numref:`sec_linear-algebra` で見られるように、以下のようにテンソルを作成することができます。

```{.python .input}
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

Einstein の総和が直接実装されました。アインシュタインの総和で発生するインデックスは文字列として渡され、その後に作用するテンソルが続きます。たとえば、行列の乗算を実装するために、上記の Einstein 加算 ($\mathbf{A}\mathbf{v} = a_{ij}v_j$) を考慮し、インデックス自体を取り除いて実装を得ることができます。

```{.python .input}
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

これは非常に柔軟な表記法です。例えば、伝統的に何と書かれていたかを計算したいとします。 

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

Einstein の総和によって次のように実装できます。

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

この表記法は人間にとって読みやすく効率的ですが、何らかの理由でテンソル収縮をプログラムで生成する必要がある場合はかさばります。このため、`einsum` は、各テンソルに整数インデックスを指定することで、代替表記法を提供します。たとえば、同じテンソル収縮は次のように表すこともできます。

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow doesn't support this type of notation.
```

どちらの表記法でも、コード内でテンソル収縮を簡潔かつ効率的に表現できます。 

## 概要 * ベクトルは、空間内の点または方向として幾何学的に解釈できます。* ドット積は、任意の高次元空間に対する角度の概念を定義します。* 超平面は、線と平面を高次元で一般化したものです。これらは、分類タスクの最終ステップとしてよく使用される決定平面を定義するために使用できます。* 行列の乗算は、基礎となる座標の一様な歪みとして幾何学的に解釈できます。これらは、ベクトルを変換するための非常に制限された、しかし数学的にクリーンな方法を表しています。* 線形依存は、ベクトルの集合が予想よりも低い次元空間に存在することを知る方法です ($2$ 次元の空間に存在する $3$ 個のベクトルがあるとします)。行列のランクは、線形独立している列の中で最大のサブセットのサイズです。* 行列の逆行列が定義されると、行列の逆行列を使用すると、最初の行列の動作を取り消す別の行列を見つけることができます。行列反転は理論上は有用ですが、数値が不安定なため実際には注意が必要です。* 行列式により、行列が空間をどれだけ拡大または縮小するかを測定できます。非ゼロ行列式は可逆 (非特異な) 行列を意味し、ゼロ値の行列式は行列が非可逆 (特異数) であることを意味します。* テンソル収縮と Einstein 加算は、機械学習で見られる多くの計算を表現するための、すっきりとした簡潔な表記法を提供します。 

## 演習 1.$$\ vec v_1 =\ begin {bmatrix} 1\\ 0\\ -1\\ 2\ 終了 {bmatrix},\ quad\ vec v_2 =\ begin {bmatrix} 3\\ 1\\ 0\\ 1\ end {bmatrix} の間の角度は何ですか？$$2。正か偽か:$\ begin {bmatrix} 1 & 2\\ 0&1\ end {bmatrix} $ and $\ begin {bmatrix} 1 & -2\\ 0&1\ end {bmatrix} $ は互いに逆ですか？3。面積 $100\mathrm{m}^2$ の平面上にシェイプを描画するとします。行列 $$\ begin {bmatrix} 2 & 3\\ 1 & 2\ end {bmatrix} によって図形を変換した後の領域は何ですか。$$ 4.次のベクトルセットのうち、線形的に独立しているのはどれですか?$\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$ * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$$A =\ begin {bmatrix} c\\ d\ end {bmatrix}\ cdot\ begin {bmatrix} a & b\ end {bmatrix} $ for some choice of values $a, b, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0 $？6。ベクトル $e_1 =\ begin {bmatrix} 1\\ 0\ end {bmatrix} $ and $e_2 =\ begin {bmatrix} 0\\ 1\ end {bmatrix} $ は直交しています。$Ae_1$ と $Ae_2$ が直交するように、行列 $A$ の条件は何ですか？7。任意行列 $A$ に対して $\mathrm{tr}(\mathbf{A}^4)$ をアインシュタイン記法でどのように書くことができますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:

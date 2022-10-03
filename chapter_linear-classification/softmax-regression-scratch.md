```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Softmax 回帰のゼロからの実装
:label:`sec_softmax_scratch`

ソフトマックス回帰はとても基本的なものなので、自分で実装する方法を知っておくべきだと考えています。ここでは、モデルのソフトマックス固有の側面の定義に限定し、トレーニングループを含む線形回帰セクションの他のコンポーネントを再利用します。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## ザ・ソフトマックス

まず、最も重要な部分、つまりスカラーから確率へのマッピングから始めましょう。復習では、:numref:`subsec_lin-alg-reduction`と:numref:`subsec_lin-alg-non-reduction`で説明されているように、テンソルの特定の次元に沿った合計演算子の演算を思い出してください。[**行列 `X` を指定すると、すべての要素 (デフォルト) を合計するか、同じ軸の要素のみを合計できます**] `axis` 変数を使用すると、行と列の合計を計算できます。

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

ソフトマックスの計算には 3 つのステップが必要です。(i) 各項のべき乗、(ii) 各例の正規化定数を計算するための各行の合計、(iii) 各行を正規化定数で除算し、結果の合計が 1 になるようにします。 

(** $\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$ドル)
**)

分母 (の対数) は (log) *パーティション関数* と呼ばれます。これは [統計物理学](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) で導入され、熱力学的アンサンブルのすべての可能な状態を合計しました。実装は簡単です。

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

どの入力でも `X`、[**各要素を非負の数に変換します。各行は、確率に必要な最大で 1**] になります。注意:上記のコードは、非常に大きな引数や非常に小さな引数に対して堅牢ではありません。何が起こっているのかを説明するにはこれで十分ですが、このコードを重大な目的にはそのまま使用しないでください。ディープラーニングフレームワークにはこのような保護機能が組み込まれており、今後は組み込みのソフトマックスを使用する予定です。

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## ザ・モデル

We now have everything that we need
to implement [**the softmax regression model.**]
As in our linear regression example,
each instance will be represented
by a fixed-length vector.
Since the raw data here consists
of $28 \times 28$ pixel images,
[**we flatten each image,
treating them as vectors of length 784.**]
In later chapters, we will introduce
convolutional neural networks,
which exploit the spatial structure
in a more satisfying way.


In softmax regression,
the number of outputs from our network
should be equal to the number of classes.
(**Since our dataset has 10 classes,
our network has an output dimension of 10.**)
Consequently, our weights constitute a $784 \times 10$ matrix
plus a $1 \times 10$ dimensional row vector for the biases.
As with linear regression,
we initialize the weights `W`
with Gaussian noise.
The biases are initialized as zeros.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

以下のコードは、ネットワークが各入力を出力にどのようにマッピングするかを定義しています。データをモデルに渡す前に、バッチ内の各 $28 \times 28$ ピクセルイメージを `reshape` を使用してベクトルにフラット化することに注意してください。

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    return softmax(d2l.matmul(d2l.reshape(
        X, (-1, self.W.shape[0])), self.W) + self.b)
```

## クロスエントロピー損失

次に、クロスエントロピー損失関数 (:numref:`subsec_softmax-regression-loss-func` で導入) を実装する必要があります。これは、すべてのディープラーニングで最も一般的な損失関数かもしれません。現時点では、ディープラーニングのアプリケーションは、回帰問題としてより適切に扱われる分類問題をはるかに上回っています。 

クロスエントロピーは、真のラベルに割り当てられた予測確率の負の対数尤度を取ることを思い出してください。効率化のため、Python の for ループを避け、代わりにインデックスを使用します。特に、$\mathbf{y}$のワンホットエンコーディングでは、$\hat{\mathbf{y}}$で一致する用語を選択できます。 


To see this in action we [**create sample data `y_hat`
with 2 examples of predicted probabilities over 3 classes and their corresponding labels `y`.**]
The correct labels are $1$ and $2$ respectively.
[**Using `y` as the indices of the probabilities in `y_hat`,**]
we can pick out terms efficiently.

```{.python .input}
%%tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

これで、選択した確率の対数を平均化することで (**クロスエントロピー損失関数を実装**) できます。

```{.python .input}
%%tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.reduce_mean(d2l.log(y_hat[range(len(y_hat)), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return - tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

## トレーニング

:numref:`sec_linear_scratch` で定義された `fit` メソッドを再利用して [**10 エポックでモデルをトレーニングします。**] エポック数 (`max_epochs`)、ミニバッチサイズ (`batch_size`)、学習率 (`lr`) はどちらも調整可能なハイパーパラメータであることに注意してください。つまり、これらの値は主要なトレーニングループでは学習されませんが、トレーニングとジェネラライズのパフォーマンスの両方に対して、モデルのパフォーマンスに影響を与えます。実際には、データの*検証*分割に基づいてこれらの値を選択し、最終的に*テスト*分割で最終モデルを評価します。:numref:`subsec_generalization-model-selection` で説明したように、Fashion-MNIST のテストデータを検証セットとして扱い、この分割の検証損失と検証精度を報告します。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 予測

これでトレーニングが完了し、モデルが [**いくつかの画像を分類する**] 準備が整いました。

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
preds = d2l.argmax(model(X), axis=1)
preds.shape
```

私たちは、*間違って*ラベル付けした画像にもっと関心があります。実際のラベル (テキスト出力の1行目) とモデルからの予測 (テキスト出力の2行目) を比較して視覚化します。

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## まとめ

今では、線形回帰と分類の問題の解決についてある程度の経験を積み始めています。これにより、1960〜1970年代の統計モデリングの最先端であると思われるものに到達しました。次のセクションでは、ディープラーニングフレームワークを活用してこのモデルをより効率的に実装する方法を説明します。 

## 演習

1. このセクションでは、softmax演算の数学的定義に基づいて、softmax関数を直接実装しました。:numref:`sec_softmax`で説明したように、これは数値の不安定性を引き起こす可能性があります。
    1. 入力の値が$100$の場合でも、`softmax`が正しく機能するかどうかをテストしますか？
    1. 全入力のうち最大値が$-100$より小さい場合でも、`softmax`が正しく動作するかどうかテストしますか？
    1. 引数の最大のエントリに対する相対的な値を見て、修正を実装します。
1. クロスエントロピー損失関数 $\sum_i y_i \log \hat{y}_i$ の定義に従う `cross_entropy` 関数を実装します。
    1. 上記のコード例で試してみてください。
    1. なんでもっとゆっくり走ると思う？
    1. それを使うべきか？どのような場合に意味がありますか？
    1. 注意すべきことは何ですか？ヒント:対数のドメインを考えてみましょう。
1. 最も可能性の高いラベルを返品するのは常に良い考えですか？例えば、医学的診断のためにこれをしますか？この件にどう対処しようと思う？
1. ソフトマックス回帰を使用して、いくつかの特徴に基づいて次の単語を予測するとします。大きな語彙から生じる可能性のある問題は何ですか？
1. 上記のコードのハイパーパラメータを試してみてください。特に:
    1. 学習率の変化に伴って検証損失がどのように変化するかをプロットします。
    1. ミニバッチのサイズを変更すると、検証と学習の損失も変化しますか?効果が出る前にどれくらいの大きさ、それとも小さくなる必要がありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:

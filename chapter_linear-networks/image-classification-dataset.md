# 画像分類データセット
:label:`sec_fashion_mnist`

(~~MNIST データセットは画像分類に広く使われているデータセットの一つですが、ベンチマークデータセットとしてはシンプルすぎます。似ているがもっと複雑な Fashion-MNIST データセットを使います~~) 

画像分類に広く使用されているデータセットの 1 つに、MNIST データセット :cite:`LeCun.Bottou.Bengio.ea.1998` があります。ベンチマークデータセットとしては好調でしたが、今日の標準では単純なモデルでも 95% を超える分類精度が得られ、強いモデルと弱いモデルの区別には不向きです。現在、MNISTはベンチマークというよりもサニティチェックの役割を果たしています。少し前置きにするために、2017年にリリースされた、質的に類似しているが比較的複雑なファッションMNISTデータセット:cite:`Xiao.Rasul.Vollgraf.2017`に関する次のセクションで議論を集中します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## データセットの読み取り

[**フレームワークの組み込み関数を使用して、Fashion-MNIST データセットをダウンロードしてメモリに読み込む**]

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST は 10 個のカテゴリの画像で構成され、それぞれがトレーニングデータセットでは 6000 個、テストデータセットでは 1000 個の画像で表されます。*test dataset* (または*test set*) は、トレーニングではなくモデルの性能を評価するために使用されます。したがって、トレーニングセットとテストセットにはそれぞれ 60000 と 10000 のイメージが含まれます。

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

各入力イメージの高さと幅はどちらも 28 ピクセルです。データセットは、チャンネル数が 1 のグレースケールイメージで構成されていることに注意してください。簡潔にするために、本書では、高さ $h$、幅 $w$ ピクセルの画像の形状を $h \times w$ または ($h$, $w$) として保存しています。

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

[~~データセットを可視化する2つのユーティリティ関数~~] 

Fashion-mnistの画像は次のカテゴリに関連付けられています：Tシャツ、ズボン、プルオーバー、ドレス、コート、サンダル、シャツ、スニーカー、バッグ、アンクルブーツ。次の関数は、数値ラベルインデックスとテキスト内の名前との間で変換を行います。

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

これで、これらの例を可視化する関数を作成できるようになりました。

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

トレーニングデータセットの最初のいくつかの例の [**画像とそれに対応するラベル**](本文) を以下に示します。

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## ミニバッチの読み方

トレーニングセットとテストセットから読みやすくなるように、ゼロからデータイテレーターを作成するのではなく、組み込みのデータイテレーターを使用します。繰り返しのたびに、データイテレータ [**サイズが `batch_size` のデータのミニバッチを毎回読み取ります。**] また、学習データイテレータの例をランダムにシャッフルします。

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

トレーニングデータを読み取るのにかかる時間を見てみましょう。

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## すべてのものをまとめる

ここで、[**Fashion-MNIST データセットを取得して読み込む `load_data_fashion_mnist` 関数**] を定義します。この関数は、トレーニングセットと検証セットの両方のデータイテレータを返します。また、イメージを別のシェイプにサイズ変更するオプションの引数も使用できます。

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

以下では、`resize` 引数を指定して `load_data_fashion_mnist` 関数のイメージサイズ変更機能をテストします。

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

これで、以降のセクションで Fashion-MNIST データセットを操作する準備が整いました。 

## [概要

* Fashion-MNIST は、10 種類のカテゴリを表す画像で構成されるアパレル分類データセットです。このデータセットを以降のセクションと章で使用して、さまざまな分類アルゴリズムを評価します。
* 高さが$h$、幅が $w$ ピクセルのイメージのシェイプは、$h \times w$ または ($h$, $w$) として格納されます。
* データイテレータは、パフォーマンスを効率化するための重要な要素です。トレーニングループの速度を低下させないように、ハイパフォーマンスコンピューティングを利用する、適切に実装されたデータイテレーターを利用してください。

## 演習

1. `batch_size` (たとえば 1) を減らすと、読み取りのパフォーマンスに影響しますか?
1. データイテレータのパフォーマンスは重要です。現在の実装は十分速いと思いますか？それを改善するためのさまざまなオプションを探る。
1. フレームワークのオンライン API ドキュメントを確認してください。他にどのようなデータセットがありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:

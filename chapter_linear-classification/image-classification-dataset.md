```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 画像分類データセット
:label:`sec_fashion_mnist`

(~~ MNIST データセットは、画像分類に広く使用されているデータセットの 1 つですが、ベンチマークデータセットとしてはあまりにも単純です。似ているがもっと複雑なFashion-MNISTデータセットを使用します~~) 

画像分類に広く使用されているデータセットの1つは、手書き数字の [MNISTデータセット]（https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`）です。1990年代にリリースされた時点では、$28 \times 28$ピクセルの解像度の60,000枚の画像（および10,000枚の画像のテストデータセット）で構成されるほとんどの機械学習アルゴリズムに手ごわい課題がありました。物事を展望すると、当時、なんと64MBのRAMと5mFlopsの猛烈な5MFLOPSを備えたSun SparcStation 5は、1995年にAT＆Tベル研究所で機械学習のための最先端の機器と見なされていました。数字認識の高精度を達成することは、1990年代のUSPSの文字ソートを自動化するための重要な要素でした。LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`、不変性を持つサポートベクターマシン :cite:`Scholkopf.Burges.Vapnik.1996`、接線距離分類器 :cite:`Simard.LeCun.Denker.ea.1998` などのディープネットワークはすべて、1% 未満の誤り率に達することができました。  

10年以上にわたり、MNISTは機械学習アルゴリズムを比較するための*基準点*としての役割を果たしました。ベンチマークデータセットとしては好調でしたが、今日の標準による単純なモデルでも 95% を超える分類精度が得られるため、強いモデルと弱いモデルを区別するのには適していません。さらに、このデータセットは、多くの分類問題では一般的に見られない「非常に」高いレベルの精度を可能にします。このアルゴリズム開発は、アクティブセットメソッドや境界探索アクティブセットアルゴリズムなど、クリーンなデータセットを利用できる特定のアルゴリズムファミリーに偏っていました。今日、MNISTはベンチマークとしてよりも健全性チェックの役割を果たしています。ImageNET :cite:`Deng.Dong.Socher.ea.2009` は、はるかに重要な課題を提起します。残念ながら、ImageNetは、この本の多くの例やイラストには大きすぎます。例をインタラクティブにするにはトレーニングに時間がかかりすぎるからです。代替として、2017年にリリースされた、質的に類似しているがはるかに小さいFashion-MNISTデータセット:cite:`Xiao.Rasul.Vollgraf.2017`について、今後のセクションで議論に焦点を当てます。これには、$28 \times 28$ピクセルの解像度で10種類の衣類の画像が含まれています。

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## データセットの読み込み

これは頻繁に使用されるデータセットであるため、すべての主要なフレームワークは前処理されたバージョンを提供します。[**組み込みのフレームワーク関数を使用して、Fashion-Mnist データセットをダウンロードしてメモリに読み込むことができます。**]

```{.python .input  n=5}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input  n=6}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input  n=7}
%%tab tensorflow
class FashionMNIST(d2l.DataModule):  #@save
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-mnist は 10 のカテゴリの画像で構成され、それぞれがトレーニングデータセットの 6,000 枚の画像、テストデータセットの 1,000 枚の画像で表されます。*テストデータセット*は、モデルの性能を評価するために使用されます（トレーニングには使用しないでください）。その結果、トレーニングセットとテストセットにはそれぞれ60,000と10,000の画像が含まれます。

```{.python .input  n=8}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input  n=9}
%%tab tensorflow
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

画像はグレースケールで、上記の解像度で$32 \times 32$ピクセルにアップスケールされています。これは、（バイナリ）白黒画像で構成された元のMNISTデータセットに似ています。ただし、最新の画像データには3チャンネル（赤、緑、青）があり、ハイパースペクトル画像は100チャンネルを超える場合があります（HyMapセンサーには126チャンネルあります）。慣例により、画像を$c \times h \times w$テンソルとして保存します。ここで、$c$はカラーチャンネルの数、$h$は高さ、$w$は幅です。

```{.python .input  n=10}
%%tab all
data.train[0][0].shape
```

[~~データセットを可視化する2つのユーティリティ関数~~] 

Fashion-mnistのカテゴリーには、人間が理解できる名前があります。次の便利な関数は、数値ラベルとその名前を変換します。

```{.python .input  n=11}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## ミニバッチを読み取る

トレーニングセットとテストセットから読み取るときの生活を楽にするために、ゼロから作成するのではなく、組み込みのデータイテレーターを使用します。各反復で、データイテレータ [**サイズ `batch_size`.のデータのミニバッチを読み取る**] を思い出してください。また、トレーニングデータイテレータの例をランダムにシャッフルします。

```{.python .input  n=12}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input  n=13}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input  n=14}
%%tab tensorflow
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
        self.batch_size).map(resize_fn).shuffle(shuffle_buf)
```

これがどのように機能するかを確認するために、新しく追加された`train_dataloader`メソッドを呼び出して画像のミニバッチをロードしましょう。64枚の画像が含まれています。

```{.python .input  n=15}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

画像を読むのにかかる時間を見てみましょう。組み込みのローダーですが、驚くほど高速ではありません。それでも、ディープネットワークでの画像の処理にはかなり時間がかかるため、これで十分です。したがって、ネットワークのトレーニングが IO の制約を受けないほど十分です。

```{.python .input  n=16}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## 視覚化

Fashion-mnist データセットをかなり頻繁に使用します。便利な機能`show_images`を使用して、画像と関連するラベルを視覚化できます。その実装の詳細は付録に延期されています。

```{.python .input  n=17}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

それを有効に活用しよう。一般に、トレーニング中のデータを視覚化して調査することをお勧めします。人間は異常な側面を見つけるのが非常に得意であるため、視覚化は実験計画における間違いや誤りに対する追加の保護手段として機能します。トレーニングデータセットの最初のいくつかの例の [**画像とそれに対応するラベル**]（本文）を以下に示します。

```{.python .input  n=18}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet') or tab.selected('pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

これで、次のセクションで Fashion-mnist データセットを使用する準備が整いました。 

## まとめ

これで、分類に使用する、もう少し現実的なデータセットができました。Fashion-mnist は、10 のカテゴリを表す画像で構成されるアパレル分類データセットです。このデータセットを以降のセクションと章で使用して、単純な線形モデルから高度な残差ネットワークまで、さまざまなネットワーク設計を評価します。画像でよく行うように、それらを形状のテンソル（バッチサイズ、チャンネル数、高さ、幅）として読み取ります。今のところ、画像はグレースケールであるため、チャンネルは1つだけです（上記の視覚化では、視認性を向上させるために偽のカラーパレットを使用しています）。  

最後に、データイテレータは効率的なパフォーマンスの重要なコンポーネントです。たとえば、効率的な画像解凍、ビデオトランスコーディング、またはその他の前処理にGPUを使用する場合があります。可能な限り、トレーニングループの速度を落とさないように、ハイパフォーマンスコンピューティングを活用する適切に実装されたデータイテレータに頼るべきです。 

## 演習

1. `batch_size` を (たとえば 1 に) 下げると、読み取りパフォーマンスに影響しますか?
1. データイテレータのパフォーマンスは重要です。現在の実装は十分速いと思いますか？それを改善するためのさまざまなオプションを検討してください。システムプロファイラを使用して、ボトルネックがどこにあるかを調べます。
1. フレームワークのオンライン API ドキュメントを確認してください。他に利用できるデータセットはどれですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:

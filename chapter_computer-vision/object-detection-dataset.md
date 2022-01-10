# 物体検出データセット
:label:`sec_object-detection-dataset`

物体検出の分野では、MNISTやFashion-MNISTのような小さなデータセットはありません。物体検出モデルのデモンストレーションを迅速に行うため、[**小さなデータセットを収集してラベル付けしました**]。まず、オフィスから無料のバナナの写真を撮り、回転や大きさの異なる1000枚のバナナ画像を生成しました。次に、各バナナの画像を背景画像のランダムな位置に配置しました。最後に、これらのバナナのバウンディングボックスのラベルを画像上に付けました。 

## [**データセットのダウンロード**]

すべての画像とCSVラベルファイルを含むバナナ検出データセットは、インターネットから直接ダウンロードできます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## データセットの読み取り

以下の `read_data_bananas` 関数の [**バナナ検出データセットの読み取り**] を行います。データセットには、左上隅と右下隅にオブジェクトクラスラベルとグラウンドトゥルース境界ボックス座標の csv ファイルが含まれています。

```{.python .input}
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

`read_data_bananas` 関数を使用して画像とラベルを読み取ることにより、次の `BananasDataset` クラスでは、バナナ検出データセットをロードするための [**カスタマイズされた `Dataset` インスタンスを作成**] できます。

```{.python .input}
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最後に、`load_data_bananas` 関数を定義して [**トレーニングセットとテストセットの両方で 2 つのデータイテレーターインスタンスを返す**] テストデータセットの場合、ランダムな順序で読み取る必要はありません。

```{.python .input}
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

このミニバッチで [**ミニバッチを読んで、画像とラベルの両方の形状を印刷**] してみましょう。画像ミニバッチの形状 (バッチサイズ、チャンネル数、高さ、幅) は見覚えがあり、以前の画像分類タスクと同じです。ラベルミニバッチの形状は (バッチサイズ $m$, 5) です。$m$ は、データセット内の画像の境界ボックスの最大数です。 

ミニバッチでの計算の方が効率的ですが、連結によってミニバッチを形成するには、すべてのイメージ例に同じ数の境界ボックスが含まれている必要があります。一般に、イメージのバウンディングボックスの数はさまざまです。したがって、バウンディングボックスが $m$ 未満のイメージは $m$ に達するまで不正なバウンディングボックスで埋められます。各境界ボックスのラベルは、長さ 5 の配列で表されます。配列の最初の要素はバウンディングボックス内のオブジェクトのクラスで、-1 は不正なパディング用のバウンディングボックスを示します。配列の残りの 4 つの要素は、境界ボックスの左上隅と右下隅の ($x$, $y$) 座標値です (範囲は 0 ～ 1)。banana データセットでは、各画像に境界ボックスが 1 つしかないため、$m=1$ になります。

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**デモンストレーション**]

グラウンドトゥルースの境界ボックスがラベル付けされた 10 個のイメージについて説明しましょう。バナナの回転、大きさ、位置は、これらすべての画像で異なることがわかります。もちろん、これは単純な人工データセットです。実際には、通常、実世界のデータセットははるかに複雑です。

```{.python .input}
imgs = (batch[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## [概要

* 収集したバナナ検出データセットは、物体検出モデルのデモンストレーションに使用できます。
* 物体検出用のデータ読み込みは、画像分類の場合と似ています。ただし、物体検出では、ラベルにはグラウンドトゥルース境界ボックスの情報も含まれており、画像分類では欠落しています。

## 演習

1. バナナ検出データセットのグラウンドトゥルース境界ボックスを使用して他の画像を実演します。バウンディングボックスとオブジェクトではどう違うのですか？
1. たとえば、ランダムクロッピングなどのデータ拡張をオブジェクト検出に適用するとします。画像分類とどう違うのですか？ヒント:クロップされた画像にオブジェクトのごく一部しか含まれていない場合はどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:

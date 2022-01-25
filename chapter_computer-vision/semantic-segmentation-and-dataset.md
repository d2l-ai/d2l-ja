# セマンティックセグメンテーションとデータセット
:label:`sec_semantic_segmentation`

:numref:`sec_bbox`—:numref:`sec_rcnn` でオブジェクト検出タスクについて説明する場合、イメージ内のオブジェクトにラベルを付けて予測するために、長方形の境界ボックスが使用されます。このセクションでは、イメージを異なるセマンティッククラスに属する領域に分割する方法に焦点を当てた、*セマンティックセグメンテーション* の問題について説明します。オブジェクト検出とは異なり、セマンティックセグメンテーションは画像に含まれるものをピクセルレベルで認識して理解します。セマンティック領域のラベル付けと予測はピクセルレベルで行われます。:numref:`fig_segmentation` は、画像の犬、猫、背景のラベルをセマンティックセグメンテーションで表示します。オブジェクト検出と比較すると、セマンティックセグメンテーションでラベル付けされたピクセルレベルの境界は、明らかにきめ細かいです。 

![Labels of the dog, cat, and background of the image in semantic segmentation.](../img/segmentation.svg)
:label:`fig_segmentation`

## イメージセグメンテーションとインスタンスセグメンテーション

また、コンピュータビジョンの分野では、セマンティックセグメンテーションに似た、イメージセグメンテーションとインスタンスセグメンテーションという2つの重要なタスクがあります。以下のように、セマンティックセグメンテーションと簡単に区別します。 

* *イメージセグメンテーション* は、イメージを複数の構成領域に分割します。この種の問題の方法は、通常、イメージ内のピクセル間の相関を利用します。学習中はイメージピクセルに関するラベル情報を必要とせず、セグメント化された領域が予測中に取得したいセマンティクスを持つことを保証できません。:numref:`fig_segmentation` の画像を入力とすると、画像セグメンテーションは犬を 2 つの領域に分割します。1 つは口と目 (主に黒)、もう 1 つは体の残りの部分 (主に黄色) を覆います。
* *インスタンスセグメンテーション* は、*同時検出とセグメンテーション* とも呼ばれます。画像内の各オブジェクトインスタンスのピクセルレベルの領域を認識する方法を学習します。セマンティックセグメンテーションとは異なり、インスタンスセグメンテーションではセマンティクスだけでなく異なるオブジェクトインスタンスも区別する必要があります。たとえば、イメージに 2 匹の犬がある場合、インスタンスセグメンテーションでは、ピクセルが 2 匹の犬のどちらに属しているかを区別する必要があります。

## Pascal VOC2012 セマンティックセグメンテーションデータセット

[**最も重要なセマンティックセグメンテーションデータセットは[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**] 以下では、このデータセットを見ていきます。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

データセットの tar ファイルは約 2 GB なので、ファイルのダウンロードに時間がかかる場合があります。抽出されたデータセットは `../data/VOCdevkit/VOC2012` にあります。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

パス `../data/VOCdevkit/VOC2012` を入力すると、データセットのさまざまなコンポーネントが表示されます。`ImageSets/Segmentation` パスには学習サンプルとテストサンプルを指定するテキストファイルが含まれ、`JPEGImages` パスと `SegmentationClass` パスには各例の入力イメージとラベルがそれぞれ格納されます。このラベルもイメージ形式で、ラベル付けされた入力イメージと同じサイズです。また、ラベルイメージ内の同じ色のピクセルは、同じセマンティッククラスに属します。次の例では、[**すべての入力イメージとラベルをメモリに読み込む**] する `read_voc_images` 関数を定義します。

```{.python .input}
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

[**最初の 5 つの入力イメージとそのラベルを描画します**]。ラベルイメージでは、白と黒はそれぞれ境界線と背景を表し、他の色は異なるクラスに対応しています。

```{.python .input}
n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

次に、このデータセット内のすべてのラベルについて [**RGB カラー値とクラス名を列挙**] します。

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

上記で定義した 2 つの定数を使えば、[**ラベル内の各ピクセルのクラスインデックスを求める**] と便利です。上記の RGB カラー値からクラスインデックスへのマッピングを構築する `voc_colormap2label` 関数と、この Pascal VOC2012 データセットのクラスインデックスに任意の RGB 値をマッピングする関数 `voc_label_indices` を定義します。

```{.python .input}
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**たとえば**]、最初の例の画像では、飛行機の前部のクラスインデックスは 1 で、背景インデックスは 0 です。

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### データ前処理

:numref:`sec_alexnet`—:numref:`sec_googlenet` などの以前の実験では、イメージはモデルに必要な入力形状に合わせて再スケーリングされます。ただし、セマンティックセグメンテーションでは、予測されたピクセルクラスを入力イメージの元の形状に再スケーリングし直す必要があります。このような再スケーリングは、特にクラスが異なるセグメント化された領域では不正確になる場合があります。この問題を回避するために、画像は再スケーリングではなく*固定*シェイプにトリミングされます。具体的には、[**イメージオーグメンテーションからのランダムクロップを使用して、入力イメージとラベルの同じ領域をクロップ**] とします。

```{.python .input}
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**カスタムセマンティックセグメンテーションデータセットクラス**]

高レベル API が提供する `Dataset` クラスを継承して、カスタムセマンティックセグメンテーションデータセットクラス `VOCSegDataset` を定義します。`__getitem__` 関数を実装することで、データセット内の `idx` としてインデックス付けされた入力イメージと、このイメージ内の各ピクセルのクラスインデックスに任意にアクセスできます。データセット内の一部の画像はランダムクロップの出力サイズよりも小さいので、これらの例はカスタム `filter` 関数によって除外されます。また、入力イメージの 3 つの RGB チャンネルの値を標準化する関数 `normalize_image` も定義します。

```{.python .input}
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**データセットの読み取り**]

カスタム `VOCSegDatase`t クラスを使用して、トレーニングセットとテストセットのインスタンスをそれぞれ作成します。ランダムにトリミングされたイメージの出力形状を $320\times 480$ と指定するとします。以下に、トレーニングセットとテストセットに保持されている例の数を示します。

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

バッチサイズを 64 に設定し、トレーニングセットのデータイテレーターを定義します。最初のミニバッチの形状を印刷してみましょう。画像分類や物体検出とは異なり、ラベルは三次元テンソルです。

```{.python .input}
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**すべてのものをまとめる**]

最後に、Pascal VOC2012 セマンティックセグメンテーションデータセットをダウンロードして読み取るために、次の `load_data_voc` 関数を定義します。トレーニングデータセットとテストデータセットの両方のデータイテレーターを返します。

```{.python .input}
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## [概要

* セマンティックセグメンテーションは、イメージを異なるセマンティッククラスに属する領域に分割することによって、イメージ内の内容をピクセルレベルで認識して理解します。
* 最も重要なセマンティックセグメンテーションデータセットの 1 つが Pascal VOC2012 です。
* セマンティックセグメンテーションでは、入力イメージとラベルはピクセル上で 1 対 1 で対応するため、入力イメージは再スケーリングされるのではなく、固定された形状にランダムにトリミングされます。

## 演習

1. セマンティックセグメンテーションは、自律走行車や医用画像診断にどのように適用できるのでしょうか？他のアプリケーションも考えられますか？
1. :numref:`sec_image_augmentation` のデータ拡張の説明を思い出してください。画像分類で使用される画像拡張手法のうち、セマンティックセグメンテーションに適用することが不可能なのはどれですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:

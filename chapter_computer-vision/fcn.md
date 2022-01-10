# 完全畳み込みネットワーク
:label:`sec_fcn`

:numref:`sec_semantic_segmentation` で説明したように、セマンティックセグメンテーションはイメージをピクセルレベルで分類します。完全畳み込みネットワーク (FCN) は、畳み込みニューラルネットワークを使用してイメージピクセルをピクセルクラス :cite:`Long.Shelhamer.Darrell.2015` に変換します。画像分類や物体検出のために以前に遭遇した CNN とは異なり、完全畳み込みネットワークは中間特徴マップの高さと幅を入力画像の高さと幅に戻します。これは :numref:`sec_transposed_conv` で導入された転置畳み込み層によって実現されます。その結果、分類出力と入力イメージはピクセルレベルが 1 対 1 で対応します。つまり、出力ピクセルのチャネル次元は、入力ピクセルの分類結果を同じ空間位置で保持します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## ザ・モデル

ここでは、完全畳み込みネットワークモデルの基本設計について説明します。:numref:`fig_fcn` に示すように、このモデルはまず CNN を使用して画像の特徴を抽出し、次に $1\times 1$ 畳み込み層を介してチャネル数をクラス数に変換し、最後に導入された転置畳み込みによって特徴マップの高さと幅を入力イメージの高さと幅に変換します。:numref:`sec_transposed_conv`に書かれていますその結果、モデル出力の高さと幅は入力イメージと同じになり、出力チャネルには同じ空間位置にある入力ピクセルの予測クラスが格納されます。 

![Fully convolutional network.](../img/fcn.svg)
:label:`fig_fcn`

以下では、[**ImageNet データセットで事前学習された ResNet-18 モデルを使用して画像の特徴を抽出する**]、モデルインスタンスを `pretrained_net` と表します。このモデルの最後の数層には、グローバル平均プーリング層と全結合層が含まれており、完全畳み込みネットワークでは必要ありません。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

次に、[**完全畳み込みネットワークインスタンス `net`**] を作成します。最終的なグローバル平均プーリング層と出力に最も近い全結合層を除くすべての事前学習済み層が ResNet-18 でコピーされます。

```{.python .input}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

高さと幅がそれぞれ 320 と 480 の入力がある場合、`net` の前方伝播により、入力の高さと幅は元の入力の 1/32、つまり 10 と 15 に減少します。

```{.python .input}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

次に、[**$1\times 1$ 畳み込み層を使用して、出力チャネル数を Pascal VOC2012 データセットのクラス数 (21) に変換します。**] 最後に、入力イメージの高さと幅に戻すために (**特徴マップの高さと幅を 32 倍に増やす**) 必要があります。:numref:`sec_padding` の畳み込み層の出力形状を計算する方法を思い出してください。$(320-64+16\times2+32)/32=10$ と $(480-64+16\times2+32)/32=15$ 以降、ストライドが $32$ の転置畳み込み層を構築し、カーネルの高さと幅を $64$、パディングを $16$ に設定しています。一般に、ストライド $s$、$s/2$ のパディング ($s/2$ が整数であると仮定)、カーネル $2s$ の高さと幅では、転置畳み込みによって入力の高さと幅が $s$ 倍増加することがわかります。

```{.python .input}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**転置畳み込み層の初期化**]

転置畳み込み層は特徴マップの高さと幅を広げることができることは既にわかっています。画像処理では、*アップサンプリング* のように、画像を拡大する必要がある場合があります。
*共一次内挿法*
は、よく使用されるアップサンプリング手法の 1 つです。転置畳み込み層の初期化にもよく使われます。 

共一次内挿法を説明するために、入力イメージを指定して、アップサンプリングされた出力イメージの各ピクセルを計算するとします。座標 $(x, y)$ の出力イメージのピクセルを計算するには、まず $(x, y)$ を入力イメージの座標 $(x', y')$ に、たとえば入力サイズと出力サイズの比率に従ってマッピングします。マップされた $x′$ and $y′$ は実数であることに注意してください。次に、入力イメージ上の座標 $(x', y')$ に最も近い 4 つのピクセルを求めます。最後に、入力イメージ上のこれら 4 つの最も近い 4 つのピクセルと $(x', y')$ からの相対距離に基づいて、座標 $(x, y)$ の出力イメージのピクセルが計算されます。  

共一次内挿のアップサンプリングは、次の `bilinear_kernel` 関数によって構築されたカーネルをもつ転置畳み込み層によって実装できます。スペースの制限により、以下では `bilinear_kernel` 関数の実装のみを提供し、アルゴリズムの設計については説明しません。

```{.python .input}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

転置畳み込み層によって実装される [**共一次内挿のアップサンプリングを実験**] してみましょう。高さと重みを 2 倍にする転置畳み込み層を構築し、そのカーネルを関数 `bilinear_kernel` で初期化します。

```{.python .input}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

イメージ `X` を読み取り、アップサンプリング出力を `Y` に割り当てます。画像を印刷するには、チャンネルの寸法の位置を調整する必要があります。

```{.python .input}
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

ご覧のとおり、転置畳み込み層はイメージの高さと幅の両方を 2 倍に増やします。座標の縮尺が異なる場合を除き、共一次内挿法で拡大されたイメージと :numref:`sec_bbox` で印刷された元のイメージは同じように見えます。

```{.python .input}
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

完全畳み込みネットワークでは、転置畳み込み層を双一次内挿のアップサンプリングで初期化します。$1\times 1$ 畳み込み層には Xavier 初期化を使います。**]

```{.python .input}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**データセットの読み取り**]

:numref:`sec_semantic_segmentation` で紹介されたセマンティックセグメンテーションデータセットを読みました。ランダムクロップの出力イメージシェイプは $320\times 480$ として指定されます。高さと幅の両方が $32$ で割り切れます。

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**トレーニング**]

これで、構築した完全畳み込みネットワークに学習させることができます。ここでの損失関数と精度の計算は、前の章の画像分類のものと本質的に変わりません。転置畳み込み層の出力チャネルを使用して各ピクセルのクラスを予測するため、チャネル次元は損失計算で指定されます。さらに、すべてのピクセルについて、予測されたクラスの正確さに基づいて精度が計算されます。

```{.python .input}
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**予測**]

予測時には、各チャネルで入力イメージを標準化し、そのイメージを CNN が必要とする 4 次元の入力形式に変換する必要があります。

```{.python .input}
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

各ピクセルの [**予測されたクラスを可視化**] するために、予測されたクラスをデータセットのラベル色にマッピングし直します。

```{.python .input}
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

テストデータセット内の画像のサイズと形状はさまざまです。このモデルはストライドが 32 の転置畳み込み層を使用するため、入力イメージの高さまたは幅が 32 で割り切れない場合、転置された畳み込み層の出力の高さまたは幅は入力イメージの形状から逸脱します。この問題に対処するために、イメージ内で 32 の整数倍の高さと幅を持つ複数の矩形領域をトリミングし、これらの領域のピクセルに対して個別に順方向伝播を実行できます。これらの矩形領域の和集合は、入力イメージを完全に覆う必要があることに注意してください。ピクセルが複数の矩形領域で覆われている場合、同じピクセルの別々の領域における転置された畳み込み出力の平均を softmax 演算に入力して、クラスを予測できます。 

わかりやすくするために、いくつかの大きなテストイメージのみを読み取り、予測のために $320\times480$ の領域をイメージの左上隅から切り抜きます。これらのテストイメージでは、トリミングされた領域、予測結果、グラウンドトゥルースを行ごとに印刷します。

```{.python .input}
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## [概要

* 完全畳み込みネットワークは、まず CNN を使用して画像の特徴を抽出し、$1\times 1$ 畳み込み層を介してチャネル数をクラス数に変換し、最後に転置畳み込みによって特徴マップの高さと幅を入力画像の高さと幅に変換します。
* 完全畳み込みネットワークでは、共一次内挿のアップサンプリングを使用して、転置畳み込み層を初期化できます。

## 演習

1. 実験で転置畳み込み層にXavier初期化を使用した場合、結果はどのように変化しますか？
1. ハイパーパラメーターを調整することで、モデルの精度をさらに向上させることができますか。
1. テストイメージ内のすべてのピクセルのクラスを予測します。
1. オリジナルの完全畳み込みネットワーク論文では、いくつかの中間 CNN 層 :cite:`Long.Shelhamer.Darrell.2015` の出力も使用しています。このアイデアを実装してみてください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1582)
:end_tab:

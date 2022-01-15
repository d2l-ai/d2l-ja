# イメージオーグメンテーション
:label:`sec_image_augmentation`

:numref:`sec_alexnet` では、さまざまなアプリケーションでディープニューラルネットワークを成功させるには、大きなデータセットが前提条件であると述べました。
*イメージ・オーグメンテーション* 
は、学習イメージに一連のランダムな変更を加えた後に、類似しているが明確な学習例を生成し、それによって学習セットのサイズを拡大します。また、トレーニング例をランダムに微調整することで、モデルが特定の属性に依存しなくなり、汎化能力が向上するという事実によって、イメージ増強が促進されます。たとえば、画像をさまざまな方法でトリミングして、対象のオブジェクトをさまざまな位置に表示できます。これにより、モデルのオブジェクトの位置への依存を減らすことができます。また、明るさや色などの要素を調整して、モデルの色に対する感度を下げることもできます。当時のAlexNetの成功には、イメージオーグメンテーションが不可欠だったのは事実でしょう。このセクションでは、コンピュータビジョンで広く使用されているこの手法について説明します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
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
```

## 一般的なイメージオーグメンテーション手法

一般的なイメージ増強手法の調査では、次の $400\times 500$ イメージを例として使用します。

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

ほとんどのイメージ増強方法には、ある程度のランダム性があります。イメージ増強の効果を観察しやすくするために、次に補助関数 `apply` を定義します。この関数は、入力イメージ `img` に対してイメージ拡張法 `aug` を複数回実行し、すべての結果を表示します。

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### フリップとクロップ

:begin_tab:`mxnet`
[**画像を左右に反転**] しても、通常、オブジェクトのカテゴリは変わりません。これは、最も早く、最も広く使用されている画像増強方法の 1 つです。次に `transforms` モジュールを使用して `RandomFlipLeftRight` インスタンスを作成し、50% の確率でイメージを左右に反転させます。
:end_tab:

:begin_tab:`pytorch`
[**画像を左右に反転**] しても、通常、オブジェクトのカテゴリは変わりません。これは、最も早く、最も広く使用されている画像増強方法の 1 つです。次に `transforms` モジュールを使用して `RandomHorizontalFlip` インスタンスを作成し、50% の確率でイメージを左右に反転させます。
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**上下反転**] は左右に反転するほど一般的ではありません。しかし、少なくともこの例の画像では、上下に反転しても認識が妨げられることはありません。次に、`RandomFlipTopBottom` インスタンスを作成し、50% の確率でイメージを上下に反転させます。
:end_tab:

:begin_tab:`pytorch`
[**上下反転**] は左右に反転するほど一般的ではありません。しかし、少なくともこの例の画像では、上下に反転しても認識が妨げられることはありません。次に、`RandomVerticalFlip` インスタンスを作成し、50% の確率でイメージを上下に反転させます。
:end_tab:

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

使用した画像の例では、猫は画像の中央にいますが、一般的にはそうではないかもしれません。:numref:`sec_pooling` では、プーリング層は畳み込み層の目標位置に対する感度を下げることができると説明した。また、画像をランダムに切り抜いて、オブジェクトが画像内の異なる位置に異なる縮尺で表示されるようにすることもできます。これにより、ターゲット位置に対するモデルの感度を下げることもできます。 

以下のコードでは、$10\%\ sim 100\ %$ of the original area each time, and the ratio of width to height of this area is randomly selected from $0.5\ sim 2$ のエリアを [**ランダムにトリミング**] しています。次に、領域の幅と高さが両方とも 200 ピクセルにスケーリングされます。特に指定がない限り、このセクションの $a$ から $b$ までの乱数は、区間 $[a, b]$ からランダムかつ均一にサンプリングされた連続値を参照します。

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 色を変える

もうひとつの拡張方法は、色の変更です。画像の色は、明るさ、コントラスト、彩度、色相の 4 つの要素を変更できます。以下の例では、イメージの [**明るさをランダムに変更**] して、元のイメージの 50% ($1-0.5$) から 150% ($1+0.5$) の間の値にします。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

同様に、画像の [**色相をランダムに変える**] も可能です。

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

また、`RandomColorJitter` インスタンスを作成し、[**イメージの `brightness`、`contrast`、`saturation`、`hue` を同時にランダムに変更する方法を設定することもできます**]。

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 複数のイメージオーグメンテーション手法を組み合わせる

実際には、[**複数の画像増強方法を組み合わせる**] します。たとえば、上で定義したさまざまなイメージ拡張メソッドを組み合わせて、`Compose` インスタンスを介して各イメージに適用できます。

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**イメージ増強によるトレーニング**]

イメージオーグメンテーションでモデルをトレーニングしてみましょう。ここでは、前に使用した Fashion-MNIST データセットの代わりに CIFAR-10 データセットを使用します。これは、Fashion-MNIST データセット内のオブジェクトの位置とサイズが正規化されているのに対し、CIFAR-10 データセット内のオブジェクトの色とサイズには大きな違いがあるためです。CIFAR-10 データセットの最初の 32 個のトレーニングイメージを以下に示します。

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

予測中に決定的な結果を得るために、通常、イメージ拡張はトレーニング例にのみ適用し、予測中はランダム演算によるイメージ拡張を使用しません。[**ここでは、最も単純なランダムな左右の反転方法のみを使用しています**]さらに、`ToTensor` インスタンスを使用して、イメージのミニバッチをディープラーニングフレームワークで要求される形式、つまり (バッチサイズ、チャネル数、高さ、幅) の形状を持つ 0 ～ 1 の 32 ビット浮動小数点数に変換します。

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
次に、イメージの読み取りとイメージ拡張の適用を容易にする補助関数を定義します。Gluon のデータセットが提供する関数 `transform_first` は、各トレーニング例 (イメージとラベル) の最初の要素、つまりイメージにイメージ拡張を適用します。`DataLoader` の詳細については、:numref:`sec_fashion_mnist` を参照してください。
:end_tab:

:begin_tab:`pytorch`
次に、[**画像の読み取りと画像拡張の適用を容易にする補助関数を定義する**]。PyTorch のデータセットが提供する `transform` 引数は、拡張を適用してイメージを変換します。`DataLoader` の詳細については、:numref:`sec_fashion_mnist` を参照してください。
:end_tab:

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### マルチ GPU トレーニング

:numref:`sec_resnet` の ResNet-18 モデルを CIFAR-10 データセットでトレーニングします。:numref:`sec_multi_gpu_concise` で紹介されたマルチ GPU トレーニングを思い出してください。以下では、[**複数のGPUを使ってモデルをトレーニングし評価する関数を定義します**]。

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with mutiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

これで、[**イメージ拡張を使用してモデルをトレーニングする `train_with_data_aug` 関数を定義する**] ことができます。この関数は、使用可能なすべての GPU を取得し、Adam を最適化アルゴリズムとして使用し、イメージ拡張をトレーニングデータセットに適用し、最後に定義したばかりの関数 `train_ch13` を呼び出してモデルのトレーニングと評価を行います。

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

ランダムな左右反転に基づくイメージ増強を使って [**モデルをトレーニング**] してみましょう。

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## [概要

* イメージオーグメンテーションは、既存のトレーニングデータに基づいてランダムなイメージを生成し、モデルの汎化能力を向上させます。
* 予測中に決定的な結果を得るために、通常、イメージ拡張はトレーニング例にのみ適用し、予測中はランダム演算によるイメージ拡張を使用しません。
* ディープラーニングフレームワークには、同時に適用できるさまざまなイメージ拡張手法が用意されています。

## 演習

1. イメージオーグメンテーション:`train_with_data_aug(test_augs, test_augs)` を使用せずにモデルに学習をさせます。イメージオーグメンテーションを使用する場合と使用しない場合の学習とテストの精度を比較します。この比較実験は、イメージオーグメンテーションが過適合を緩和できるという主張を裏付けることができるか？なぜ？
1. CIFAR-10 データセットのモデルトレーニングで、複数の異なるイメージ拡張手法を組み合わせます。テストの精度は向上しますか？ 
1. ディープラーニングフレームワークのオンラインドキュメントを参照してください。他にどのような画像増強方法が提供されていますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:

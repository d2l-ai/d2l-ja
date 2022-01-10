# Kaggleでの画像分類 (CIFAR-10)
:label:`sec_kaggle_cifar10`

これまで、ディープラーニングフレームワークの高レベル API を使用して、テンソル形式の画像データセットを直接取得してきました。ただし、カスタムイメージデータセットはイメージファイルの形式で提供されることがよくあります。このセクションでは、未加工の画像ファイルから始めて、整理、読み取り、テンソル形式への変換を段階的に行います。 

:numref:`sec_image_augmentation` の CIFAR-10 データセットを試しました。これはコンピュータービジョンにおいて重要なデータセットです。このセクションでは、前のセクションで学習した知識を、CIFAR-10 画像分類の Kaggle コンペティションを実践するために適用します。(**コンペティションのウェブアドレスは https://www.kaggle.com/c/cifar-10 **) 

:numref:`fig_kaggle_cifar10` は、コンペティションのウェブページに情報を表示します。結果を送信するには、Kaggle アカウントを登録する必要があります。 

![CIFAR-10 image classification competition webpage information. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## データセットの取得と整理

競技データセットはトレーニングセットとテストセットに分かれており、それぞれ50000枚と300000枚の画像が含まれています。テストセットでは、10000 枚の画像が評価に使用され、残りの 290000 枚の画像は評価されません。これらはチートされにくいようにするために含まれています。
*テストセットの結果を手作業で* ラベル付けします。
このデータセットのイメージはすべて PNG カラー (RGB チャネル) イメージファイルで、高さと幅はどちらも 32 ピクセルです。画像は飛行機、車、鳥、猫、鹿、犬、カエル、馬、ボート、トラックの合計10のカテゴリをカバーしています。:numref:`fig_kaggle_cifar10` の左上隅には、データセットの飛行機、車、鳥の画像がいくつか表示されています。 

### データセットのダウンロード

Kaggleにログイン後、:numref:`fig_kaggle_cifar10`に示したCIFAR-10画像分類コンペティションWebページの「データ」タブをクリックし、「Download All」ボタンをクリックしてデータセットをダウンロードできます。ダウンロードしたファイルを `../data` に解凍し、その中で `train.7z` と `test.7z` を解凍すると、データセット全体が次のパスで見つかります。 

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

`train` ディレクトリと `test` ディレクトリにはそれぞれトレーニングイメージとテストイメージが格納され、`trainLabels.csv` はトレーニングイメージのラベルを、`sample_submission.csv` はサンプル送信ファイルです。 

開始しやすくするために、[**最初の 1000 個のトレーニングイメージと 5 個のランダムなテストイメージを含むデータセットの小規模サンプルを提供します。**] Kaggle コンペティションの全データセットを使用するには、次の `demo` 変数を `False` に設定する必要があります。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# If you use the full dataset downloaded for the Kaggle competition, set
# `demo` to False
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**データセットの整理**]

モデルのトレーニングとテストを容易にするため、データセットを整理する必要があります。まず、csvファイルからラベルを読み取ってみましょう。次の関数は、ファイル名の拡張子以外の部分をラベルにマップするディクショナリを返します。

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))
```

次に、`reorg_train_valid` 関数を定義して [**検証セットを元のトレーニングセットから分割**] します。この関数の引数 `valid_ratio` は、検証セットの例の数と元のトレーニングセットの例数の比率です。具体的には、$n$ を例数が最も少ないクラスのイメージ数、$r$ を比率とします。検証セットでは、クラスごとに $\max(\lfloor nr\rfloor,1)$ 個のイメージが分割されます。`valid_ratio=0.1` を例に挙げてみましょう。元のトレーニングセットには 50000 個のイメージが含まれているため、パス `train_valid_test/train` では 45000 個のイメージがトレーニングに使用され、残りの 5000 個のイメージはパス `train_valid_test/valid` で検証セットとして分割されます。データセットを整理すると、同じクラスの画像が同じフォルダーの下に配置されます。

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

以下の `reorg_test` 関数 [**予測中にデータをロードするためのテストセットを整理する**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

最後に、`read_csv_labels`、`reorg_train_valid`、`reorg_test` (**上記で定義した関数**) を [**呼び出す**] 関数を使用します。

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

ここでは、データセットの小規模サンプルに対してのみ、バッチサイズを 32 に設定します。Kaggle コンペティションのデータセット全体をトレーニングおよびテストする場合、`batch_size` は 128 などの大きい整数に設定する必要があります。学習例の 10% をハイパーパラメーターを調整するための検証セットとして分割しました。

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**イメージオーグメンテーション**]

オーバーフィッティングに対処するためにイメージ拡張を使用します。たとえば、学習中に画像をランダムに水平方向に反転させることができます。また、カラー画像の 3 つの RGB チャンネルの標準化も実行できます。以下に、微調整できる操作の一部を示します。

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    gluon.data.vision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Scale the image up to a square of 40 pixels in both height and width
    torchvision.transforms.Resize(40),
    # Randomly crop a square image of 40 pixels in both height and width to
    # produce a small square of 0.64 to 1 times the area of the original
    # image, and then scale it to a square of 32 pixels in both height and
    # width
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

テスト中は、評価結果のランダム性を取り除くため、画像の標準化のみを行います。

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## データセットの読み取り

次に、[**RAW画像ファイルで構成される整理されたデータセットを読み取る**]。各例にはイメージとラベルが含まれています。

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

学習中は、[**上記で定義した画像拡張操作をすべて指定する**] 必要があります。ハイパーパラメーターの調整中に検証セットをモデル評価に使用する場合、イメージ拡張によるランダム性は導入されません。最終予測の前に、ラベル付けされたすべてのデータを最大限に活用するために、トレーニングセットと検証セットを組み合わせてモデルをトレーニングします。

```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**モデル**] の定義

:begin_tab:`mxnet`
ここでは、:numref:`sec_resnet` で説明した実装とは少し異なる `HybridBlock` クラスに基づいて残差ブロックを作成します。これは、計算効率を向上させるためのものです。
:end_tab:

```{.python .input}
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
次に、ResNet-18 モデルを定義します。
:end_tab:

```{.python .input}
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
トレーニング開始前に :numref:`subsec_xavier` で説明されている Xavier 初期化を使用します。
:end_tab:

:begin_tab:`pytorch`
:numref:`sec_resnet` で説明されている ResNet-18 モデルを定義します。
:end_tab:

```{.python .input}
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## [**トレーニング関数**] の定義

検証セットでのモデルのパフォーマンスに応じて、モデルを選択し、ハイパーパラメーターを調整します。以下では、モデルトレーニング関数 `train` を定義します。

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**モデルのトレーニングと検証**]

これで、モデルのトレーニングと検証が可能になりました。次のハイパーパラメータはすべて調整できます。たとえば、エポック数を増やすことができます。`lr_period` と `lr_decay` をそれぞれ 4 と 0.9 に設定すると、最適化アルゴリズムの学習率は 4 エポックごとに 0.9 倍になります。デモンストレーションを容易にするために、ここでは20エポックしかトレーニングしません。

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**テストセットの分類**] と Kaggle での結果の送信

ハイパーパラメーターを持つ有望なモデルが得られたら、すべてのラベル付きデータ (検証セットを含む) を使用してモデルを再トレーニングし、テストセットを分類します。

```{.python .input}
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

上記のコードは `submission.csv` ファイルを生成し、そのフォーマットは Kaggle コンペティションの要件を満たしています。Kaggle に結果を送信する方法は :numref:`sec_kaggle_house` と同様です。 

## [概要

* RAW 画像ファイルを含むデータセットは、必要な形式に整理してから読み取ることができます。

:begin_tab:`mxnet`
* 畳み込みニューラルネットワーク、イメージ増強、ハイブリッドプログラミングをイメージ分類コンペティションで使用できます。
:end_tab:

:begin_tab:`pytorch`
* 畳み込みニューラルネットワークとイメージオーグメンテーションをイメージ分類コンペティションで使用できます。
:end_tab:

## 演習

1. この Kaggle コンペティションには、完全な CIFAR-10 データセットを使用してください。ハイパーパラメータを `batch_size = 128`、`num_epochs = 100`、`lr = 0.1`、`lr_period = 50`、`lr_decay = 0.1` として設定します。このコンペティションで達成できる精度とランキングをご覧ください。さらに改善してもらえますか？
1. イメージオーグメンテーションを使用しない場合、どの程度の精度が得られますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:

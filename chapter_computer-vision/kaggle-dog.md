# Kaggleでの犬種識別 (ImageNet 犬)

このセクションでは、Kaggle で犬種識別問題を練習します。(**このコンペティションのウェブアドレスは https://www.kaggle.com/c/dog-breed-identification  です**)

このコンテストでは、120種類の犬種が表彰されます。実際、このコンペティションのデータセットは ImageNet データセットのサブセットです。:numref:`sec_kaggle_cifar10` の CIFAR-10 データセット内のイメージとは異なり、ImageNet データセット内のイメージは、さまざまな次元でより高く、幅が広くなっています。:numref:`fig_kaggle_dog` は、競合他社のウェブページに情報を表示します。結果を送信するには Kaggle アカウントが必要です。 

![The dog breed identification competition website. The competition dataset can be obtained by clicking the "Data" tab.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## データセットの取得と整理

競技用データセットはトレーニングセットとテストセットに分かれており、3 つの RGB (カラー) チャンネルの 10222 と 10357 の JPEG 画像が含まれています。トレーニングデータセットには、ラブラドール、プードル、ダックスフント、サモエド、ハスキー、チワワ、ヨークシャーテリアなど、120 種類の犬種があります。 

### データセットのダウンロード

Kaggle にログイン後、:numref:`fig_kaggle_dog` に示されたコンペティションウェブページの「データ」タブをクリックし、「Download All」ボタンをクリックしてデータセットをダウンロードできます。`../data` でダウンロードしたファイルを解凍すると、次のパスにデータセット全体が見つかります。 

* 。/data/dog-breed-identification/labels.csv
* 。/data/dog-breed-identification/sample_submission.csv
* 。/data/犬種同定/訓練
* 。/data/犬種同定/テスト

上記の構造は :numref:`sec_kaggle_cifar10` の CIFAR-10 の競合製品と同様で、フォルダー `train/` と `test/` にはそれぞれトレーニング用とテスト用の犬のイメージが含まれ、`labels.csv` にはトレーニングイメージのラベルが含まれています。同様に、開始しやすくするために、前述の [**データセットの小さなサンプルを提供します**]: `train_valid_test_tiny.zip`。Kaggle コンペティションで完全なデータセットを使用する場合は、`demo` 変数を `False` に変更する必要があります。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**データセットの整理**]

データセットは :numref:`sec_kaggle_cifar10` で行ったのと同じように整理できます。つまり、検証セットを元のトレーニングセットから分割し、イメージをラベルでグループ化されたサブフォルダーに移動するということです。 

次の `reorg_dog_data` 関数は、トレーニングデータラベルを読み取り、検証セットを分割して、トレーニングセットを整理します。

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**イメージオーグメンテーション**]

この犬種データセットは ImageNet データセットのサブセットであり、その画像は :numref:`sec_kaggle_cifar10` の CIFAR-10 データセットよりも大きいことを思い出してください。以下に、比較的大きなイメージに役立つ、イメージ拡張操作をいくつか挙げます。

```{.python .input}
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

予測時には、ランダム性のないイメージの前処理操作のみを使用します。

```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**データセットの読み取り**]

:numref:`sec_kaggle_cifar10` のように、未加工の画像ファイルで構成される整理されたデータセットを読み取ることができます。

```{.python .input}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

以下では、:numref:`sec_kaggle_cifar10` と同じ方法でデータイテレーターインスタンスを作成します。

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

## [**事前学習済みモデルの微調整**]

この場合も、このコンペティションのデータセットは ImageNet データセットのサブセットです。したがって、:numref:`sec_fine_tuning` で説明したアプローチを使用して、完全な ImageNet データセットで事前学習済みのモデルを選択し、それを使用してカスタムの小規模出力ネットワークに供給するイメージ特徴を抽出できます。ディープラーニングフレームワークの高レベル API は、ImageNet データセットで事前トレーニングされた幅広いモデルを提供します。ここでは、事前学習済みの ResNet-34 モデルを選択します。このモデルでは、このモデルの出力層の入力 (抽出された特徴) を単純に再利用します。その後、元の出力層を、2 つの全結合層を積み重ねるなど、学習可能な小さなカスタム出力ネットワークに置き換えることができます。:numref:`sec_fine_tuning` の実験とは異なり、以下は特徴抽出に使用された事前学習済みモデルを再学習させません。これにより、グラデーションを保存するためのトレーニング時間とメモリが削減されます。 

完全な ImageNet データセットに対して 3 つの RGB チャネルの平均と標準偏差を使用してイメージを標準化したことを思い出してください。実際、これは ImageNet の事前学習済みモデルによる標準化操作とも一致しています。

```{.python .input}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

[**損失の計算**] の前に、まず事前学習済みモデルの出力層、つまり抽出された特徴量の入力を取得します。次に、この機能を小さなカスタム出力ネットワークの入力として使用し、損失を計算します。

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## [**トレーニング関数**] の定義

検証セットでのモデルのパフォーマンスに応じて、モデルを選択し、ハイパーパラメーターを調整します。モデルトレーニング関数 `train` は、スモールカスタム出力ネットワークのパラメーターのみを反復します。

```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**モデルのトレーニングと検証**]

これで、モデルのトレーニングと検証が可能になりました。次のハイパーパラメーターはすべて調整可能です。たとえば、エポック数を増やすことができます。`lr_period` と `lr_decay` はそれぞれ 2 と 0.9 に設定されているため、最適化アルゴリズムの学習率は 2 エポックごとに 0.9 倍になります。

```{.python .input}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**テストセットの分類**] と Kaggle での結果の送信

:numref:`sec_kaggle_cifar10` の最後のステップと同様に、ラベル付けされたすべてのデータ (検証セットを含む) がモデルのトレーニングとテストセットの分類に使用されます。学習済みのカスタム出力ネットワークを分類に使用します。

```{.python .input}
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=0)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

上記のコードは :numref:`sec_kaggle_house` で説明したのと同じ方法で Kaggle に送信される `submission.csv` ファイルを生成します。 

## [概要

* ImageNet データセット内のイメージは、CIFAR-10 イメージよりもサイズが大きくなります (寸法はさまざまです)。異なるデータセット上のタスクのイメージ拡張操作を変更することがあります。
* ImageNet データセットのサブセットを分類するために、完全な ImageNet データセットで事前学習済みモデルを活用して特徴を抽出し、カスタムの小規模出力ネットワークのみを学習させることができます。これにより、計算時間とメモリコストが削減されます。

## 演習

1. 完全な Kaggle 競合データセットを使用する場合、`batch_size` (バッチサイズ) と `num_epochs` (エポック数) を増やし、他のいくつかのハイパーパラメーターを `lr = 0.01`、`lr_period = 10`、および `lr_decay = 0.1` に設定すると、どのような結果が得られますか?
1. より深い事前学習済みモデルを使用した方が良い結果が得られますか？ハイパーパラメータはどのように調整しますか？結果をさらに向上させることはできますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:

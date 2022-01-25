# シングルショットマルチボックス検出
:label:`sec_ssd`

:numref:`sec_bbox`—:numref:`sec_object-detection-dataset` では、境界ボックス、アンカーボックス、マルチスケールオブジェクト検出、オブジェクト検出用のデータセットが導入されました。これで、このような背景知識を使用して、物体検出モデル (シングルショットマルチボックス検出 (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`) を設計する準備が整いました。このモデルはシンプルで高速で、広く使用されています。これは膨大な量の物体検出モデルの 1 つにすぎませんが、このセクションで説明する設計原則と実装の詳細の一部は、他のモデルにも適用できます。 

## モデル

:numref:`fig_ssd` では、シングルショットマルチボックス検出の設計の概要を説明します。このモデルは主に、ベースネットワークとそれに続く複数のマルチスケールフィーチャマップブロックで構成されます。ベースネットワークは入力イメージから特徴を抽出するためのものなので、深い CNN を使用できます。たとえば、オリジナルのシングルショットマルチボックス検出用紙では、分類層 :cite:`Liu.Anguelov.Erhan.ea.2016` の前に切り捨てられた VGG ネットワークが採用されていますが、ResNet も一般的に使用されています。この設計により、ベースネットワークがより大きな特徴マップを出力し、より小さなオブジェクトを検出するためのアンカーボックスをより多く生成することができます。続いて、各マルチスケール特徴マップブロックは、前のブロックからの特徴マップの高さと幅を (例えば、半分に) 縮小し、特徴マップの各ユニットが入力画像上の受容野を増加させることを可能にする。 

:numref:`sec_multiscale-object-detection` のディープニューラルネットワークによる画像の層単位表現によるマルチスケール物体検出の設計を思い出してください。:numref:`fig_ssd` の最上部に近いマルチスケール特徴マップは小さくても受容場が大きいため、検出されるオブジェクトの数は少なくても大きいオブジェクトに適しています。 

一言で言えば、シングルショットマルチボックス検出は、ベースネットワークといくつかのマルチスケールフィーチャマップブロックを介して、サイズの異なるさまざまなアンカーボックスを生成し、これらのアンカーボックス (つまり境界ボックス) のクラスとオフセットを予測することでさまざまなサイズのオブジェクトを検出します。したがって、これはマルチスケールオブジェクト検出モデル。 

![As a multiscale object detection model, single-shot multibox detection mainly consists of a base network followed by several multiscale feature map blocks.](../img/ssd.svg)
:label:`fig_ssd`

以下では、:numref:`fig_ssd` の異なるブロックの実装の詳細を説明します。まず、クラスと境界ボックス予測の実装方法について説明します。 

### [**クラス予測層**]

オブジェクトクラスの数を $q$ とします。アンカーボックスには $q+1$ クラスがあり、クラス 0 はバックグラウンドです。ある縮尺では、フィーチャマップの高さと幅がそれぞれ $h$ と $w$ であると仮定します。これらのフィーチャマップの各空間位置を中心として $a$ のアンカーボックスを生成する場合、合計 $hwa$ 個のアンカーボックスを分類する必要があります。このため、パラメーター化のコストが高くなる可能性があるため、完全に接続されたレイヤーでの分類が不可能になることがよくあります。:numref:`sec_nin` で、畳み込み層のチャネルを使用してクラスを予測した方法を思い出してください。シングルショットマルチボックス検出では、同じ手法を使用してモデルの複雑さを軽減します。 

具体的には、クラス予測層は、特徴マップの幅や高さを変更せずに畳み込み層を使用します。このようにして、フィーチャマップの空間次元 (幅と高さ) が同じである出力と入力を 1 対 1 で対応させることができます。具体的には、任意の空間位置 ($x$、$y$) にある出力フィーチャマップのチャネルは、入力フィーチャマップの ($x$、$y$) を中心とするすべてのアンカーボックスのクラス予測を表します。有効な予測を生成するには、$a(q+1)$ 個の出力チャネルが必要です。同じ空間位置の場合、インデックス $i(q+1) + j$ を持つ出力チャネルは、アンカーボックス $i$ ($0 \leq i < a$) のクラス $j$ ($0 \leq j \leq q$) の予測を表します。 

以下では、このようなクラス予測層を定義し、$a$ と $q$ を引数 `num_anchors` と `num_classes` でそれぞれ指定します。この層は、パディングが 1 の畳み込み層 $3\times3$ を使用します。この畳み込み層の入力と出力の幅と高さは変わりません。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**境界ボックス予測レイヤー**)

境界ボックス予測層の設計は、クラス予測層の設計と似ています。唯一の違いは、各アンカーボックスの出力数にあります。ここでは $q+1$ クラスではなく 4 つのオフセットを予測する必要があります。

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**複数スケールの予測を連結する**]

前述したように、シングルショットマルチボックス検出では、マルチスケールの特徴マップを使用してアンカーボックスを生成し、そのクラスとオフセットを予測します。縮尺が異なると、フィーチャマップの形状や、同じ単位を中心とするアンカーボックスの数が異なる場合があります。したがって、異なる縮尺での予測出力の形状は異なる場合があります。 

次の例では、同じミニバッチに対して `Y1` と `Y2` という 2 つの異なる縮尺で特徴マップを作成します。`Y2` の高さと幅は `Y1` の半分になります。クラス予測を例に挙げてみましょう。`Y1` と `Y2` のユニットごとに 5 個と 3 個のアンカーボックスがそれぞれ生成されているとします。さらに、オブジェクトクラスの数が 10 であるとします。特徴マップ `Y1` と `Y2` の場合、クラス予測出力のチャネル数はそれぞれ $5\times(10+1)=55$ と $3\times(10+1)=33$ で、どちらの出力形状も (バッチサイズ、チャネル数、高さ、幅) になります。

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

ご覧のとおり、バッチサイズのディメンションを除いて、他の 3 つのディメンションはすべてサイズが異なります。より効率的な計算のためにこれら 2 つの予測出力を連結するために、これらのテンソルをより一貫性のある形式に変換します。 

チャネルディメンションには、同じ中心をもつアンカーボックスの予測が保持されることに注意してください。まず、この次元を最も内側に動かします。バッチサイズは異なるスケールで同じままなので、予測出力を形状 (バッチサイズ、高さ $\times$ 幅 $\times$ チャネル数) を持つ 2 次元テンソルに変換できます。そうすれば、そのような出力を次元 1 に沿って異なるスケールで連結できます。

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

この方法では、`Y1` と `Y2` のチャネル、高さ、幅のサイズは異なりますが、これら 2 つの予測出力を同じミニバッチの 2 つの異なるスケールで連結できます。

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**ダウンサンプリングブロック**]

複数の縮尺でオブジェクトを検出するために、入力特徴マップの高さと幅を半分にする次のダウンサンプリングブロック `down_sample_blk` を定義します。実際、このブロックは :numref:`subsec_vgg-blocks` の VGG ブロックの設計を適用しています。より具体的には、各ダウンサンプリングブロックは、パディングが 1 の 2 つの $3\times3$ 畳み込み層と、ストライドが 2 の $2\times2$ 最大プーリング層で構成されます。ご存知のように、パディングが 1 の $3\times3$ 畳み込み層では、特徴マップの形状は変わりません。ただし、その後の $2\times2$ の最大プーリングにより、入力フィーチャマップの高さと幅が半分に減少します。$1\times 2+(3-1)+(3-1)=6$ のため、このダウンサンプリングブロックの入力と出力の両方の特徴マップでは、出力の各ユニットには入力に $6\times6$ 受容場があります。したがって、ダウンサンプリングブロックは出力特徴マップ内の各ユニットの受容場を拡大します。

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

次の例では、構築した downsampling ブロックは入力チャネル数を変更し、入力特徴マップの高さと幅を半分にします。

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**基本ネットワークブロック**]

Base Network ブロックは、入力イメージからフィーチャを抽出するために使用されます。簡単にするために、各ブロックでチャネル数を 2 倍にする 3 つのダウンサンプリングブロックで構成される小規模なベースネットワークを構築します。$256\times256$ の入力イメージがある場合、このベースネットワークブロックは $32 \times 32$ フィーチャマップ ($256/2^3=32$) を出力します。

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### コンプリートモデル

[**完全なシングルショットマルチボックス検出モデルは 5 つのブロックで構成されます。**] 各ブロックで生成された特徴マップは、(i) アンカーボックスの生成と (ii) アンカーボックスのクラスとオフセットの予測の両方に使用されます。これら 5 つのブロックのうち、1 つ目はベースネットワークブロック、2 番目から 4 つ目はダウンサンプリングブロック、最後のブロックはグローバル最大プーリングを使用して高さと幅の両方を 1 に減らします。技術的には、2 番目から 5 番目のブロックは :numref:`fig_ssd` のすべてのマルチスケール特徴マップブロックです。

```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

ここで、ブロックごとに [**順伝播の定義**] を行います。画像分類タスクとは異なり、ここでの出力には、(i) CNN 特徴マップ `Y`、(ii) 現在の縮尺で `Y` を使用して生成されたアンカーボックス、(iii) これらのアンカーボックスについて (`Y` に基づく) 予測されたクラスとオフセットが含まれます。

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

:numref:`fig_ssd` では、上部に近いマルチスケールの特徴マップブロックは、より大きなオブジェクトを検出するためのものであるため、より大きなアンカーボックスを生成する必要があることを思い出してください。上記の順伝播では、各マルチスケール特徴マップブロックで、呼び出された `multibox_prior` 関数 (:numref:`sec_anchor` で説明) の `sizes` 引数を介して 2 つのスケール値のリストを渡します。次の例では、0.2 と 1.05 の間隔が 5 つのセクションに均等に分割され、5 つのブロック (0.2、0.37、0.54、0.71、0.88) で小さいスケール値が決定されます。その場合、$\sqrt{0.2 \times 0.37} = 0.272$、$\sqrt{0.37 \times 0.54} = 0.447$ などによって大きなスケール値が与えられます。 

[~~各ブロックのハイパーパラメータ~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

これで、次のように [**完全なモデルを定義する**] `TinySSD` ができる。

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

$256 \times 256$ イメージ `X` のミニバッチに対して [**モデルインスタンスを作成し、それを使用して順伝播を実行します**]。 

このセクションで既に示したように、最初のブロックは $32 \times 32$ 特徴マップを出力します。2 番目から 4 番目のダウンサンプリングブロックでは高さと幅が半分になり、5 番目のブロックではグローバルプーリングが使用されていることを思い出してください。フィーチャマップの空間次元に沿って、ユニットごとに 4 つのアンカーボックスが生成されるため、5 つの縮尺すべてで、画像ごとに合計 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ 個のアンカーボックスが生成されます。

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 訓練

ここでは、物体検出用のシングルショットマルチボックス検出モデルを学習させる方法を説明します。 

### データセットの読み取りとモデルの初期化

はじめに、:numref:`sec_object-detection-dataset` で説明されている [**バナナ検出データセットを読む**]。

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

バナナ検出データセットにはクラスが 1 つしかありません。モデルを定義したら、(**パラメーターの初期化と最適化アルゴリズムの定義**) を行う必要があります。

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**損失関数と評価関数の定義**]

物体検出には 2 種類の損失があります。最初の損失はアンカーボックスのクラスに関係します。その計算では、画像分類に使用したクロスエントロピー損失関数を簡単に再利用できます。2 つ目の損失は、ポジティブ (バックグラウンドでない) アンカーボックスのオフセットに関するものです。これは回帰問題です。ただし、この回帰問題では :numref:`subsec_normal_distribution_and_squared_loss` で説明した二乗損失は使用しません。代わりに、予測とグラウンドトゥルースの差の絶対値である $L_1$ ノルム損失を使用します。マスク変数 `bbox_masks` は、損失計算で負のアンカーボックスと不正な (埋め込まれた) アンカーボックスを除外します。最後に、アンカーボックスクラス損失とアンカーボックスオフセット損失を合計して、モデルの損失関数を求めます。

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

精度を使用して分類結果を評価できます。オフセットには $L_1$ ノルム損失が使用されているため、予測された境界ボックスの評価には*平均絶対誤差* を使用します。これらの予測結果は、生成されたアンカーボックスとアンカーボックスの予測オフセットから得られます。

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**モデルのトレーニング**]

モデルの学習時には、マルチスケールのアンカーボックス (`anchors`) を生成し、順伝播でそれらのクラス (`cls_preds`) とオフセット (`bbox_preds`) を予測する必要があります。次に、ラベル情報 `Y` に基づいて、生成されたアンカーボックスのクラス (`cls_labels`) とオフセット (`bbox_labels`) にラベルを付けます。最後に、クラスとオフセットの予測値とラベル付けされた値を使用して、損失関数を計算します。実装を簡潔にするため、ここではテストデータセットの評価は省略しています。

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**予測**]

予測時の目標は、イメージ上の対象オブジェクトをすべて検出することです。以下では、テストイメージを読み取ってサイズを変更し、畳み込み層が必要とする 4 次元テンソルに変換します。

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

次の `multibox_detection` 関数を使用すると、アンカーボックスとその予測オフセットから予測される境界ボックスが取得されます。次に、非最大抑制を使用して、予測される類似の境界ボックスを削除します。

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

最後に、[**信頼度 0.9 以上のすべての予測境界ボックスを表示する**] を出力します。

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## [概要

* シングルショットマルチボックス検出は、マルチスケールの物体検出モデルです。シングルショットマルチボックス検出は、そのベースネットワークといくつかのマルチスケールフィーチャマップブロックを介して、サイズの異なるさまざまなアンカーボックスを生成し、これらのアンカーボックス (つまり境界ボックス) のクラスとオフセットを予測して、さまざまなサイズのオブジェクトを検出します。
* シングルショットマルチボックス検出モデルに学習をさせる場合、アンカーボックスクラスとオフセットの予測値とラベル付けされた値に基づいて損失関数が計算されます。

## 演習

1. 損失関数を改善して、シングルショットマルチボックス検出を改善できますか？たとえば、予測されたオフセットの $L_1$ ノルム損失を滑らかな $L_1$ ノルム損失に置き換えます。この損失関数は平滑化のためにゼロ付近の二乗関数を使用し、ハイパーパラメーター $\sigma$ によって制御されます。

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

$\sigma$ が非常に大きい場合、この損失は $L_1$ ノルム損失とほぼ同じです。値が小さいほど、損失関数はより滑らかになります。

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

また、実験ではクラス予測にクロスエントロピー損失を使用しました。$p_j$ はグラウンドトゥルースクラス $j$ の予測確率を示し、クロスエントロピー損失は $-\log p_j$ です。フォーカルロス :cite:`Lin.Goyal.Girshick.ea.2017` も使用できます。ハイパーパラメータ $\gamma > 0$ と $\alpha > 0$ を指定すると、この損失は次のように定義されます。 

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

ご覧のとおり、$\gamma$ を増やすと、適切に分類された例 ($p_j > 0.5$ など) の相対損失を効果的に減らすことができるため、誤分類された難しい例に学習を集中させることができます。

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. スペースの制限により、このセクションではシングルショットマルチボックス検出モデルの実装の詳細をいくつか省略しました。次の点でモデルをさらに改善できますか。
    1. オブジェクトがイメージに比べてはるかに小さい場合、モデルは入力イメージのサイズを大きくすることができます。
    1. 通常、負のアンカーボックスは膨大です。クラス分布のバランスをより良くするために、負のアンカーボックスをダウンサンプリングすることができます。
    1. 損失関数で、クラス損失とオフセット損失に異なる重みハイパーパラメーターを割り当てます。
    1. シングルショットマルチボックス検出ペーパー :cite:`Liu.Anguelov.Erhan.ea.2016` のように、オブジェクト検出モデルを評価するには、他の方法を使用します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:

# オブジェクト検出とバウンディングボックス
:label:`sec_bbox`

以前のセクション (:numref:`sec_alexnet`—:numref:`sec_googlenet` など) では、画像分類のためのさまざまなモデルを紹介しました。画像分類タスクでは、画像内に主要なオブジェクトが 1 つしかないと想定し、そのカテゴリの認識方法のみに焦点を当てます。ただし、対象のイメージには「複数」のオブジェクトが含まれていることがよくあります。カテゴリだけでなく、画像内の特定の位置も知りたいです。コンピュータビジョンでは、このようなタスクを*物体検出* (または*物体認識*) と呼びます。 

物体検出は多くの分野で広く適用されています。たとえば、自動運転では、撮影したビデオ画像から車両、歩行者、道路、障害物の位置を検出して、走行ルートを計画する必要があります。さらに、ロボットはこの技術を使用して、環境内を移動する間、関心のあるオブジェクトを検出して位置を特定できます。さらに、セキュリティシステムでは、侵入者や爆弾などの異常な物体を検出する必要がある場合があります。 

次のいくつかのセクションでは、オブジェクト検出のためのディープラーニング手法をいくつか紹介します。まず、オブジェクトの*位置* (または*位置*) について紹介します。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

このセクションで使用するサンプルイメージをロードします。画像の左側に犬がいて、右側に猫がいるのがわかります。これらは、このイメージの 2 つの主要なオブジェクトです。

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## バウンディングボックス

物体検出では、通常、物体の空間的位置を記述するために*境界ボックス*を使用します。境界ボックスは長方形で、矩形の左上隅の $x$ と $y$ の座標と右下隅の座標によって決まります。もう 1 つのバウンディングボックス表現は、バウンディングボックスの中心の $(x, y)$ 軸座標と、ボックスの幅と高さです。 

[**ここで変換する関数を定義します**] これら (** 2 つの表現**): `box_corner_to_center` は 2 つのコーナー表現から中央幅-高さの表示に、`box_center_to_corner` はその逆に変換します。入力引数 `boxes` は、形状の 2 次元テンソル ($n$, 4) でなければなりません。$n$ は境界ボックスの数です。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

座標情報に基づいて [**画像内の犬と猫の境界ボックスを定義する**] します。イメージ内の座標の原点はイメージの左上隅で、右と下はそれぞれ $x$ と $y$ の軸の正の方向です。

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

2 つのバウンディングボックス変換関数の正しさは、2 回変換することで検証できます。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

[**画像にバウンディングボックスを描画**] して、それらが正確かどうかを確認してみましょう。描画する前に、ヘルパー関数 `bbox_to_rect` を定義します。`matplotlib` パッケージのバウンディングボックス形式でバウンディングボックスを表します。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

画像にバウンディングボックスを追加すると、2 つのオブジェクトの主要なアウトラインが基本的に 2 つのボックスの内側にあることがわかります。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## [概要

* 物体検出は、画像内のすべての対象物だけでなく、その位置も認識します。通常、位置は長方形のバウンディングボックスで表されます。
* よく使用される 2 つのバウンディングボックス表現間で変換できます。

## 演習

1. 別のイメージを探して、そのオブジェクトを含むバウンディングボックスにラベルを付けます。バウンディングボックスとカテゴリのラベル付けを比較します。通常どちらに時間がかかりますか？
1. `box_corner_to_center` と `box_center_to_corner` の入力引数 `boxes` の最も内側の次元が常に 4 なのはなぜですか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:

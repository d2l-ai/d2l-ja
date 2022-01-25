# アンカーボックス
:label:`sec_anchor`

オブジェクト検出アルゴリズムは、通常、入力イメージ内の多数の領域をサンプリングし、これらの領域に関心のあるオブジェクトが含まれているかどうかを判断し、領域の境界を調整して
*グラウンドトゥルースバウンディングボックス*
オブジェクトをより正確に。モデルが異なれば、リージョンのサンプリング方式も異なります。ここでは、そのような方法の 1 つを紹介します。各ピクセルを中心に、さまざまな縮尺とアスペクト比を持つ複数のバウンディングボックスを生成します。これらのバウンディングボックスは*アンカーボックス* と呼ばれます。:numref:`sec_ssd` では、アンカーボックスに基づく物体検出モデルを設計します。 

まず、出力をより簡潔にするため、印刷精度を変更してみましょう。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## 複数のアンカーボックスを生成する

入力イメージの高さが $h$ で、幅が $w$ であるとします。画像の各ピクセルを中心に、さまざまな形状のアンカーボックスを生成します。*スケール* を $s\in (0, 1]$ とし、*アスペクト比* (幅と高さの比率) を $r > 0$ とします。すると、[**アンカーボックスの幅と高さはそれぞれ $ws\sqrt{r}$ と $hs/\sqrt{r}$ です。**] 中心位置を指定すると、幅と高さがわかっているアンカーボックスが決定されます。 

異なる形状のアンカーボックスを複数生成するために、一連のスケール $s_1,\ldots, s_n$ と一連のアスペクト比 $r_1,\ldots, r_m$ を設定しましょう。これらのスケールとアスペクト比のすべての組み合わせを各ピクセルを中心として使用すると、入力イメージのアンカーボックスは合計 $whnm$ 個になります。これらのアンカーボックスはすべてのグラウンドトゥルースバウンディングボックスをカバーする可能性がありますが、計算の複雑さが高すぎるのは簡単です。実際には、$s_1$ または $r_1$ しかできません (**を含む組み合わせを考慮して**)。 

(** $(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$ドル**) 

つまり、同じピクセルを中心とするアンカーボックスの数は $n+m-1$ です。入力イメージ全体に対して、合計 $wh(n+m-1)$ 個のアンカーボックスを生成します。 

上記のアンカーボックスの生成方法は、次の `multibox_prior` 関数に実装されています。入力画像、スケールのリスト、縦横比のリストを指定すると、この関数はすべてのアンカーボックスを返します。

```{.python .input}
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

[**返されるアンカーボックス変数 `Y`**] は (バッチサイズ、アンカーボックスの数、4) であることがわかります。

```{.python .input}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

アンカーボックス変数 `Y` の形状を (イメージの高さ、イメージの幅、同じピクセルを中心とするアンカーボックスの数、4) に変更すると、指定したピクセル位置を中心とするすべてのアンカーボックスを取得できます。以下では、[** (250, 250) を中心とする最初のアンカーボックスにアクセスする**]これには、アンカーボックスの左上隅にある $(x, y)$ 軸座標と、アンカーボックスの右下隅にある $(x, y)$ 軸座標の 4 つの要素があります。両方の軸の座標値は、それぞれイメージの幅と高さで除算されます。したがって、範囲は 0 から 1 の間になります。

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

[**画像内の 1 つのピクセルを中心とするすべてのアンカーボックスを表示**] するために、次の `show_bboxes` 関数を定義して、画像上に複数の境界ボックスを描画します。

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

先ほど見てきたように、`boxes` の変数 $x$ と $y$ の座標値は、それぞれイメージの幅と高さで除算されています。アンカーボックスを描画するときは、元の座標値に戻す必要があります。したがって、変数 `bbox_scale` を以下に定義します。これで、画像の (250, 250) を中心にすべてのアンカーボックスを描画できます。ご覧のとおり、スケール 0.75、アスペクト比 1 の青いアンカーボックスがイメージ内の犬をうまく囲んでいます。

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**ユニオンをめぐる交差点 (IoU) **]

画像では、アンカーボックスが「うまく」犬を囲んでいると述べました。物体のグラウンドトゥルースの境界ボックスがわかっている場合、ここで「うまく」はどのように定量化できるのでしょうか。アンカーボックスとグラウンドトゥルースバウンディングボックスの類似度を直感的に測定できます。*Jaccard index* は、2 つのセット間の類似性を測定できることがわかっています。セット $\mathcal{A}$ と $\mathcal{B}$ が与えられると、Jaccard インデックスは交差のサイズをユニオンのサイズで割った値になります。 

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

実際、バウンディングボックスのピクセル領域はピクセルの集合と見なすことができます。このようにして、2 つのバウンディングボックスのピクセルセットの Jaccard インデックスで類似度を測定できます。2 つのバウンディングボックスでは、通常、Jaccard インデックスを *intersection over union (*IOU*) と呼びます。これは :numref:`fig_iou` に示すように、交差面積とユニオン面積の比率です。IoU の範囲は 0 ～ 1 です。0 は 2 つのバウンディングボックスがまったく重ならないことを意味し、1 は 2 つのバウンディングボックスが等しいことを意味します。 

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

このセクションの残りの部分では、IoU を使用して、アンカーボックスとグラウンドトゥルースの境界ボックス間、および異なるアンカーボックス間の類似性を測定します。次の `box_iou` は、アンカーボックスまたはバウンディングボックスのリストが 2 つある場合、これら 2 つのリストでペアワイズ IoU を計算します。

```{.python .input}
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## トレーニングデータ内のアンカーボックスのラベル付け
:label:`subsec_labeling-anchor-boxes`

トレーニングデータセットでは、各アンカーボックスをトレーニング例と見なします。オブジェクト検出モデルをトレーニングするには、各アンカーボックスに*class* ラベルと*offset* ラベルが必要です。前者はアンカーボックスに関連するオブジェクトのクラスで、後者はアンカーボックスに対するグラウンドトゥルース境界ボックスのオフセットです。予測中、各画像に対して複数のアンカーボックスを生成し、すべてのアンカーボックスのクラスとオフセットを予測し、予測されたオフセットに従ってそれらの位置を調整して予測された境界ボックスを取得し、最終的に特定の基準を満たす予測された境界ボックスのみを出力します。 

ご存知のように、オブジェクト検出トレーニングセットには、*グラウンドトゥルース境界ボックス*の位置と囲まれたオブジェクトのクラスのラベルが付属しています。生成された*アンカーボックス*にラベルを付けるには、アンカーボックスに最も近い、*割り当てられた*グラウンドトゥルース境界ボックスのラベル付けされた位置とクラスを参照します。以下では、最も近いグラウンドトゥルースバウンディングボックスをアンカーボックスに割り当てるアルゴリズムについて説明します。  

### [**グラウンドトゥルース境界ボックスをアンカーボックスに割り当てる**]

イメージを指定して、アンカーボックスが $A_1, A_2, \ldots, A_{n_a}$、グラウンドトゥルースの境界ボックスが $B_1, B_2, \ldots, B_{n_b}$ ($n_a \geq n_b$) であるとします。$i^\mathrm{th}$ 行と $j^\mathrm{th}$ 列の要素 $x_{ij}$ がアンカーボックス $A_i$ とグラウンドトゥルース境界ボックス $B_j$ の IOU である行列 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$ を定義してみましょう。このアルゴリズムは、以下のステップで構成されます。 

1. 行列 $\mathbf{X}$ で最大要素を検出し、その行インデックスと列インデックスをそれぞれ $i_1$ と $j_1$ と表します。次に、グラウンドトゥルース境界ボックス $B_{j_1}$ がアンカーボックス $A_{i_1}$ に割り当てられます。$A_{i_1}$ と $B_{j_1}$ はアンカーボックスとグラウンドトゥルースバウンディングボックスのすべてのペアの中で最も近いので、これは非常に直感的です。最初の代入後、行列 $\mathbf{X}$ の ${i_1}^\mathrm{th}$ 行と ${j_1}^\mathrm{th}$ 列のすべての要素を破棄します。 
1. 行列 $\mathbf{X}$ の残りの要素の中で最大の要素を見つけ、その行と列のインデックスをそれぞれ $i_2$ と $j_2$ と表します。グラウンドトゥルース境界ボックス $B_{j_2}$ をアンカーボックス $A_{i_2}$ に割り当て、行列 $\mathbf{X}$ の ${i_2}^\mathrm{th}$ 行と ${j_2}^\mathrm{th}$ 列のすべての要素を破棄します。
1. この時点で、行列 $\mathbf{X}$ の 2 行 2 列の要素は破棄されています。行列 $\mathbf{X}$ の $n_b$ 列のすべての要素が破棄されるまで続行します。この時点で、$n_b$ の各アンカーボックスにグラウンドトゥルース境界ボックスを割り当てました。
1. 残りの $n_a - n_b$ アンカーボックスのみをトラバースします。たとえば、アンカーボックス $A_i$ がある場合、マトリックス $\mathbf{X}$ の $i^\mathrm{th}$ 行全体で、IoU が最大 $A_i$ のグラウンドトゥルース境界ボックス $B_j$ を探し、この IOU が事前定義されたしきい値より大きい場合にのみ $B_j$ を $A_i$ に割り当てます。

上記のアルゴリズムを具体的な例を用いて説明しましょう。:numref:`fig_anchor_label` (左) に示すように、行列 $\mathbf{X}$ の最大値が $x_{23}$ であると仮定して、グラウンドトゥルース境界ボックス $B_3$ をアンカーボックス $A_2$ に割り当てます。次に、行列の行 2 と列 3 のすべての要素を破棄し、残りの要素 (影付きの領域) で最大の $x_{71}$ を見つけ、グラウンドトゥルース境界ボックス $B_1$ をアンカーボックス $A_7$ に割り当てます。次に、:numref:`fig_anchor_label` (中央) に示すように、行列の行 7 と列 1 のすべての要素を破棄し、残りの要素 (影付きの領域) で最大の $x_{54}$ を見つけ、グラウンドトゥルース境界ボックス $B_4$ をアンカーボックス $A_5$ に割り当てます。最後に :numref:`fig_anchor_label` (右) に示すように、行列の行 5 と列 4 のすべての要素を破棄し、残りの要素 (影付きの領域) で最大の $x_{92}$ を見つけ、グラウンドトゥルース境界ボックス $B_2$ をアンカーボックス $A_9$ に割り当てます。その後、残りのアンカーボックス $A_1, A_3, A_4, A_6, A_8$ をトラバースし、しきい値に従ってグラウンドトゥルースバウンディングボックスを割り当てるかどうかを決定するだけで済みます。 

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

このアルゴリズムは、次の `assign_anchor_to_bbox` 関数に実装されています。

```{.python .input}
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= 0.5)[0]
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### クラスとオフセットのラベル付け

これで、各アンカーボックスのクラスとオフセットにラベルを付けることができます。アンカーボックス $A$ にグラウンドトゥルースバウンディングボックス $B$ が割り当てられているとします。一方では、アンカーボックス $A$ のクラスは $B$ のクラスとしてラベル付けされます。一方、アンカーボックス $A$ のオフセットは、$B$ と $A$ の中心座標間の相対位置と、これら 2 つのボックス間の相対サイズに従ってラベル付けされます。データセット内のさまざまなボックスの位置とサイズが異なる場合、相対的な位置とサイズに変換を適用して、オフセットをより均一に分布させ、適合しやすくすることができます。ここでは、一般的な変換について説明します。[** $A$] と [$B$] の中央座標を $(x_a, y_a)$ と $(x_b, y_b)$ とし、幅を $w_a$ と $w_b$、高さを $h_a$ と $h_b$ とします。$A$ のオフセットを次のようにラベル付けします。 

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
where default values of the constants are $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, and $\sigma_w=\sigma_h=0.2$.
This transformation is implemented below in the `offset_boxes` function.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

アンカーボックスにグラウンドトゥルースバウンディングボックスが割り当てられていない場合は、アンカーボックスのクラスに「background」というラベルを付けます。クラスがバックグラウンドであるアンカーボックスは、しばしば*ネガティブ* アンカーボックスと呼ばれ、残りは*ポジティブ*アンカーボックスと呼ばれます。グラウンドトゥルース境界ボックス (`labels` 引数) を使用して [**アンカーボックスのクラスとオフセットにラベルを付ける**](`anchors` 引数) するために、次の `multibox_target` 関数を実装します。この関数は、バックグラウンドクラスを 0 に設定し、新しいクラスの整数インデックスを 1 ずつインクリメントします。

```{.python .input}
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 一例

アンカーボックスのラベル付けを具体的な例で説明しましょう。ロードされたイメージで dog と cat のグラウンドトゥルース境界ボックスを定義します。最初のエレメントはクラス (dog は 0、cat は 1) で、残りの 4 つのエレメントは左上隅と右下コーナーの $(x, y)$ 軸座標 (範囲は 0 ～ 1) です。また、左上隅と右下隅の座標 $A_0, \ldots, A_4$ (インデックスは 0 から始まる) を使用してラベル付けする 5 つのアンカーボックスを作成します。次に [**これらのグラウンドトゥルースの境界ボックスとアンカーボックスをイメージにプロットします**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

上で定義した`multibox_target` 関数を使用して、dog と cat の [**グラウンドトゥルースの境界ボックスに基づいてこれらのアンカーボックスのクラスとオフセットにラベルを付ける**] ことができます。この例では、バックグラウンド、dog、cat クラスのインデックスはそれぞれ 0、1、2 です。以下に、アンカーボックスとグラウンドトゥルースの境界ボックスの例を示すディメンションを追加します。

```{.python .input}
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

返される結果には 3 つの項目があり、すべてテンソル形式です。3 番目の項目には、入力アンカーボックスのラベル付きクラスが含まれます。 

イメージ内のアンカーボックスとグラウンドトゥルースのバウンディングボックスの位置に基づいて、以下の返されるクラスラベルを解析します。まず、アンカーボックスとグラウンドトゥルースバウンディングボックスのすべてのペアの中で、アンカーボックス $A_4$ の IoU と猫のグラウンドトゥルースバウンディングボックスが最大です。したがって、$A_4$ のクラスには cat というラベルが付けられます。$A_4$ を含むペア、または猫のグラウンドトゥルースバウンディングボックスを取り出します。残りの中で、アンカーボックス $A_1$ と犬のグラウンドトゥルースバウンディングボックスのペアは IoU が最大になります。したがって、$A_1$のクラスは犬としてラベル付けされています。次に、ラベルの付いていない残りの 3 つのアンカーボックス ($A_0$、$A_2$、および $A_3$) をトラバースする必要があります。$A_0$ では、IoU が最も大きいグラウンドトゥルースバウンディングボックスのクラスは犬ですが、IoU は定義済みのしきい値 (0.5) を下回っているため、このクラスはバックグラウンドとしてラベル付けされます。$A_2$ では、IoU が最大であるグラウンドトゥルースバウンディングボックスのクラスは猫で、IoU はしきい値を超えているため、class には cat というラベルが付けられます。$A_3$ では、IoU が最も大きいグラウンドトゥルースバウンディングボックスのクラスは猫ですが、値がしきい値を下回っているため、クラスには background というラベルが付けられます。

```{.python .input}
#@tab all
labels[2]
```

2 番目に返されるアイテムは、シェイプのマスク変数 (バッチサイズ、アンカーボックスの数の 4 倍) です。mask 変数の 4 つの要素は、各アンカーボックスの 4 つのオフセット値に対応します。バックグラウンド検出は考慮しないため、この陰性クラスのオフセットは目的関数に影響しないはずです。要素単位の乗算により、マスク変数の 0 は目的関数を計算する前に負のクラスオフセットを除外します。

```{.python .input}
#@tab all
labels[1]
```

最初に返される項目には、アンカーボックスごとにラベル付けされた 4 つのオフセット値が含まれます。ネガティブクラスのアンカーボックスのオフセットにはゼロというラベルが付けられていることに注意してください。

```{.python .input}
#@tab all
labels[0]
```

## 最大値以外の抑制による境界ボックスの予測
:label:`subsec_predicting-bounding-boxes-nms`

予測時には、イメージに対して複数のアンカーボックスを生成し、それぞれに対してクラスとオフセットを予測します。したがって、予測されたオフセットを持つアンカーボックスに従って、*予測された境界ボックス*が取得されます。以下では、アンカーとオフセット予測を入力として受け取り、[**逆オフセット変換を適用して予測された境界ボックスの座標を返す**] 関数を実装します。

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

アンカーボックスが多数ある場合、同じオブジェクトを囲むように、類似した (重なりが大きい) 予測境界ボックスが多数出力される可能性があります。出力を簡略化するために、*non-maximum suppression* (NMS) を使用して、同じオブジェクトに属する類似の予測境界ボックスをマージできます。 

非最大抑制の仕組みは次のとおりです。予測境界ボックス $B$ の場合、オブジェクト検出モデルはクラスごとに予測尤度を計算します。$p$ で予測される最大尤度を表し、この確率に対応するクラスは $B$ の予測クラスです。具体的には、$p$ を、予測境界ボックス $B$ の*信頼度* (スコア) と呼びます。同じイメージ上で、予測されたすべての非バックグラウンド境界ボックスが信頼度によって降順にソートされ、リスト $L$ が生成されます。次に、ソートされたリスト $L$ を次の手順で操作します。 

1. $L$ の信頼度が最も高い予測境界ボックス $B_1$ を基準として選択し、$B_1$ の IOU が $L$ の事前定義済みしきい値 $\epsilon$ を超える非基底予測境界ボックスをすべて削除します。この時点で、$L$ は予測された境界ボックスを最も高い信頼度で維持しますが、類似しすぎる境界ボックスは削除します。一言で言えば、*非最大* 信頼スコアを持つものは*抑制* されます。
1. $L$ から 2 番目に高い信頼度をもつ予測境界ボックス $B_2$ をもう 1 つの基準として選択し、$B_2$ の IOU が $L$ から $\epsilon$ を超える非基底予測境界ボックスをすべて削除します。
1. $L$ で予測されたすべての境界ボックスが基準として使用されるまで、上記のプロセスを繰り返します。この時点で、$L$ で予測された境界ボックスのペアの IoU はしきい値 $\epsilon$ を下回っています。したがって、互いに類似しすぎるペアはありません。 
1. リスト $L$ 内の予測された境界ボックスをすべて出力します。

[**次の `nms` 関数は、信頼スコアを降順にソートし、インデックスを返す**]

```{.python .input}
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

次の `multibox_detection` を [**予測する境界ボックスに最大値以外の抑制を適用する**] と定義しています。実装が少し複雑になっても心配しないでください。実装直後に具体的な例を使ってどのように動作するかを示します。

```{.python .input}
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

ここで [**上記の実装を4つのアンカーボックスを持つ具体的な例に適用する**]。簡略化のため、予測されるオフセットはすべてゼロであると仮定します。これは、予測されるバウンディングボックスがアンカーボックスであることを意味します。背景、犬、猫のクラスごとに、その予測尤度も定義します。

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

[**これらの予測された境界ボックスをイメージに自信を持ってプロットできます**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

これで `multibox_detection` 関数を呼び出して、しきい値を 0.5 に設定した非最大抑制を実行できます。テンソル入力の例に次元を追加することに注意してください。 

[**返される結果の形状**] は (バッチサイズ、アンカーボックスの数、6) であることがわかります。最も内側の次元の 6 つの要素は、同じ予測境界ボックスの出力情報を提供します。最初の要素は予測されるクラスインデックスで、0 から始まります (0 は dog、1 は cat)。値 -1 は、バックグラウンドまたは最大抑制以外での削除を示します。2 番目の要素は、予測された境界ボックスの信頼度です。残りの 4 つの要素は、予測される境界ボックスの左上隅と右下隅の $(x, y)$ 軸座標です (範囲は 0 ～ 1)。

```{.python .input}
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

クラス -1 の予測境界ボックスを削除すると、[**非最大抑制によって保持された最終的な予測境界ボックスを出力する**] ことができます。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

実際には、非最大抑制を実行する前であっても、予測された境界ボックスを低い信頼度で削除できるため、このアルゴリズムでの計算が削減されます。また、非最大抑制の出力を後処理することもできます。たとえば、最終出力の信頼性が高い結果のみを保持するなどです。 

## [概要

* 画像の各ピクセルを中心に、さまざまな形状のアンカーボックスを生成します。
* Jaccard インデックスとも呼ばれるユニオン交差点 (IoU) は、2 つの境界ボックスの類似度を測定します。これは、交差面積とユニオン面積の比率です。
* トレーニングセットでは、アンカーボックスごとに 2 種類のラベルが必要です。1 つはアンカーボックスに関連するオブジェクトのクラスで、もう 1 つはアンカーボックスに対するグラウンドトゥルースバウンディングボックスのオフセットです。
* 予測時には、非最大抑制 (NMS) を使用して類似の予測境界ボックスを削除し、出力を簡略化できます。

## 演習

1. `multibox_prior` 関数で `sizes` と `ratios` の値を変更します。生成されたアンカーボックスにはどのような変更が加えられましたか？
1. IoU が 0.5 の 2 つのバウンディングボックスを構築して可視化します。それらはどのように重なり合っていますか？
1. :numref:`subsec_labeling-anchor-boxes` および :numref:`subsec_predicting-bounding-boxes-nms` の変数 `anchors` を修正します。結果はどう変わるのですか？
1. 非最大抑制は、予測された境界ボックスを*削除* して抑制する欲張りアルゴリズムです。これらの削除されたもののいくつかが実際に役立つ可能性はありますか？*softly*を抑制するためにこのアルゴリズムをどのように修正できますか？ソフト NMS :cite:`Bodla.Singh.Chellappa.ea.2017` を参照してください。
1. 手作りというよりは、非最大抑圧を学べるのか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:

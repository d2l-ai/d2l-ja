# ニューラルスタイル転送

あなたが写真愛好家なら、フィルターに慣れ親しんでいるかもしれません。写真のカラースタイルを変更して、風景写真をよりシャープにしたり、ポートレート写真のスキンを白くしたりすることができます。ただし、通常、1 つのフィルターで変更されるのは写真の 1 つのアスペクトのみです。写真に理想的なスタイルを適用するには、さまざまなフィルターの組み合わせを試す必要があります。このプロセスは、モデルのハイパーパラメーターの調整と同じくらい複雑です。 

このセクションでは、CNN のレイヤーワイズ表現を利用して、ある画像のスタイルを別の画像に自動的に適用します (*style transfer* :cite:`Gatys.Ecker.Bethge.2016`)。このタスクには 2 つの入力イメージが必要です。1 つは*content イメージ* で、もう 1 つは*style イメージ* です。ニューラルネットワークを使用して、コンテンツイメージを修正し、スタイルイメージに近いスタイルイメージにします。例えば、:numref:`fig_style_transfer`のコンテンツ画像はシアトル郊外のマウントレーニア国立公園で撮影した風景写真で、スタイル画像は秋の樫の木をテーマにした油絵です。合成された出力イメージでは、スタイルイメージのオイルブラシストロークが適用され、コンテンツイメージ内のオブジェクトの主要な形状を維持しながら、より鮮明なカラーが得られます。 

![Given content and style images, style transfer outputs a synthesized image.](../img/style-transfer.svg)
:label:`fig_style_transfer`

## メソッド

:numref:`fig_style_transfer_model` は、CNN ベースの転送方式を簡略化した例で示しています。まず、合成されたイメージを、たとえばコンテンツイメージに初期化します。この合成イメージは、スタイル転送プロセス中に更新する必要がある唯一の変数、つまりトレーニング中に更新されるモデルパラメーターです。次に、事前学習済み CNN を選択してイメージ特徴を抽出し、学習中にそのモデルパラメーターをフリーズします。このディープ CNN は、複数のレイヤーを使用して画像の階層的特徴を抽出します。これらのレイヤーの一部の出力は、格納物フィーチャまたはスタイルフィーチャとして選択できます。:numref:`fig_style_transfer_model` を例に挙げてみましょう。ここでの事前学習済みニューラルネットワークには 3 つの畳み込み層があり、2 番目の層は格納物の特徴を出力し、1 番目と 3 番目の層はスタイル特徴を出力します。 

![CNN-based style transfer process. Solid lines show the direction of forward propagation and dotted lines show backward propagation. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`

次に、順伝播 (実線の矢印の方向) によるスタイル伝達の損失関数を計算し、バックプロパゲーション (破線の矢印の方向) によってモデルパラメーター (出力用の合成イメージ) を更新します。スタイル転送で一般的に使用される損失関数は、(i) *content loss* は合成画像とコンテンツ画像をコンテンツ特徴に近づける、(ii) *style loss* は合成画像とスタイル画像をスタイル特徴に近づける、(iii) *全変動損失* は合成された画像のノイズ。最後に、モデルトレーニングが終了したら、スタイル転送のモデルパラメーターを出力して、最終的な合成イメージを生成します。 

以下では、具体的な実験を経て、スタイル移転の技術的詳細を説明します。 

## [**コンテンツとスタイル画像の読み方**]

まず、コンテンツとスタイル画像を読み取ります。印刷された座標軸から、これらの画像のサイズが異なることがわかります。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## [**前処理と後処理**]

以下では、イメージの前処理と後処理のための 2 つの関数を定義します。関数 `preprocess` は、入力イメージの 3 つの RGB チャネルをそれぞれ標準化し、結果を CNN 入力形式に変換します。関数 `postprocess` は、出力イメージのピクセル値を標準化前の元の値に戻します。イメージ印刷関数では、各ピクセルが 0 から 1 までの浮動小数点値を持つ必要があるため、0 より小さい値または 1 より大きい値はそれぞれ 0 または 1 に置き換えます。

```{.python .input}
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## [**地物を抽出する**]

ImageNet データセットで事前学習された VGG-19 モデルを使用して、イメージの特徴 :cite:`Gatys.Ecker.Bethge.2016` を抽出します。

```{.python .input}
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

画像のコンテンツフィーチャとスタイルフィーチャを抽出するために、VGG ネットワーク内の特定のレイヤーの出力を選択できます。一般的に、入力レイヤーに近いほどイメージの詳細を抽出しやすく、逆に言えば、イメージのグローバル情報を抽出しやすくなります。合成された画像にコンテンツ画像の詳細が過度に保持されないようにするために、画像のコンテンツ特徴を出力する*content layer* として、出力に近い VGG レイヤーを選択します。また、ローカルスタイルフィーチャとグローバルスタイルフィーチャを抽出するために、さまざまな VGG レイヤの出力も選択します。これらのレイヤーは*スタイルレイヤー* とも呼ばれます。:numref:`sec_vgg` で述べたように、VGG ネットワークは 5 つの畳み込みブロックを使用します。実験では、4 番目の畳み込みブロックの最後の畳み込み層をコンテンツ層として選択し、各畳み込みブロックの 1 番目の畳み込み層をスタイル層として選択します。これらのレイヤーのインデックスは、`pretrained_net` インスタンスを出力することで取得できます。

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

VGG レイヤーを使用してフィーチャを抽出する場合、入力レイヤーから、出力レイヤーに最も近いコンテンツレイヤーまたはスタイルレイヤーまでのすべてのフィーチャのみを使用する必要があります。新しいネットワークインスタンス `net` を構築します。このインスタンスでは、特徴抽出に使用されるすべての VGG レイヤのみが保持されます。

```{.python .input}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

入力 `X` が与えられた場合、単純にフォワード伝播 `net(X)` を呼び出すと、最後の層の出力しか得られません。中間レイヤーの出力も必要なため、レイヤーごとの計算を実行し、コンテンツとスタイルレイヤーの出力を保持する必要があります。

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

次の 2 つの関数が定義されています。`get_contents` 関数はコンテンツイメージからコンテンツフィーチャを抽出し、`get_styles` 関数はスタイルイメージからスタイルフィーチャを抽出します。事前学習済みの VGG のモデルパラメーターを学習中に更新する必要がないため、学習開始前でも内容とスタイルの特徴を抽出できます。合成イメージはスタイル転送用に更新されるモデルパラメーターのセットであるため、トレーニング中に関数 `extract_features` を呼び出して、合成されたイメージのコンテンツとスタイルの特徴を抽出することしかできません。

```{.python .input}
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## [**損失関数の定義**]

次に、スタイル転送の損失関数について説明します。損失関数は、コンテンツ損失、スタイル損失、および総変動損失で構成されます。 

### コンテンツロス

線形回帰の損失関数と同様に、コンテンツ損失は、合成されたイメージとコンテンツイメージの間のコンテンツの特徴量の差を 2 乗損失関数によって測定します。二乗損失関数の 2 つの入力は、どちらも関数 `extract_features` で計算されたコンテンツ層の出力です。

```{.python .input}
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### スタイルロス

コンテンツの損失と同様に、スタイル損失も二乗損失関数を使用して、合成されたイメージとスタイルイメージのスタイルの違いを測定します。スタイルレイヤーのスタイル出力を表現するには、まず `extract_features` 関数を使用してスタイルレイヤー出力を計算します。出力に $c$ チャネル、高さ $h$、幅 $w$ の 1 つの例があるとします。この出力を $c$ 行と $hw$ 列の行列 $\mathbf{X}$ に変換できます。この行列は $c$ ベクトル $\mathbf{x}_1, \ldots, \mathbf{x}_c$ を連結したものと考えることができ、それぞれの長さは $hw$ です。ここで、ベクトル $\mathbf{x}_i$ はチャネル $i$ のスタイル特徴を表しています。 

これらのベクトル $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$ の*グラム行列* では、行 $i$ と列 $j$ の $x_{ij}$ の要素 $x_{ij}$ がベクトル $\mathbf{x}_i$ と $\mathbf{x}_j$ の内積です。これは、チャネル $i$ と $j$ のスタイル特徴の相関関係を表します。この Gram 行列は、任意のスタイルレイヤーのスタイル出力を表すために使用します。$hw$ の値が大きいと、Gram 行列の値が大きくなる可能性があることに注意してください。また、Gram 行列の高さと幅はどちらもチャネル数 $c$ であることに注意してください。スタイル損失がこれらの値に影響されないようにするために、以下の `gram` 関数は Gram 行列を要素数 ($chw$) で除算します。

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

明らかに、スタイル損失の二乗損失関数の 2 つのグラム行列入力は、合成イメージとスタイルイメージのスタイルレイヤー出力に基づいています。ここでは、スタイルイメージに基づくグラム行列 `gram_Y` が事前に計算されていると仮定します。

```{.python .input}
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### 総変動損失

学習した合成画像に高周波ノイズ、特に明るいピクセルや暗いピクセルが多く含まれることがあります。一般的なノイズリダクション方法の1つは、
*総変動ノイズ除去*。
$(i, j)$ 座標のピクセル値を $x_{i, j}$ で表します。総変動損失の低減 

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

合成されたイメージ上の隣接するピクセルの値を近づけます。

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### 損失関数

[**スタイル転送の損失関数は、コンテンツ損失、スタイル損失、および総変動損失の加重合計です**]。これらのウェイトハイパーパラメータを調整することで、合成された画像のコンテンツ保持、スタイル転送、ノイズリダクションのバランスを取ることができます。

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## [**合成イメージの初期化**]

スタイル転送では、学習中に更新する必要のある変数は合成イメージだけです。したがって、単純なモデル `SynthesizedImage` を定義し、合成されたイメージをモデルパラメーターとして扱うことができます。このモデルでは、前方伝播はモデルパラメーターを返すだけです。

```{.python .input}
class SynthesizedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

次に、`get_inits` 関数を定義します。この関数は、合成されたイメージモデルインスタンスを作成し、イメージ `X` に初期化します。さまざまなスタイルレイヤーのスタイルイメージのグラム行列 `styles_Y_gram` は、学習前に計算されます。

```{.python .input}
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## [**トレーニング**]

スタイル伝達のためにモデルをトレーニングする場合、合成された画像のコンテンツ特徴とスタイル特徴を連続的に抽出し、損失関数を計算します。以下はトレーニングループの定義です。

```{.python .input}
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs], ylim=[0, 20],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.8)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

ここで [**モデルのトレーニングを開始します**]。コンテンツとスタイル画像の高さと幅が 300 x 450 ピクセルに再調整されます。コンテンツイメージを使用して、合成されたイメージを初期化します。

```{.python .input}
device, image_shape = d2l.try_gpu(), (450, 300)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.9, 500, 50)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (300, 450)  # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)
```

合成された画像は、コンテンツ画像の風景とオブジェクトを保持し、同時にスタイル画像の色を転写していることがわかります。たとえば、合成されたイメージには、スタイルイメージと同じような色のブロックがあります。これらのブロックの中には、ブラシストロークの微妙な質感を持つものもあります。 

## [概要

* スタイル転送で一般的に使用される損失関数は、(i) コンテンツの損失は、合成されたイメージとコンテンツイメージをコンテンツの特徴に近づける、(ii) スタイルの損失は、合成されたイメージとスタイルイメージをスタイルフィーチャに近づける、(iii) 全変動損失は、合成された画像。
* 事前学習済み CNN を使用して画像の特徴を抽出し、損失関数を最小化して、合成された画像を学習中にモデルパラメーターとして継続的に更新できます。
* グラム行列を使用して、スタイルレイヤーからのスタイル出力を表します。

## 演習

1. 異なるコンテンツ画層とスタイル画層を選択すると、出力はどのように変化しますか。
1. 損失関数の重みハイパーパラメーターを調整します。出力は保持するコンテンツが多いですか、それともノイズが少ないですか？
1. 異なるコンテンツとスタイルのイメージを使用します。もっと面白い合成画像を作れますか？
1. テキストにスタイル転送を適用できますか？ヒント: you may refer to the survey paper by Hu et al. :cite:`Hu.Lee.Aggarwal.ea.2020`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:

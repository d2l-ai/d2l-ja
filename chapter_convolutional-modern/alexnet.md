# ディープ畳み込みニューラルネットワーク (AlexNet)
:label:`sec_alexnet`

CNNは、LeNetの導入後、コンピュータービジョンや機械学習のコミュニティではよく知られていましたが、CNNはすぐにはこの分野を支配していませんでした。LeNetは初期の小規模データセットで良好な結果を達成しましたが、より大規模でより現実的なデータセットでCNNをトレーニングするパフォーマンスと実現可能性はまだ確立されていませんでした。実際、1990 年代初頭から 2012 年の分水界の結果に至るまでのほとんどの期間、ニューラルネットワークは、サポートベクターマシンなどの他の機械学習手法よりも優れていました。 

コンピュータビジョンの場合、この比較はおそらく公平ではありません。つまり、畳み込みネットワークへの入力は生のピクセル値または軽く処理された (センタリングなど) ピクセル値で構成されますが、実務家は従来のモデルに生のピクセルを供給することはありませんでした。その代わり、典型的なコンピュータービジョンパイプラインは、特徴抽出パイプラインを手作業でエンジニアリングすることで構成されていました。*機能を学ぶ*のではなく、*細工された*。進歩の大半は、機能に関するより巧妙なアイデアを持つことから来ており、学習アルゴリズムはしばしば後付けに追いやられました。 

1990年代には一部のニューラルネットワークアクセラレータが利用可能でしたが、多数のパラメータを持つディープマルチチャネルマルチレイヤCNNを作成するにはまだ十分に強力ではありませんでした。さらに、データセットはまだ比較的小さかった。これらの障害に加えて、パラメーター初期化ヒューリスティック、確率的勾配降下法の巧妙な変種、非スカッシング活性化関数、効果的な正則化手法など、ニューラルネットワークをトレーニングするための重要なトリックがまだ欠けていました。 

したがって、従来のパイプラインは、*エンドツーエンド* (ピクセルから分類) システムをトレーニングするよりも、次のようになります。 

1. 興味深いデータセットを取得します。初期の頃、これらのデータセットには高価なセンサーが必要でした (当時、1 メガピクセルの画像は最先端でした)。
2. 光学、ジオメトリ、その他の解析ツールに関する知識に基づいて、また場合によっては幸運な大学院生の偶然の発見に基づいて、手作りのフィーチャを使用してデータセットを前処理します。
3. SIFT (スケール不変特徴変換) :cite:`Lowe.2004`、SURF (高速化されたロバストな特徴) :cite:`Bay.Tuytelaars.Van-Gool.2006`、その他の任意の数の手作業で調整されたパイプラインなど、特徴抽出の標準セットを介してデータをフィードします。
4. 結果の表現をお気に入りの分類器 (線形モデルやカーネル法など) にダンプして、分類器に学習をさせます。

機械学習の研究者に話をすると、機械学習は重要かつ美しいものであると彼らは信じていました。優雅な理論は様々な分類器の性質を証明した。機械学習の分野は盛んで、厳格で、非常に有用でした。しかし、コンピュータービジョンの研究者に話しかけると、まったく違う話が聞こえてきます。画像認識の汚い真実は、学習アルゴリズムではなく機能が進歩を促進したということです。コンピュータビジョンの研究者は、データセットが少し大きくなったり、クリーンだったり、特徴抽出パイプラインが少し改善されたりすることが、どの学習アルゴリズムよりも最終的な精度にとってはるかに重要であると正当に信じていました。 

## ラーニングリプレゼンテーション

状況をキャストするもう1つの方法は、パイプラインの最も重要な部分が表現だったということです。2012年までは、表現は機械的に計算されていました。実際、新しい一連の特徴関数を設計し、結果を改善し、メソッドを記述することは、紙の主要なジャンルでした。SIFT :cite:`Lowe.2004`、SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`、HOG（指向グラデーションのヒストグラム）:cite:`Dalal.Triggs.2005`、[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)、および同様の特徴抽出器がねぐらを支配しました。 

ヤン・ルクン、ジェフ・ヒントン、ヨシュア・ベンジオ、アンドリュー・ン、アマリ俊一、ユルゲン・シュミドゥーバーなど、別の研究者グループは異なる計画を立てていた。彼らは、特徴そのものを学ぶべきだと信じていました。さらに、彼らは合理的に複雑であるためには、特徴は複数の共同学習層で階層的に構成され、それぞれが学習可能なパラメータを持つべきだと考えました。画像の場合、最下層がエッジ、カラー、テクスチャを検出するようになることがあります。実際、アレックス・クリジェフスキー、イリヤ・サツケバー、ジェフ・ヒントンは、CNNの新しい変種を提案しました。
*AlexNet*、
2012年のImageNetチャレンジで優れたパフォーマンスを達成しました。AlexNet の名前は、画期的な ImageNet 分類紙 :cite:`Krizhevsky.Sutskever.Hinton.2012` の最初の著者であるアレックス・クリジェフスキーにちなんで名付けられました。 

興味深いことに、ネットワークの最下層では、モデルは従来のフィルターに似た特徴抽出器を学習しました。:numref:`fig_filters` は、AlexNet ペーパー :cite:`Krizhevsky.Sutskever.Hinton.2012` から再現され、低レベルのイメージ記述子について説明しています。 

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

ネットワーク内の上位レイヤは、目、鼻、草の葉などの大きな構造を表すために、これらの表現に基づいて構築されることがあります。さらに高いレイヤーでも、人、飛行機、犬、フリスビーなどのオブジェクト全体を表すことがあります。最終的に、最終的な隠れ状態は、異なるカテゴリに属するデータを容易に分離できるように、その内容をまとめたイメージのコンパクトな表現を学習します。 

多層CNNの究極のブレークスルーは2012年に到来しましたが、中核となる研究者グループはこのアイデアに専念し、長年にわたって視覚データの階層表現を学ぼうと試みていました。2012年の究極のブレークスルーは、2つの重要な要素に起因する可能性があります。 

### 成分不足:データ

多数の層をもつディープモデルでは、凸最適化に基づく従来の手法 (線形法やカーネル法など) を大幅に上回るような体制に入るには、大量のデータが必要です。しかし、1990 年代にはコンピューターのストレージ容量が限られており、センサーの相対的なコストがかかり、研究予算が比較的厳しくなっていたため、ほとんどの研究は小さなデータセットに依存していました。UCIのデータセットのコレクションを取り上げた論文は数多くあり、その多くには、低解像度で不自然な環境でキャプチャされた数百または数千の画像しか含まれていませんでした。 

2009 年に ImageNet データセットがリリースされ、研究者は 1000 個の異なるカテゴリのオブジェクトからそれぞれ 1000 個の 100 万個の例からモデルを学ぶことを困難にしました。このデータセットを導入した Fei-Fei Li が率いる研究者は、Google 画像検索を活用して各カテゴリの大規模な候補セットを事前にフィルタリングし、Amazon Mechanical Turk クラウドソーシングパイプラインを使用して、各画像が関連するカテゴリに属しているかどうかを確認しました。この規模は前例のないものでした。ImageNet Challengeと呼ばれるこのコンペティションは、コンピュータービジョンと機械学習の研究を前進させ、研究者は、学者が以前に検討していたよりも大きな規模で最も優れた性能を発揮するモデルを特定することに挑戦しました。 

### 不足している成分:ハードウェア

ディープラーニングモデルは、コンピューティングサイクルを貪欲に消費しています。トレーニングには数百エポックが必要で、反復ごとに計算量のかかる線形代数演算の多層にデータを渡す必要があります。これが、1990 年代から 2000 年代初頭にかけて、より効率的に最適化された凸対物レンズに基づく単純なアルゴリズムが好まれた主な理由の 1 つです。 

*グラフィカル・プロセッシング・ユニット* (GPU) がゲームチェンジャーであることが判明
ディープラーニングを実現可能にするということですこれらのチップは、コンピュータゲームのためにグラフィックス処理を高速化するために長い間開発されてきました。特に、多くのコンピュータグラフィックス作業に必要な、高スループットの $4 \times 4$ 行列-ベクトル製品向けに最適化されています。幸いなことに、この計算は畳み込み層の計算に必要な計算と非常に似ています。その頃、NVIDIA と ATI は GPU を一般的なコンピューティング操作向けに最適化し始め、*汎用 GPU * (GPGPU) として販売するようになりました。 

直感的に理解できるように、最新のマイクロプロセッサ (CPU) のコアについて考えてみましょう。各コアはかなり強力で、高いクロック周波数で動作し、大きなキャッシュ（最大数メガバイトのL3）を備えています。各コアは、分岐予測子、ディープパイプライン、および多種多様なプログラムを実行できるようにするその他の機能を備えた、幅広い命令の実行に適しています。しかし、この明らかな強さはアキレス腱でもあります。汎用コアは構築に非常に費用がかかります。それらには多くのチップ面積、高度なサポート構造 (メモリインターフェイス、コア間のキャッシングロジック、高速インターコネクトなど) が必要であり、いずれのタスクでも比較的劣っています。最新のラップトップには最大4つのコアがあり、ハイエンドサーバーでさえ64コアを超えることはめったにありません。 

比較すると、GPU は $100 \sim 1000$ 個の小さな処理要素 (NVIDIA、ATI、ARM、その他のチップベンダーでは詳細が多少異なります) で構成され、多くの場合、大きなグループにグループ化されます (NVIDIA ではワープと呼ばれます)。各コアは比較的弱く、ときには1GHz未満のクロック周波数で動作することもありますが、そのようなコアの総数により、GPUはCPUよりも桁違いに高速になります。たとえば、NVIDIA の最新世代の Volta は、特殊な命令に対してチップあたり最大 120 TFLOP (より汎用的な命令では最大 24 TFLOP) を提供しますが、CPU の浮動小数点パフォーマンスは現在まで 1 TFlop を超えていません。これが可能である理由は、実際には非常に単純です。第一に、消費電力はクロック周波数によって*二次的に*増加する傾向にあります。したがって、4 倍の速さ (一般的な数) の CPU コアの電力バジェットを考慮すると、16 個の GPU コアを $1/4$ の速度で使用でき、$16 \times 1/4 = 4$ 倍のパフォーマンスが得られます。さらに、GPU コアははるかにシンプルで (実際、長い間、汎用コードを実行することすらできませんでした)、エネルギー効率が向上します。最後に、ディープラーニングの操作の多くは、高いメモリ帯域幅を必要とします。繰り返しになりますが、GPU は CPU の少なくとも 10 倍の幅を持つバスで輝いています。 

2012年に戻ります。大きなブレークスルーは、Alex Krizhevsky と Ilya Sutskever が GPU ハードウェアで実行できるディープな CNN を実装したときでした。彼らは、CNN、畳み込み、行列乗算における計算上のボトルネックは、すべてハードウェアで並列化できる演算であることを認識しました。3GB のメモリを搭載した 2 台の NVIDIA GTX 580 を使用して、高速畳み込みを実装しました。コード [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) は、数年前から業界標準であり、ディープラーニングブームの最初の数年間を支えるほど十分に優れていました。 

## AlexNet

8層CNNを採用したAlexNetは、ImageNet大規模視覚認識チャレンジ2012で驚異的な大差で優勝しました。このネットワークは、学習によって得られる特徴量が、手作業で設計された特徴を超えて、コンピュータビジョンの従来のパラダイムを打ち破ることを初めて示した。 

:numref:`fig_alexnet` が示すように、AlexNet と LeNet のアーキテクチャは非常に似ています。このモデルを2つの小さなGPUに適合させるために2012年に必要だった設計上の癖をいくつか取り除いたAlexNetのやや合理化されたバージョンが提供されています。 

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNetとLeNetのデザイン哲学はよく似ていますが、大きな違いもあります。まず、AlexNetは比較的小さいLenet5よりもはるかに深いです。AlexNet は、5 つの畳み込み層、2 つの完全に接続された隠れ層、1 つの完全接続された出力層の 8 つの層で構成されています。次に、AlexNetはシグモイドの代わりにReLUをアクティベーション関数として使用しました。以下の詳細を掘り下げてみましょう。 

### 建築

AlexNet の最初のレイヤーでは、畳み込みウィンドウの形状は $11\times11$ です。ImageNet のほとんどのイメージは MNIST イメージの 10 倍以上高く、幅が広いため、ImageNet データ内のオブジェクトはより多くのピクセルを占める傾向があります。したがって、オブジェクトをキャプチャするには、より大きい畳み込みウィンドウが必要となります。2 番目のレイヤーの畳み込みウィンドウ形状は $5\times5$ に縮小され、その後に $3\times3$ が続きます。さらに、1 番目、2 番目、5 番目の畳み込み層の後に、ウィンドウ形状 $3\times3$、ストライド 2 を持つ最大プーリング層が追加されます。さらに、AlexNetにはLeNetの10倍の畳み込みチャネルがあります。 

最後の畳み込み層の後には、出力が 4096 の完全結合層が 2 つあります。これら 2 つの巨大な全結合層は、ほぼ 1 GB のモデルパラメーターを生成します。初期のGPUではメモリが限られていたため、元のAlexNetはデュアルデータストリーム設計を採用していたため、2つのGPUはそれぞれモデルの半分だけを保存して計算することができました。幸いなことに、GPU メモリは現在比較的豊富であるため、最近では GPU 間でモデルを分割する必要はほとんどありません (この点では、AlexNet モデルのバージョンは元の論文から逸脱しています)。 

### アクティベーション関数

さらに、AlexNetはSigmoidアクティベーション関数をよりシンプルなReLUアクティベーション関数に変更しました。一方では、ReLU アクティベーション関数の計算はより簡単です。たとえば、シグモイド活性化関数に見られるべき乗演算はありません。一方、ReLU アクティベーション関数を使用すると、さまざまなパラメーター初期化方法を使用する場合にモデルトレーニングが容易になります。これは、シグモイド活性化関数の出力が 0 または 1 に非常に近い場合、これらの領域の勾配はほぼ 0 になり、逆伝播では一部のモデルパラメーターの更新を継続できないためです。一方、正の区間における ReLU 活性化関数の勾配は常に 1 です。したがって、モデルパラメーターが適切に初期化されていない場合、シグモイド関数は正の区間でほぼ 0 の勾配を得るため、モデルを効果的に学習させることができません。 

### キャパシティコントロールと前処理

AlexNet は完全結合層のモデルの複雑度をドロップアウト (:numref:`sec_dropout`) によって制御しますが、LeNet は重みの減衰のみを使用します。データをさらに増強するために、AlexNetのトレーニングループでは、反転、クリッピング、色の変化など、大量のイメージ拡張が追加されました。これにより、モデルのロバスト性が高まり、サンプルサイズが大きくなると過適合が効果的に減少します。データ拡張については :numref:`sec_image_augmentation` で詳しく説明します。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

高さと幅の両方が224の [**シングルチャネルデータ例を構築**](**各層の出力形状を観察するため**)。:numref:`fig_alexnet` の AlexNet アーキテクチャと一致します。

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## データセットの読み取り

AlexNet は論文では ImageNet でトレーニングされていますが、ここでは Fashion-MNist を使用します。これは、ImageNet モデルのコンバージェンスのトレーニングには、最新の GPU でも数時間から数日かかる場合があるためです。AlexNetを [**Fashion-MNist**] に直接適用する際の問題の1つは、(**画像の解像度が低い**) ($28 \times 28$ ピクセル) (**ImageNet 画像よりも**) (**ImageNet 画像よりも**) です (**$224 \times 224$**) (一般的には賢明ではありませんが、ここではAlexNetに忠実であるために行います)建築)。このサイズ変更は `d2l.load_data_fashion_mnist` 関数の `resize` 引数を使用して実行します。

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## 訓練

これで [**AlexNetのトレーニングを開始**] :numref:`sec_lenet` の LeNet と比較すると、ここでの主な変更点は、ネットワークの深さと広さ、イメージ解像度が高く、畳み込みのコストがかかるため、学習率が小さくなり、トレーニングが大幅に遅くなることです。

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## [概要

* AlexNet の構造は LeNet と似ていますが、大規模 ImageNet データセットに適合するように、より多くの畳み込み層と大きなパラメーター空間を使用します。
* 今日、AlexNetははるかに効果的なアーキテクチャに勝っていますが、今日使用されている浅いネットワークから深いネットワークへの重要なステップです。
* AlexNetの実装にはLeNetよりも数行しかないようですが、学術界がこの概念の変化を受け入れ、その優れた実験結果を活用するには何年もかかりました。これは、効率的な計算ツールがなかったことも原因でした。
* ドロップアウト、ReLU、および前処理は、コンピュータービジョンのタスクで優れたパフォーマンスを達成するための他の重要なステップでした。

## 演習

1. エポック数を増やしてみてください。LeNetと比べて、結果はどう違うのですか？なぜ？
1. AlexNet は Fashion-MNIST データセットには複雑すぎるかもしれません。
    1. モデルを単純化して学習を高速化し、精度が大幅に低下しないようにします。
    1. $28 \times 28$ イメージで直接動作する、より優れたモデルを設計します。
1. バッチサイズを変更し、精度と GPU メモリの変化を観察します。
1. AlexNet の計算パフォーマンスを分析します。
    1. AlexNetのメモリフットプリントの主要部分は何ですか？
    1. AlexNetの計算で支配的な部分は何ですか？
    1. 結果を計算するときのメモリ帯域幅はどうですか？
1. LeNet-5 にドロップアウトと ReLU を適用します。改善しますか？前処理はどうですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:

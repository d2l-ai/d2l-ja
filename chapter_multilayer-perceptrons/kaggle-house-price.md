```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Kaggleで住宅価格を予測する
:label:`sec_kaggle_house`

ディープネットワークを構築してトレーニングし、ウェイトディケイやドロップアウトなどのテクニックでそれらを正規化するための基本的なツールをいくつか紹介したので、Kaggleコンペティションに参加してこの知識をすべて実践する準備が整いました。住宅価格予測競争は、始めるのに最適な場所です。データはかなり汎用的で、特殊なモデル（オーディオやビデオなど）を必要とするようなエキゾチックな構造を示していません。このデータセットは、2011年にバート・デ・コックによって収集された:cite:`De-Cock.2011`で、2006年から2010年のアイオワ州エイムズの住宅価格をカバーしています。それは有名なハリソンとルビンフェルド（1978）の[Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)（1978）よりもかなり大きく、より多くの例とより多くの機能の両方を誇っています。 

このセクションでは、データの前処理、モデル設計、およびハイパーパラメータの選択について詳しく説明します。実践的なアプローチを通じて、データサイエンティストとしてのキャリアを導く直感が得られることを願っています。 

## データをダウンロードする

本書全体を通して、ダウンロードしたさまざまなデータセットでモデルのトレーニングとテストを行います。ここでは、ファイルをダウンロードし、zipまたはtarファイルを抽出する（**2つのユーティリティ関数を実装**）します。繰り返しますが、それらの実装は :numref:`sec_utils` に延期します。

```{.python .input  n=2}
%%tab all

def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
```

## Kaggle

[Kaggle](https://www.kaggle.com)は、機械学習コンペティションを主催する人気のあるプラットフォームです。各コンペティションはデータセットを中心としており、その多くは、受賞したソリューションに賞品を提供する利害関係者によって後援されています。このプラットフォームは、ユーザーがフォーラムや共有コードを介して対話し、コラボレーションと競争の両方を促進するのに役立ちます。リーダーボードの追跡は制御不能になることが多く、研究者は基本的な質問をするのではなく前処理のステップに近視的に焦点を合わせていますが、競合するアプローチとコード間の直接的な定量的比較を容易にするプラットフォームの客観性にも大きな価値があります。共有することで、誰もが何がうまくいったか、何がうまくいかなかったかを知ることができます。Kaggleコンペティションに参加するには、まずアカウントを登録する必要があります（:numref:`fig_kaggle`を参照）。 

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing`に示されている住宅価格予測コンペティションページでは、データセット（[データ] タブの下）を見つけ、予測を送信し、ランキングを確認できます。URLはここにあります。 

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques 

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## データセットのアクセスと読み取り

競技データはトレーニングセットとテストセットに分かれていることに注意してください。各レコードには、住宅のプロパティ値と、道路タイプ、建設年、屋根タイプ、地下の状態などの属性が含まれます。フィーチャはさまざまなデータタイプで構成されています。たとえば、建設年は整数で表され、屋根タイプは個別のカテゴリ割り当てで表され、その他のフィーチャは浮動小数点数で表されます。そして、ここで現実は物事を複雑にします。いくつかの例として、一部のデータは完全に欠落しており、欠落している値は単に「na」とマークされています。各家の価格は、トレーニングセットにのみ含まれています（結局それは競争です）。トレーニングセットを分割して検証セットを作成したいと思いますが、Kaggleに予測をアップロードした後にのみ、公式テストセットでモデルを評価できます。:numref:`fig_house_pricing`の競技タブの「データ」タブには、データをダウンロードするためのリンクがあります。

```{.python .input  n=14}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

はじめに、:numref:`sec_pandas`で紹介した [**`pandas`を使用してデータを読み込んで処理する**] を行います。便宜上、Kaggleの住宅データセットをダウンロードしてキャッシュすることができます。このデータセットに対応するファイルが既にキャッシュディレクトリに存在し、その SHA-1 が `sha1_hash` と一致する場合、コードはキャッシュされたファイルを使用して、冗長なダウンロードでインターネットが詰まるのを防ぎます。

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

トレーニングデータセットには 1460 個の例、80 個のフィーチャー、1 個のラベルが含まれていますが、検証データには 1459 個の例と 80 個のフィーチャが含まれています。

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## データ前処理

最初の4つの例から [**最初の4つと最後の2つの機能、およびラベル (SalePrice) **] を見てみましょう。

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

それぞれの例で、最初の特徴はIDであることがわかります。これは、モデルが各トレーニング例を識別するのに役立ちます。これは便利ですが、予測のための情報は含まれていません。したがって、データをモデルに入力する前に、データセットから削除します。また、さまざまなデータタイプがあるため、モデリングを開始する前にデータを前処理する必要があります。 

数値的特徴から始めましょう。まず、ヒューリスティックを適用し、[**すべての欠損値を対応する地物の平均で置き換える**]。次に、すべての特徴を共通の尺度に置くために、(***標準化* 特徴量をゼロ平均と単位分散に再スケーリングすることによってデータを***標準化**)。 

$$x \leftarrow \frac{x - \mu}{\sigma},$$

ここで、$\mu$と$\sigma$はそれぞれ平均と標準偏差を示します。これが実際に私たちの特徴（変数）をゼロ平均と単位分散を持つように変換することを検証するには、$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$と$E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$に注意してください。直感的に、データを標準化する理由は2つあります。まず、最適化に便利であることがわかります。第2に、どのフィーチャーが関連するか*アプリオリ*がわからないため、あるフィーチャーに割り当てられた係数を他のどのフィーチャーよりも多くペナルティを課したくないということです。 

[**次に離散値を扱います。**] これには「MSZoning」などの機能が含まれます。(**ワンホットエンコーディングに置き換えます**) 以前にマルチクラスラベルをベクトルに変換したのと同じ方法で (:numref:`subsec_classification-problem` 参照)。たとえば、「MSZoning」は「RL」と「RM」という値を想定しています。「msZoning」機能を削除すると、2つの新しいインジケーター機能「MsZoning_RL」と「MsZoning_RM」が0または1のいずれかの値で作成されます。ワンホットエンコーディングによると、「msZoning」の元の値が「RL」の場合、「msZoning_RL」は1で、「msZoning_RM」は0です。`pandas` パッケージはこれを自動的に行います。

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding.
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

この変換により、フィーチャの数が 79 から 331 に増加することがわかります (ID 列とラベル列を除く)。

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## エラーメジャー

はじめに、損失を二乗した線形モデルをトレーニングします。当然のことながら、私たちの線形モデルは競争に勝つ提出につながることはありませんが、データに意味のある情報があるかどうかを確認するための健全性チェックを提供します。ここでランダムに推測するよりもうまくできないなら、データ処理のバグが発生する可能性が高いかもしれません。そして、うまくいけば、線形モデルはベースラインとして機能し、単純なモデルが最良の報告モデルにどれだけ近づくかについての直感を与え、より洗練されたモデルからどれだけの利益を期待すべきかを私たちに与えます。 

住宅価格は、株価と同様に、絶対数量よりも相対的な数量を重視します。したがって [**絶対誤差 $y - \hat{y}$ よりも相対誤差 $\frac{y - \hat{y}}{y}$** を重視する傾向があります]。たとえば、典型的な住宅の価値が125,000米ドルであるオハイオ州の農村部の住宅価格を見積もるときに、予測が100,000米ドルずれている場合、おそらく恐ろしい仕事をしているでしょう。一方、カリフォルニアのロスアルトスヒルズでこの金額を間違えた場合、これは驚くほど正確な予測を表している可能性があります（そこでは、住宅価格の中央値が400万米ドルを超えています）。 

(**この問題に対処する1つの方法は、価格見積もりの対数の不一致を測定することです。**) 実際、これはコンテストが提出物の品質を評価するために使用する公式の誤差測定でもあります。結局のところ、$|\log y - \log \hat{y}| \leq \delta$の小さな値$\delta$は、$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$に変換されます。これにより、予測価格の対数とラベル価格の対数の間に次の二乗平均二乗誤差が生じます。 

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values, dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$ 分割交差検証

:numref:`subsec_generalization-model-selection`で [**交差検証**] を導入したことを覚えているかもしれません。そこではモデル選択の扱い方について議論しました。これを活用して、モデル設計を選択し、ハイパーパラメータを調整します。まず、$K$ 分割交差検証手順でデータの $i^\mathrm{th}$ 分割を返す関数が必要です。次に、$i^\mathrm{th}$ セグメントを検証データとしてスライスし、残りをトレーニングデータとして返します。これはデータを処理する最も効率的な方法ではないことに注意してください。データセットがかなり大きければ、もっと賢いことをすることは間違いありません。しかし、この複雑さが増すと、コードが不必要に難読化される可能性があるため、問題が単純であるため、ここでは安全に省略できます。

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**平均検証エラーが返されます**] $K$分割交差検証で$K$回学習させたとき。

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**モデル選択**]

この例では、調整されていないハイパーパラメーターのセットを選択し、読者に任せてモデルを改善します。最適化する変数の数によっては、適切な選択肢を見つけるのに時間がかかる場合があります。十分な大きさのデータセットと通常の種類のハイパーパラメータを使用すると、$K$倍の交差検証は複数のテストに対して適度に回復する傾向があります。しかし、不当に多数のオプションを試すと、運が良ければ、検証のパフォーマンスが真のエラーを表していないことに気付くかもしれません。

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

$K$ 分割交差検証のエラー数がかなり多い場合でも、ハイパーパラメーターのセットに対する学習エラーの数が非常に少ない場合があることに注意してください。これは、私たちが過剰適合していることを示しています。トレーニング中、両方の数値を監視したいと思うでしょう。過適合が少ないということは、データがより強力なモデルをサポートできることを示している可能性があります。大規模なオーバーフィットは、正則化手法を組み込むことで得られることを示唆しているかもしれません。 

##  [**Kaggleで予測を送信する**]

ハイパーパラメーターの適切な選択がわかったので、すべての $K$ モデルによって設定されたテストの平均予測を計算します。予測をCSVファイルに保存すると、結果をKaggleにアップロードするのが簡単になります。次のコードは、`submission.csv` という名前のファイルを生成します。

```{.python .input}
%%tab all
preds = [model(d2l.tensor(data.val.values, dtype=d2l.float32))
         for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

次に、:numref:`fig_kaggle_submit2`で示されているように、Kaggleで予測を送信し、テストセットの実際の住宅価格（ラベル）とどのように比較されるかを確認できます。手順は非常に簡単です。 

* KaggleのWebサイトにログインし、住宅価格予測コンペのページにアクセスします。
* 「予測を送信」または「提出遅延」ボタンをクリックします（この記事を書いている時点で、ボタンは右側にあります）。
* ページ下部の破線ボックスにある「提出ファイルのアップロード」ボタンをクリックし、アップロードする予測ファイルを選択します。
* ページの下部にある「提出する」ボタンをクリックして結果を表示します。

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## まとめ

* 実際のデータにはさまざまなデータ型が混在していることが多く、前処理が必要です。
* 実数値データをゼロ平均と単位分散に再スケーリングするのが適切なデフォルトです。欠損値をその平均値に置き換えます。
* カテゴリカル特徴を指標特徴に変換することで、それらをワンホットベクトルのように扱うことができます。
* $K$ 分割交差検証を使用してモデルを選択し、ハイパーパラメーターを調整できます。
* 対数は相対誤差に役立ちます。

## 演習

1. このセクションの予測を Kaggle に送信してください。あなたの予測はどれくらい良いですか？
1. 欠損値をその平均値で置き換えるのは常に良い考えですか？ヒント:値がランダムに欠落していない状況を構築できますか?
1. $K$ 分割交差検証によってハイパーパラメーターを調整して Kaggle のスコアを改善します。
1. モデルを改善してスコアを改善する (レイヤー、重量の減衰、ドロップアウトなど)。
1. このセクションで行ったような連続数値特徴を標準化しないとどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

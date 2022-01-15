# Kaggleで住宅価格を予測する
:label:`sec_kaggle_house`

ディープネットワークを構築してトレーニングし、ウェイトディケイやドロップアウトなどのテクニックで正規化するための基本的なツールをいくつか紹介したので、Kaggleコンペティションに参加することで、この知識をすべて実践する準備が整いました。住宅価格予測コンペティションは、始めるのに最適な場所です。データはかなり汎用的で、特殊なモデル (オーディオやビデオなど) を必要とする特殊な構造を示しません。2011 年 :cite:`De-Cock.2011` 年に Bart de Cock によって収集されたこのデータセットは、2006年から2010年までのアイオワ州エイムズの住宅価格を対象としています。ハリソンとルービンフェルド (1978) の有名な[Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)よりかなり大きく、より多くの例とより多くの特徴を誇っています。 

このセクションでは、データの前処理、モデル設計、ハイパーパラメーター選択について詳しく説明します。実践的なアプローチを通じて、データサイエンティストとしてのキャリアを導く直感が得られることを願っています。 

## データセットのダウンロードとキャッシュ

本書全体を通して、ダウンロードしたさまざまなデータセットでモデルのトレーニングとテストを行います。ここでは、(**データのダウンロードを容易にするいくつかのユーティリティ関数を実装**) します。まず、文字列 (データセットの*name*) を、データセットを検索するための URL とファイルの整合性を検証する SHA-1 キーの両方を含むタプルにマップするディクショナリ `DATA_HUB` を維持します。このようなデータセットはすべて、アドレスが `DATA_URL` のサイトでホストされています。

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

次の `download` 関数は、データセットをダウンロードしてローカルディレクトリ (デフォルトは `../data`) にキャッシュし、ダウンロードしたファイルの名前を返します。このデータセットに対応するファイルがすでにキャッシュディレクトリに存在し、その SHA-1 が `DATA_HUB` に格納されているものと一致する場合、このコードはキャッシュされたファイルを使用して、冗長なダウンロードによるインターネットの詰まりを回避します。

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

また、2 つのユーティリティ関数も実装しています。1 つは zip ファイルまたは tar ファイルをダウンロードして解凍し、もう 1 つは、本書で使用しているすべてのデータセットを `DATA_HUB` からキャッシュディレクトリにダウンロードするためのものです。

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## Kaggle

[Kaggle](https://www.kaggle.com) は、機械学習のコンペティションを主催する人気のプラットフォームです。各コンペティションはデータセットを中心としており、その多くは受賞したソリューションに賞品を提供するステークホルダーによって後援されています。このプラットフォームは、ユーザーがフォーラムや共有コードを介して対話し、コラボレーションと競争の両方を促進するのに役立ちます。リーダーボードの追跡は制御不能になることが多く、研究者は基本的な質問をするのではなく前処理ステップに近視的に焦点を合わせていますが、競合するアプローチとコード間の直接的な定量的比較を容易にするプラットフォームの客観性にも大きな価値があります。何がうまくいったのか、何がうまくいかなかったのかを誰もが知ることができるように分かち合う。Kaggle コンペティションに参加するには、まずアカウントを登録する必要があります (:numref:`fig_kaggle` 参照)。 

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing` に示すように、住宅価格予測コンペページでは、データセット ([データ] タブ) を検索し、予測を送信し、ランキングを確認できます。URL はここにあります。 

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques 

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## データセットへのアクセスと読み取り

競技データはトレーニングセットとテストセットに分かれています。各レコードには、住宅のプロパティ値と、道路タイプ、建設年、屋根のタイプ、地下の状態などの属性が含まれます。フィーチャは、さまざまなデータタイプで構成されます。たとえば、建設年は整数で表され、屋根のタイプは個別のカテゴリ割り当てで表され、その他のフィーチャは浮動小数点数で表されます。そして、現実が物事を複雑にしているのはここです。いくつかの例として、一部のデータは完全に欠落しており、欠損値は単に「na」とマークされています。各ハウスの価格はトレーニングセットのみに含まれています（結局コンペティションです）。トレーニングセットを分割して検証セットを作成しますが、Kaggle に予測をアップロードした後に公式テストセットでモデルを評価することしかできません。:numref:`fig_house_pricing` の「競技」タブの「データ」タブには、データをダウンロードするためのリンクがあります。 

はじめに、:numref:`sec_pandas` で導入した [**`pandas` を使用してデータを読み込んで処理します**]。したがって、先に進む前に `pandas` がインストールされていることを確認してください。幸いなことに、Jupyterで読んでいる場合は、ノートブックを離れることなくパンダをインストールできます。

```{.python .input}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

便宜上、上で定義したスクリプトを使用して Kaggle の住宅データセットをダウンロードしてキャッシュすることができます。

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

`pandas` を使用して、それぞれトレーニングデータとテストデータを含む 2 つの csv ファイルをロードします。

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

トレーニングデータセットには 1460 個の例、80 個のフィーチャ、1 個のラベルが含まれ、テストデータには 1459 個のサンプルと 80 個のフィーチャが含まれています。

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

最初の 4 つの例の [**最初の 4 つと最後の 2 つのフィーチャと、ラベル (SalePrice) **] を見てみましょう。

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

各例で (**最初の特徴はID**)、モデルが各トレーニング例を識別するのに役立ちます。これは便利ですが、予測を目的とした情報は一切含まれていません。したがって、データをモデルに入力する前に (**データセットから削除**) します。

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## データ前処理

前述のとおり、データタイプは多種多様です。モデリングを開始する前に、データを前処理する必要があります。数値的な特徴から始めましょう。まず、ヒューリスティックを適用します [**すべての欠損値を対応する特徴の平均で置き換える**]。次に、すべての特徴を共通の尺度に置くために、特徴をゼロ平均と単位分散に再スケーリングしてデータを***標準化* します。 

$$x \leftarrow \frac{x - \mu}{\sigma},$$

$\mu$ と $\sigma$ はそれぞれ平均偏差と標準偏差を表します。これが実際にフィーチャ (変数) を平均と単位分散がゼロになるように変換することを検証するには、$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ と $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$ に注意してください。直感的に、2 つの理由からデータを標準化しています。まず、最適化に便利であることがわかります。第2に、どの地物が関連するかが*事前に*わからないため、ある地物に割り当てられた係数を他の地物よりも多くペナルティを課したくありません。

```{.python .input}
#@tab all
# If test data were inaccessible, mean and standard deviation could be 
# calculated from training data
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

[**次は離散値を扱います。**] これには「MSZoning」などの機能が含まれます。(**ワンホットエンコーディングに置き換えます**)、以前にマルチクラスラベルをベクトルに変換したのと同じ方法で (:numref:`subsec_classification-problem` を参照)。たとえば、「MSZoning」は「RL」と「RM」という値を想定しています。「msZoning」機能を削除すると、2 つの新しいインジケーター機能「msZoning_RL」と「msZoning_RM」が作成され、値は 0 または 1 になります。ワンホットエンコーディングによると、「msZoning」の元の値が「RL」の場合、「msZoning_RL」は 1、「msZoning_RM」は 0 になります。`pandas` パッケージはこれを自動的に行います。

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

この変換により、フィーチャの数が 79 から 331 に増加することがわかります。最後に、`values` 属性を介して、[**`pandas` 形式から NumPy 形式を抽出し、テンソルに変換する**] トレーニング用に。

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## [**トレーニング**]

はじめに、二乗損失をもつ線形モデルに学習をさせます。当然のことながら、私たちの線形モデルは、競争に勝つ提出には至りませんが、データに意味のある情報があるかどうかを確認するためのサニティチェックを提供します。ここでランダムな推測よりもうまく行けない場合は、データ処理のバグがある可能性が高くなります。そして、うまくいけば、線形モデルがベースラインとして機能し、単純なモデルが報告された最良のモデルにどれだけ近づくかについてある程度の直感が得られ、より洗練されたモデルからどれだけのゲインが期待できるかがわかります。

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

住宅価格では、株価と同様に、絶対量よりも相対数量を重視しています。したがって、[**絶対誤差 $y - \hat{y}$ よりも相対誤差 $\frac{y - \hat{y}}{y}$**] を重視する傾向があります。たとえば、典型的な住宅の価値が125,000米ドルであるオハイオ州の農村部の住宅価格を見積もるときに予測が100,000米ドルずれている場合、私たちは恐らくひどい仕事をしているでしょう。一方、カリフォルニア州ロスアルトスヒルズでこの金額を誤ると、驚くほど正確な予測になるかもしれません（そこでは、住宅価格の中央値は400万米ドルを超えています）。 

(**この問題に対処する1つの方法は、価格見積もりの対数の不一致を測定することです**) 実際、これは応募作品の質を評価するために競合他社が使用する公式の誤差測定でもあります。結局のところ、$|\log y - \log \hat{y}| \leq \delta$ の小さい値 $\delta$ は $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$ に変換されます。これにより、予測価格の対数とラベル価格の対数の間に、次の二乗平均平方根誤差が生じます。 

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

前のセクションとは異なり、[**私たちのトレーニング関数は Adam オプティマイザーに依存しています (これについては後で詳しく説明します) **]。このオプティマイザの主な魅力は、ハイパーパラメータ最適化のためのリソースが無制限に与えられても、初期学習率に対する感度が大幅に低くなる傾向があることです。

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

## $K$ 分割交差検証

モデル選択の扱い方について説明したセクション (:numref:`sec_model_selection`) に [**$K$ 分割交差検証**] を導入したことを思い出してください。これを、モデル設計の選択とハイパーパラメータの調整に有効に活用します。まず、$K$ 分割の交差検証手順で $i^\mathrm{th}$ 倍のデータを返す関数が必要です。$i^\mathrm{th}$ セグメントを検証データとしてスライスし、残りをトレーニングデータとして返します。これはデータを処理する上で最も効率的な方法ではないことに注意してください。データセットがかなり大きければ、はるかにスマートな処理を行うことは間違いありません。しかし、この複雑さが増すと、コードが不必要に難読化される可能性があるため、問題が単純なため、ここでは安全に省略できます。

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

$K$ 分割交差検証で $K$ 回学習させると [**学習誤差と検証誤差の平均が返されます**]。

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## [**モデル選択**]

この例では、調整されていないハイパーパラメーターのセットを選択し、それを読者に任せてモデルを改善します。最適化する変数の数によっては、適切な選択肢を見つけるのに時間がかかる場合があります。$K$ 分割交差検証では、データセットが十分に大きく、通常の種類のハイパーパラメーターを使用すると、複数の検定に対して適度に回復力がある傾向があります。しかし、不当に多数のオプションを試してみると、運が良ければ、検証のパフォーマンスがもはや真のエラーを表していないことに気付くかもしれません。

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

$K$ 分割交差検証の誤差数がかなり多い場合でも、ハイパーパラメーターのセットの学習誤差の数が非常に少なくなる場合があることに注意してください。これは、過適合していることを示しています。トレーニング中は、両方の数値を監視する必要があります。過適合が少ない場合は、データがより強力なモデルをサポートできることを示している可能性があります。大規模な過適合は、正則化手法を組み込むことで得られることを示唆している可能性があります。 

##  [**Kaggleで予測を送信する**]

ハイパーパラメーターの適切な選択がどうあるべきかがわかったので、(交差検証スライスで使用されるデータの $1-1/K$ ではなく) すべてのデータを使用してトレーニングすることもできます。この方法で得られたモデルは、テストセットに適用できます。予測を csv ファイルに保存すると、結果を Kaggle にアップロードするのが簡単になります。

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

良いサニティチェックの 1 つは、テストセットの予測が $K$ 分割交差検証プロセスの予測と似ているかどうかを確認することです。もしそうなら、Kaggleにアップロードする時です。次のコードは `submission.csv` という名前のファイルを生成します。

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

次に、:numref:`fig_kaggle_submit2` で示したように、Kaggle に関する予測を送信し、テストセットの実際の住宅価格 (ラベル) とどのように比較されるかを確認できます。手順は非常に簡単です。 

* Kaggleのウェブサイトにログインし、住宅価格予測コンペティションページにアクセスしてください。
* 「予測を送信」または「提出遅延」ボタンをクリックします（この記事の執筆時点では、このボタンは右側にあります）。
* ページ下部の破線ボックスにある [提出ファイルをアップロード] ボタンをクリックし、アップロードする予測ファイルを選択します。
* ページ下部にある「提出する」ボタンをクリックすると、結果が表示されます。

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## [概要

* 実データにはさまざまなデータ型が混在していることが多く、前処理が必要です。
* 実数値データをゼロ平均と単位分散に再スケーリングするのが適切な既定値です。欠損値をその平均値に置き換えることもそうです。
* カテゴリカル特徴量をインジケーター特徴量に変換すると、ワンホットベクトルのように扱うことができます。
* $K$ 分割交差検証を使用してモデルを選択し、ハイパーパラメーターを調整できます。
* 対数は相対誤差に便利です。

## 演習

1. このセクションの予測を Kaggle に送信してください。あなたの予測はどれくらい良いですか？
1. 価格の対数を直接最小化してモデルを改善できますか？価格ではなく価格の対数を予測しようとするとどうなりますか？
1. 欠損値を平均値で置き換えるのは常に良い考えですか？ヒント:値がランダムに欠落しない状況を構築できますか?
1. $K$ 分割交差検証によってハイパーパラメーターを調整して Kaggle のスコアを改善します。
1. モデル (レイヤー、ウェイト減衰、ドロップアウトなど) を改善してスコアを向上させます。
1. このセクションで行ったように連続的な数値特徴を標準化しないとどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

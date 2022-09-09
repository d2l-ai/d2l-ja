```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# データ前処理
:label:`sec_pandas`

これまで、既製のテンソルで届いた合成データを扱ってきました。しかし、ディープラーニングを実際に適用するには、任意の形式で保存された乱雑なデータを抽出し、ニーズに合わせて前処理する必要があります。幸いなことに、*pandas* [library](https://pandas.pydata.org/)は重い作業の多くを行うことができます。このセクションは、適切な*pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)に代わるものではありませんが、最も一般的なルーチンのいくつかについての短期集中コースを提供します。 

## データセットの読み取り

カンマ区切り値 (CSV) ファイルは、表形式 (スプレッドシートのような) データを格納するために広く使用されています。ここで、各行は1つのレコードに対応し、いくつかの（カンマ区切り）フィールドで構成されています。例えば、「アルバート・アインシュタイン、1879年3月14日、ウルム、連邦工科大学、重力物理学の分野での成果」。`pandas` で CSV ファイルを読み込む方法を示すために、(**以下で CSV ファイルを作成します**) `../data/house_tiny.csv`。このファイルは住宅のデータセットを表し、各行は個別の家に対応し、列は部屋数（`NumRooms`）、屋根のタイプ（`RoofType`）、および価格（`Price`）に対応します。

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

それでは、`pandas`をインポートして、`read_csv`でデータセットをロードしましょう。

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## データ準備

教師あり学習では、*入力*値のセットを指定して、指定された*目標*値を予測するようにモデルをトレーニングします。データセットを処理する最初のステップは、入力値とターゲット値に対応する列を分離することです。列は、名前または整数位置ベースのインデックス (`iloc`) によって選択できます。 

`pandas`が値`NA`を持つすべてのCSVエントリを特別な`NaN`（*数字ではない*）値に置き換えたことに気づいたかもしれません。これは、「3,, ,270000" など、エントリが空の場合にも発生する可能性があります。これらは*ミッシングバリュー*と呼ばれ、データサイエンスの「トコジラミ」であり、キャリアを通じて直面する永続的な脅威です。コンテキストによっては、欠落した値は*代入* または*削除* によって処理されます。補完は欠損値をその値の推定値に置き換え、削除は欠損値を含む行または列のいずれかを単に破棄します。  

以下に、一般的な帰属ヒューリスティックをいくつか示します。[**カテゴリ入力フィールドの場合、`NaN`をカテゴリとして扱うことができます。**] `RoofType`列は`Slate`と`NaN`の値を取るため、`pandas`はこの列を`RoofType_Slate`と`RoofType_nan`の2つの列に変換できます。路地タイプが `Slate` の行は、`RoofType_Slate` と `RoofType_nan` の値をそれぞれ 1 と 0 に設定します。`RoofType` の値が欠落している行については、その逆が成り立ちます。

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

欠落している数値の場合の一般的なヒューリスティックは、[**`NaN`エントリを対応する列の平均値に置き換える**] です。

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## テンソル形式への変換

[**`inputs`と`targets`のすべてのエントリが数値であるため、テンソルにロードできます**]（:numref:`sec_ndarray`を思い出してください）。

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.values), np.array(targets.values)
X, y
```

```{.python .input}
%%tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(targets.values)
X, y
```

## ディスカッション

これで、データ列を分割し、欠損変数を補完し、`pandas` データをテンソルに読み込む方法がわかりました。:numref:`sec_kaggle_house`では、もう少しデータ処理スキルを習得します。この短期集中コースは物事をシンプルに保ちましたが、データ処理は毛むくじゃらになることがあります。たとえば、データセットが 1 つの CSV ファイルに収まるのではなく、リレーショナルデータベースから抽出された複数のファイルに分散している場合があります。たとえば、eコマースアプリケーションでは、顧客の住所は1つのテーブルにあり、購入データは別のテーブルにあります。さらに、開業医は、カテゴリや数値を超える無数のデータタイプに直面しています。その他のデータタイプには、テキスト文字列、画像、オーディオデータ、および点群が含まれます。多くの場合、データ処理が機械学習パイプラインの最大のボトルネックになるのを防ぐために、高度なツールと効率的なアルゴリズムが必要です。これらの問題は、コンピュータービジョンと自然言語処理に着いたときに発生します。最後に、データ品質に注意を払う必要があります。実世界のデータセットは、外れ値、センサーからの誤った測定、および記録エラーに悩まされることが多く、データをモデルに送る前に対処する必要があります。[seaborn](https://seaborn.pydata.org/)、[Bokeh](https://docs.bokeh.org/)、[matplotlib](https://matplotlib.org/) などのデータ視覚化ツールは、データを手動で検査し、対処する必要がある問題について直感的に理解するのに役立ちます。 

## 演習

1. [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)からAbaloneなどのデータセットをロードして、そのプロパティを調べてみてください。欠損値があるのはどの割合ですか？変数のどの部分が数値、カテゴリ、またはテキストですか？
1. データ列のインデックスを作成し、列番号ではなく名前で選択してみてください。[indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)のpandasのドキュメントには、これを行う方法の詳細が記載されています。
1. この方法で読み込めるデータセットの大きさはどれくらいだと思いますか？どのような制限がありますか？ヒント:データ、表現、処理、およびメモリフットプリントを読み取る時間を考慮してください。これをラップトップで試してみてください。サーバーで試してみると何が変わりますか？ 
1. カテゴリ数が非常に多いデータをどのように扱いますか？カテゴリラベルがすべて一意の場合はどうなりますか？後者を含めるべきですか？
1. パンダに代わるものは何ですか？[loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)はどう？Python イメージングライブラリ [Pillow](https://python-pillow.org/) をチェックしてください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

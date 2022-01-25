# データ前処理
:label:`sec_pandas`

これまで、テンソルにすでに格納されているデータを操作するためのさまざまな手法を紹介してきました。ディープラーニングを現実世界の問題の解決に適用するために、テンソル形式で適切に準備されたデータではなく、生データの前処理から始めることがよくあります。Python でよく使われるデータ分析ツールの中でも、`pandas` パッケージがよく使われています。Python の広大なエコシステムにある他の多くの拡張パッケージと同様に、`pandas` はテンソルと連携して動作することができます。そこで、生データを `pandas` で前処理し、テンソル形式に変換する手順を簡単に説明します。データの前処理テクニックについては、後の章で詳しく説明します。 

## データセットの読み取り

例として、(**csv (カンマ区切り値) ファイルに格納される人工データセットを作成する**) `../data/house_tiny.csv` から始めます。他の形式で保存されたデータも同様の方法で処理される場合があります。 

以下では、データセットを行ごとにcsvファイルに書き込みます。

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

[**作成した csv ファイルから生のデータセットを読み込む**] には、`pandas` パッケージをインポートし、`read_csv` 関数を呼び出します。このデータセットには 4 つの行と 3 つの列があり、各行には家の部屋数 (「numRooms」)、路地のタイプ (「路地」)、価格 (「価格」) が記述されています。

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## 欠損データの処理

「NaN」エントリは欠損値であることに注意してください。欠損データを処理するために、典型的な方法には*imputation* と*delettion* があります。補完では欠損値が置換された値に置き換えられ、削除では欠損値は無視されます。ここでは、帰属について検討します。 

整数位置ベースのインデックス (`iloc`) により、`data` を `inputs` と `outputs` に分割しました。前者は最初の 2 つのカラムを取り、後者は最後のカラムだけを保持します。`inputs` の数値が欠落している場合は、[**「NaN」エントリを同じ列の平均値に置き換えます。**]

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[** `inputs` のカテゴリ値または不連続値については、「NaN」をカテゴリと見なします。**]「Alley」列は「Pave」と「NaN」の 2 種類のカテゴリ値しか取らないため、`pandas` はこの列を「Alley_Pave」と「alley_NAN」の 2 つの列に自動的に変換できます。路地タイプが「Pave」の行は、「Alley_Pave」と「alley_NAN」の値を 1 と 0 に設定します。路地タイプが欠落している行は、その値を 0 と 1 に設定します。

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## テンソル形式への変換

[**`inputs` と `outputs` のすべてのエントリは数値なので、テンソル形式に変換できます。**] データがこの形式になると、:numref:`sec_ndarray` で紹介したテンソル関数でさらに操作できるようになります。

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## [概要

* Python の広大なエコシステムにある他の多くの拡張パッケージと同様に、`pandas` はテンソルと連携して動作することができます。
* 補完と削除は、欠損データを処理するために使用できます。

## 演習

行と列の数が多い生データセットを作成します。 

1. 欠損値が最も多い列を削除します。
2. 前処理されたデータセットをテンソル形式に変換します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

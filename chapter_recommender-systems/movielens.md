#  MovieLens データセット

レコメンデーションリサーチに使用できるデータセットは数多くあります。その中でも [MovieLens](https://movielens.org/) データセットは、おそらく最も人気のあるデータセットの 1 つです。MovieLens は、非商用の Web ベースの映画推薦システムです。1997年に作成され、ミネソタ大学の研究所であるGroupLensによって研究目的で映画のレーティングデータを収集するために運営されています。MovieLensのデータは、パーソナライズされたレコメンデーションや社会心理学など、いくつかの調査研究にとって重要です。 

## データの入手

MovieLens データセットは [GroupLens](https://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K dataset :cite:`Herlocker.Konstan.Borchers.ea.1999`) によってホストされています。このデータセットは、1682 本の映画の 943 人のユーザーからの 1 つ星から 5 つ星までの評価の $100,000$ から構成されています。各ユーザーが少なくとも20本の映画を評価するようにクリーンアップされました。年齢、性別、ユーザーのジャンル、アイテムなどの簡単な人口統計情報も利用できます。[ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) をダウンロードして `u.data` ファイルを抽出できます。このファイルには、すべての $100,000$ 評価が csv 形式で含まれています。このフォルダには他にも多くのファイルがあり、各ファイルの詳細な説明はデータセットの [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) ファイルにあります。 

はじめに、このセクションの実験を実行するのに必要なパッケージをインポートしましょう。

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

次に、MovieLens 100k データセットをダウンロードし、インタラクションを `DataFrame` として読み込みます。

```{.python .input  n=2}
#@save
d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## データセットの統計情報

データをロードして、最初の 5 つのレコードを手動で調べてみましょう。これは、データ構造を学習し、正しくロードされたことを確認する効果的な方法です。

```{.python .input  n=3}
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

各行は、「ユーザー ID」1 ～ 943、「アイテム ID」1 ～ 1682、「評価」1 ～ 5、「タイムスタンプ」を含む 4 つの列で構成されていることがわかります。サイズが $n \times m$ の交互作用行列を作成できます。$n$ と $m$ はそれぞれユーザー数とアイテム数です。このデータセットは既存の評価のみを記録するので、評価行列と呼ぶこともできます。この行列の値が正確な評価を表す場合は、交互作用行列と評価行列を同じ意味で使用します。ユーザーが映画の大部分を評価していないため、評価マトリックスのほとんどの値は不明です。このデータセットのスパース性も示します。スパース性は `1 - number of nonzero entries / ( number of users * number of items)` と定義されています。明らかに、交互作用行列はきわめてスパースです (スパース性 = 93.695%)。現実世界のデータセットは、より広範なスパース性に悩まされる可能性があり、レコメンダーシステムの構築において長年の課題となっています。実行可能な解決策は、ユーザー/アイテム機能などの追加の副次的な情報を使用して、スパース性を緩和することです。 

次に、異なる評価の度数の分布をプロットします。予想どおり、この分布は正規分布のようで、ほとんどの評価は3-4を中心にしています。

```{.python .input  n=4}
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## データセットの分割

データセットをトレーニングセットとテストセットに分割しました。次の関数は `random` と `seq-aware` を含む 2 つのスプリットモードを提供します。`random` モードでは、この関数はタイムスタンプを考慮せずに 100k の交互作用をランダムに分割し、既定ではデータの 90% をトレーニングサンプルとして使用し、残りの 10% を検定サンプルとして使用します。`seq-aware` モードでは、ユーザーがテスト用に最近評価した項目と、ユーザーの過去のインタラクションをトレーニングセットとして除外します。ユーザーインタラクションの履歴は、タイムスタンプに基づいて古いものから新しいものへとソートされます。このモードは、シーケンス認識型レコメンデーションセクションで使用されます。

```{.python .input  n=5}
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

テストセットだけではなく検証セットを実際に使用することをおすすめします。ただし、簡潔にするために省略しています。この場合、テストセットは保留された検証セットと見なすことができます。 

## データをロードする

データセットの分割後、便宜上、トレーニングセットとテストセットをリストとディクショナリ/マトリックスに変換します。次の関数は、データフレームを 1 行ずつ読み取り、0 から始まるユーザー/アイテムのインデックスを列挙します。この関数は、ユーザー、アイテム、評価、およびインタラクションを記録する辞書/マトリックスのリストを返します。フィードバックのタイプは `explicit` または `implicit` のいずれかに指定できます。

```{.python .input  n=6}
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

その後、上記の手順をまとめて、次のセクションで使用します。結果は `Dataset` と `DataLoader` でラップされます。トレーニングデータの `DataLoader` の `last_batch` は `rollover` モードに設定され (残りのサンプルは次のエポックにロールオーバーされます)、順序がシャッフルされることに注意してください。

```{.python .input  n=7}
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## [概要

* MovieLens データセットは、レコメンデーションリサーチに広く使用されています。公開されており、無料で利用できます。
* 後のセクションで MovieLens 100k データセットをダウンロードして前処理する関数を定義します。

## 演習

* 他に類似したレコメンデーションデータセットはありますか？
* MovieLens の詳細については、[https://movielens.org/](https://movielens.org/) のサイトを参照してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab:

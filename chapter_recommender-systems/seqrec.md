# シーケンス認識型レコメンダーシステム

これまでのセクションでは、ユーザーの短期的な行動を考慮せずに、レコメンデーションタスクをマトリックス完了問題として抽象化しました。このセクションでは、順番に並べられたユーザーインタラクションログを考慮したレコメンデーションモデルを紹介します。これはシーケンスを意識したレコメンダ :cite:`Quadrana.Cremonesi.Jannach.2018` で、入力は順序付けられ、多くの場合タイムスタンプが付けられた過去のユーザーアクションのリストです。最近の多くの文献では、このような情報をユーザーの時間的行動パターンのモデル化や関心のドリフトの発見に組み込むことの有用性が実証されています。 

今回紹介するモデル Caser :cite:`Tang.Wang.2018` は、畳み込みシーケンス埋め込み推奨モデルの略で、畳み込みニューラルネットワークを採用し、ユーザーの最近の活動の動的パターンの影響を捉えます。Caserの主要コンポーネントは、水平畳み込みネットワークと垂直畳み込みネットワークで構成され、それぞれユニオンレベルとポイントレベルのシーケンスパターンを明らかにすることを目的としています。ポイントレベルパターンは、履歴シーケンスの単一アイテムがターゲットアイテムに与える影響を示し、ユニオンレベルのパターンは、以前のいくつかのアクションが後続のターゲットに及ぼす影響を示します。たとえば、牛乳とバターを一緒に購入すると、小麦粉を購入するよりも小麦粉を購入する可能性が高くなります。さらに、ユーザーの一般的な関心や長期的な嗜好も、最後に完全に接続されたレイヤーでモデル化され、ユーザーの関心をより包括的にモデル化できます。モデルの詳細は以下の通りです。 

## モデルアーキテクチャ

シーケンス対応レコメンデーションシステムでは、各ユーザはアイテムセットの一部のアイテムのシーケンスに関連付けられます。$S^u = (S_1^u, ... S_{|S_u|}^u)$は順序付けられたシーケンスを表すとしましょう。Caserの目標は、ユーザーの一般的な好みと短期的な意図を考慮して商品を推薦することです。前の $L$ 項目を考慮して、タイムステップ $t$ の以前の交互作用を表す埋め込み行列を構築できるとします。 

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

$\mathbf{Q} \in \mathbb{R}^{n \times k}$ はアイテムの埋め込みを表し、$\mathbf{q}_i$ は $i^\mathrm{th}$ 行を表します。$\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ は、タイムステップ $t$ におけるユーザー $u$ の一時的な関心を推測するために使用できます。入力行列 $\mathbf{E}^{(u, t)}$ は、後続の 2 つの畳み込み成分の入力であるイメージとして見ることができます。 

水平畳み込み層には$d$個の水平フィルタ$\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$があり、垂直畳み込み層には$d'$個の垂直フィルタ$\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$があります。一連の畳み込み演算とプール演算を実行すると、次の 2 つの出力が得られます。 

$$
\mathbf{o} = \text{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \text{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

$\mathbf{o} \in \mathbb{R}^d$ は水平畳み込みネットワークの出力、$\mathbf{o}' \in \mathbb{R}^{kd'}$ は垂直畳み込みネットワークの出力です。簡略化のため、畳み込み演算とプール演算の詳細は省略しています。これらは連結され、完全に接続されたニューラルネットワーク層に供給され、より高レベルの表現が得られます。 

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

$\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ は重み行列で、$\mathbf{b} \in \mathbb{R}^k$ はバイアスです。学習されたベクトル $\mathbf{z} \in \mathbb{R}^k$ は、ユーザーの短期的意図の表現です。 

最後に、予測関数はユーザーの短期的嗜好と一般的な嗜好を組み合わせたもので、次のように定義されます。 

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

$\mathbf{V} \in \mathbb{R}^{n \times 2k}$ は別の項目埋め込み行列です。$\mathbf{b}' \in \mathbb{R}^n$ は項目固有のバイアスです。$\mathbf{P} \in \mathbb{R}^{m \times k}$ はユーザーの一般的な好みに合わせたユーザー埋め込み行列です。$\mathbf{p}_u \in \mathbb{R}^{ k}$ は $P$ の $u^\mathrm{th}$ 行で $\mathbf{v}_i \in \mathbb{R}^{2k}$ は $\mathbf{V}$ の $i^\mathrm{th}$ 行です。 

このモデルは、BPR またはヒンジ損失で学習できます。Caser のアーキテクチャを以下に示します。 

![Illustration of the Caser Model](../img/rec-caser.svg)

まず、必要なライブラリをインポートします。

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## モデルの実装以下のコードは Caser モデルを実装しています。垂直畳み込み層、水平畳み込み層、全結合層で構成されています。

```{.python .input  n=4}
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully-connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## 負サンプリングのシーケンシャルデータセットシーケンシャルインタラクションデータを処理するには、Dataset クラスを再実装する必要があります。次のコードでは、`SeqDataset` という名前の新しいデータセットクラスを作成します。各サンプルでは、ユーザー ID、前の $L$ が操作した項目をシーケンスとして、次に操作する項目がターゲットとして出力されます。次の図は、1 人のユーザーのデータロードプロセスを示しています。このユーザーが9本の映画を気に入ったと仮定し、これらの9本の映画を時系列順に整理します。最新のムービーはテスト項目として除外されます。残りの 8 本のムービーについては、3 つのトレーニングサンプルを取得できます。各サンプルには 5 本のムービー ($L=5$) のシーケンスが含まれ、それ以降のアイテムがターゲットアイテムになります。陰性サンプルもカスタマイズデータセットに含まれます。 

![Illustration of the data generation process](../img/rec-seq-data.svg)

```{.python .input  n=5}
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## MovieLens 100K データセットを読み込む

その後、MovieLens 100K データセットをシーケンス認識モードで読み込んで分割し、上記で実装したシーケンシャルデータローダーを使用してトレーニングデータを読み込みます。

```{.python .input  n=6}
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

トレーニングデータ構造は上図に示されています。最初の要素はユーザー ID で、次のリストはこのユーザーが気に入った最後の 5 つの項目を示し、最後の要素はこのユーザーが 5 つの項目に続いて気に入った項目です。 

## モデルを訓練するさあ、モデルを訓練させよう。前のセクションでは、学習率、オプティマイザー、$k$ などの NeumF と同じ設定を使用して、結果を比較できるようにします。

```{.python .input  n=7}
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
```

## 概要 * ユーザーの短期的および長期的な関心を推測することで、ユーザーが次に好む項目をより効果的に予測できます。* 畳み込みニューラルネットワークは、連続的なインタラクションからユーザーの短期的な関心を捉えるために利用できます。 

## 演習

* 水平および垂直の畳み込みネットワークの 1 つを削除して、アブレーション研究を実施します。どのコンポーネントがより重要ですか？
* ハイパーパラメーター $L$ を変化させます。過去のインタラクションが長くなると精度が高くなるのですか？
* 上記で紹介したシーケンス認識レコメンデーションタスクとは別に、セッションベースレコメンデーション :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015` と呼ばれる別のタイプのシーケンス認識レコメンデーションタスクがあります。この2つのタスクの違いを説明してもらえますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/404)
:end_tab:

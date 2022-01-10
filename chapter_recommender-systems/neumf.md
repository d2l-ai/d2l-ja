# パーソナライズされたランキングのためのニューラル協調フィルタリング

このセクションでは、明示的なフィードバックにとどまらず、暗黙的なフィードバックを伴うレコメンデーションのためのニューラル協調フィルタリング (NCF) フレームワークを紹介します。暗黙的なフィードバックは、レコメンダーシステムに広く浸透しています。クリック、購入、ウォッチなどのアクションは、収集しやすく、ユーザーの好みを示す一般的な暗黙的なフィードバックです。ここで紹介するモデル NeumF :cite:`He.Liao.Zhang.ea.2017` は、ニューラル行列因数分解の略で、暗黙的なフィードバックでパーソナライズされたランキングタスクに対処することを目的としています。このモデルは、ニューラルネットワークの柔軟性と非線形性を利用して行列因数分解の内積を置き換え、モデルの表現力を高めることを目的としています。具体的には、このモデルは一般化行列因数分解 (GMF) と MLP を含む 2 つのサブネットワークで構成され、単純なドット積ではなく 2 つの経路からの相互作用をモデル化します。これら 2 つのネットワークの出力は、最終的な予測スコア計算のために連結されます。AutoRec の評価予測タスクとは異なり、このモデルは暗黙的なフィードバックに基づいて各ユーザーにランク付けされたレコメンデーションリストを生成します。このモデルをトレーニングするには、前のセクションで紹介したパーソナライズされたランキング損失を使用します。 

## NeumF モデル

前述のとおり、NeumF は 2 つのサブネットワークを融合します。GMF は行列因数分解の汎用ニューラルネットワークバージョンで、入力はユーザー潜在因子と項目潜在因子の要素ごとの積になります。2つのニューラル層から構成されています。 

$$
\mathbf{x} = \mathbf{p}_u \odot \mathbf{q}_i \\
\hat{y}_{ui} = \alpha(\mathbf{h}^\top \mathbf{x}),
$$

$\odot$ はベクトルのアダマール積を表します。$\mathbf{P} \in \mathbb{R}^{m \times k}$ と $\mathbf{Q} \in \mathbb{R}^{n \times k}$ はそれぞれユーザー潜在行列と項目潜在行列に対応します。$\mathbf{p}_u \in \mathbb{R}^{ k}$ は $P$ の $u^\mathrm{th}$ 行で $\mathbf{q}_i \in \mathbb{R}^{ k}$ は $Q$ の $i^\mathrm{th}$ 行です。$\alpha$ と $h$ は活性化関数を表します。と出力層の重み. $\hat{y}_{ui}$ は、$u$ がアイテム $i$ に与える可能性のあるユーザーの予測スコアです。 

このモデルのもう 1 つのコンポーネントは MLP です。モデルの柔軟性を高めるため、MLP サブネットワークはユーザーとアイテムの埋め込みを GMF と共有しません。ユーザー埋め込みとアイテム埋め込みの連結を入力として使用します。複雑な接続と非線形変換により、ユーザーとアイテム間の複雑なインタラクションを推定することができます。より正確には、MLP サブネットワークは次のように定義されます。 

$$
\begin{aligned}
z^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = \left[ \mathbf{U}_u, \mathbf{V}_i \right] \\
\phi^{(2)}(z^{(1)})  &= \alpha^1(\mathbf{W}^{(2)} z^{(1)} + b^{(2)}) \\
&... \\
\phi^{(L)}(z^{(L-1)}) &= \alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})) \\
\hat{y}_{ui} &= \alpha(\mathbf{h}^\top\phi^L(z^{(L-1)}))
\end{aligned}
$$

$\mathbf{W}^*, \mathbf{b}^*$ と $\alpha^*$ は重み行列、バイアスベクトル、活性化関数を表します。$\phi^*$ は対応する層の関数を表し、$\mathbf{z}^*$ は対応する層の出力を表します。 

GMF と MLP の結果を融合するために、NeumF は 2 つのサブネットワークの最後から 2 番目の層を連結して、次の層に渡すことができる特徴ベクトルを作成します。その後、行列 $\mathbf{h}$ とシグモイド活性化関数を使用して出力が投影されます。予測層は次のように定式化されます:$$\ hat {y} _ {ui} =\ sigma (\ mathbf {h} ^\ top [\ mathbf {x},\ phi^L (z^ {(L-1)})])。$$ 

次の図は、NeumF のモデルアーキテクチャを示しています。 

![Illustration of the NeuMF model](../img/rec-neumf.svg)

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## モデルの実装以下のコードは NeumF モデルを実装しています。一般化行列分解モデルと、ユーザーおよび項目埋め込みベクトルが異なる MLP から構成されます。MLP の構造はパラメータ `nums_hiddens` で制御されます。ReLU はデフォルトのアクティベーション関数として使用されます。

```{.python .input  n=2}
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens,
                 **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu',
                                  use_bias=True))
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## ネガティブサンプリングを使用したカスタムデータセット

ペアワイズランキング損失の場合、重要なステップは負のサンプリングです。各ユーザーについて、ユーザーが操作していないアイテムは候補アイテム (観測されていないエントリ) です。次の関数は、ユーザーの識別項目と候補項目を入力として受け取り、そのユーザーの候補セットから各ユーザーの負の項目をランダムにサンプリングします。トレーニング段階では、ユーザーが好むアイテムが、嫌いなアイテムや操作したことのないアイテムよりも上位にランク付けされます。

```{.python .input  n=3}
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## Evaluator このセクションでは、時間による分割戦略を採用して、トレーニングセットとテストセットを構築します。特定のカットオフ $\ell$ ($\ text {Hit} @\ ell$) でのヒット率と ROC 曲線下の面積 (AUC) を含む 2 つの評価尺度を使用して、モデルの有効性を評価します。各ユーザーの特定のポジション $\ell$ でのヒット率は、おすすめアイテムが上位の $\ell$ ランキングに含まれているかどうかを示します。正式な定義は次のとおりです。 

$$
\text{Hit}@\ell = \frac{1}{m} \sum_{u \in \mathcal{U}} \textbf{1}(rank_{u, g_u} <= \ell),
$$

$\textbf{1}$ は、グラウンドトゥルースアイテムが上位の $\ell$ リストにランクされている場合は 1 に等しいインジケーター関数を表し、それ以外の場合はゼロに等しくなります。$rank_{u, g_u}$ は、ユーザー $u$ のグラウンドトゥルースアイテム $g_u$ のレコメンデーションリスト (理想的な順位は 1) の順位を表します。$m$ はユーザー数。$\mathcal{U}$ はユーザーセットです。 

AUC の定義は次のとおりです。 

$$
\text{AUC} = \frac{1}{m} \sum_{u \in \mathcal{U}} \frac{1}{|\mathcal{I} \backslash S_u|} \sum_{j \in I \backslash S_u} \textbf{1}(rank_{u, g_u} < rank_{u, j}),
$$

$\mathcal{I}$ はアイテムセットです。$S_u$ はユーザー $u$ の候補アイテムです。精度、再現率、正規化割引累積ゲイン (NDCG) など、他の多くの評価プロトコルも使用できることに注意してください。 

次の関数は、各ユーザのヒット数と AUC を計算します。

```{.python .input  n=4}
#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

すると、全体のヒット率とAUCは次のように計算されます。

```{.python .input  n=5}
#@save
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## モデルのトレーニングと評価

トレーニング関数は以下のように定義されています。モデルをペアワイズ方式で学習させます。

```{.python .input  n=6}
#@save
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[0:-1])]
                p_neg = [net(*t) for t in zip(*input_data[0:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

これで、MovieLens 100k データセットを読み込み、モデルをトレーニングできます。MovieLens データセットにはレーティングしかなく、精度がいくらか低下するため、これらのレーティングを 0 と 1 に 2 値化します。ユーザーがアイテムを評価した場合、暗黙的なフィードバックは 1 つ、それ以外の場合はゼロと見なされます。アイテムを評価するアクションは、暗黙的なフィードバックを提供する形式として扱うことができます。ここでは、データセットを `seq-aware` モードで分割し、ユーザーが最後に操作した項目はテスト対象外にします。

```{.python .input  n=11}
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items ), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

次に、モデルを作成して初期化します。隠れサイズ 10 が一定の 3 層 MLP を使用します。

```{.python .input  n=8}
devices = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
```

次のコードは、モデルをトレーニングします。

```{.python .input  n=12}
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## [概要

* 行列分解モデルに非線形性を追加すると、モデルの能力と有効性が向上します。
* NeumF は行列分解と多層パーセプトロンの組み合わせです。多層パーセプトロンは、ユーザー埋め込みとアイテム埋め込みの連結を入力として受け取ります。

## 演習

* 潜在因子の大きさを変える。潜在因子の大きさがモデルのパフォーマンスにどのような影響を与えるか
* MLP のアーキテクチャ (層数、各層のニューロン数など) を変更して、MLP が性能に与える影響を確認します。
* さまざまなオプティマイザー、学習率、体重減少率を試してください。
* 前のセクションで定義したヒンジ損失を使用して、このモデルを最適化します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/403)
:end_tab:

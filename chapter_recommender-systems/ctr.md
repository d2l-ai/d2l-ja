# 機能豊富なレコメンダーシステム

インタラクションデータは、ユーザーの好みや関心を示す最も基本的な指標です。これは、以前に導入されたモデルで重要な役割を果たしています。しかし、交互作用データは通常、きわめてまばらで、雑音が多くなることがあります。この問題に対処するために、アイテムの特徴、ユーザーのプロファイル、インタラクションが発生したコンテキストなどの副次的な情報をレコメンデーションモデルに統合することができます。これらの機能を利用すると、特にインタラクションデータが不足している場合に、これらの機能がユーザーの関心を効果的に予測できるという点で、レコメンデーションの作成に役立ちます。そのため、レコメンデーションモデルにはこれらの機能を処理し、モデルにコンテンツ/コンテキスト認識を与える機能も備えていることが不可欠です。このタイプのレコメンデーションモデルを示すために、オンライン広告レコメンデーション :cite:`McMahan.Holt.Sculley.ea.2013` のクリック率 (CTR) に関する別のタスクを導入し、匿名の広告データを提示します。ターゲットを絞った広告サービスは広く注目されており、レコメンデーションエンジンとして組み立てられることがよくあります。クリック率向上のためには、ユーザーの好みや興味に合った広告を推奨することが大切です。 

デジタルマーケティング担当者は、オンライン広告を使用して顧客に広告を表示します。クリックスルー率は、インプレッション数あたりの広告主様の広告のクリック数を測定する指標で、次の式で計算されるパーセンテージで表されます。  

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

クリックスルー率は、予測アルゴリズムの有効性を示す重要なシグナルです。クリックスルー率予測は、ウェブサイト上の何かがクリックされる可能性を予測するタスクです。クリック率予測モデルは、ターゲットを絞った広告システムだけでなく、一般的なアイテム（映画、ニュース、商品など）のレコメンダーシステム、メールキャンペーン、さらには検索エンジンにも採用できます。また、ユーザー満足度やコンバージョン率にも密接に関連しており、広告主が現実的な期待値を設定するのに役立つため、キャンペーンの目標を設定するのに役立ちます。

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## オンライン広告データセット

インターネットとモバイルテクノロジーの大幅な進歩により、オンライン広告は重要な収入源となり、インターネット業界で大部分の収益を生み出しています。カジュアルな訪問者が有料顧客に変わるためには、関連性の高い広告やユーザーの興味をそそる広告を表示することが重要です。今回紹介したデータセットは、オンライン広告のデータセットです。34 個のフィールドで構成され、最初の列は広告がクリックされたか (1) かクリックされなかったか (0) を示すターゲット変数を表します。その他の列はすべてカテゴリカルフィーチャです。列には、アドバタイズメント ID、サイト ID またはアプリケーション ID、デバイス ID、時刻、ユーザプロファイルなどが表示されます。匿名化とプライバシーの懸念から、機能の実際のセマンティクスは公開されていません。 

次のコードは、データセットをサーバーからダウンロードし、ローカルデータフォルダーに保存します。

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

トレーニングセットとテストセットがあり、それぞれ 15000 サンプル/ラインと 3000 サンプル/ラインで構成されます。 

## データセットラッパー

データの読み込みに便利なように、CSV ファイルから広告データセットを読み込み、`DataLoader` で使用できる `CTRDataset` を実装しています。

```{.python .input  n=13}
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

次の例では、トレーニングデータをロードし、最初のレコードを出力します。

```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

見てわかるように、34 個のフィールドはすべてカテゴリフィーチャです。各値は、対応するエントリのワンホットインデックスを表します。$0$ というラベルは、クリックされていないことを意味します。この `CTRDataset` は、Criteo ディスプレイ広告チャレンジ [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) や Avazu クリックスルー率予測 [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction) など、他のデータセットの読み込みにも使用できます。   

## 概要 * クリック率は、広告システムやレコメンダーシステムの効果を測定するために使用される重要な指標です。* クリックスルー率の予測は、通常、二項分類問題に変換されます。ターゲットは、特定の特徴に基づいて、広告/アイテムがクリックされるかどうかを予測することです。 

## 演習

* 提供されている `CTRDataset` で Criteo および Avazu データセットをロードできますか。Criteoデータセットは実数値の特徴量で構成されているため、コードを少し修正する必要があるかもしれません。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:

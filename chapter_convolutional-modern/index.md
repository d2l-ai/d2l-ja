# 最新の畳み込みニューラルネットワーク
:label:`chap_modern_cnn`

CNN の配線の基本を理解したところで、次は最新の CNN アーキテクチャのツアーを紹介します。この章の各セクションは、ある時点で (または現在) 多くの研究プロジェクトと配備されたシステムが構築されたベースモデルであった重要な CNN アーキテクチャに対応しています。これらのネットワークはそれぞれ一時的に主要なアーキテクチャであり、その多くはImageNetコンペティションの優勝者または準優勝者でした。ImageNetコンペティションは、2010年以降、コンピュータービジョンの教師あり学習の進歩のバロメーターとして機能してきました。 

これらのモデルには、大規模なビジョンの課題で従来のコンピュータービジョン手法を打ち負かすために導入された最初の大規模ネットワークであるAlexNet、多数の要素の繰り返しブロックを利用するVGGネットワーク、ニューラルネットワーク全体をパッチワイズで畳み込むネットワークインネットワーク（NiN）が含まれます。input、並列連結のネットワークを使用する GoogleNet、残差ネットワーク (ResNet) は、コンピュータービジョンで最も一般的な市販のアーキテクチャであり続けています。密に接続されたネットワーク (DenseNet) は、計算にコストがかかりますが、最近いくつかのベンチマークを設定しています。 

*ディープ* ニューラルネットワークの考え方は非常に単純ですが (たくさんのレイヤーを積み重ねる)、パフォーマンスはアーキテクチャやハイパーパラメーターの選択によって大きく異なります。この章で説明するニューラルネットワークは、直感、いくつかの数学的洞察、そして多くの試行錯誤の産物です。これらのモデルを時系列順に提示します。これは、その分野がどこに向かっているのかについて独自の直感を形成し、おそらく独自のアーキテクチャを開発できるように、歴史の感覚を伝えるためです。たとえば、この章で説明するバッチ正規化と残差結合は、ディープモデルの学習と設計における 2 つの一般的なアイデアを提供してきました。

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```
# 深層学習入門

2016年のこと、 Joel Grusという有名なデータサイエンティストが、有名なインターネットの会社で、[仕事を得るための面接](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)を受けました。よくある話ですが、面接官の一人は彼のプログラミングスキルを確認するために、あるタスクを与えました。そのタスクとは、FizzBuzzという単純な子供の遊びを実装するというものでした。その遊びとは、プレイヤーは1からカウントを始め、3で割れる数字のときは'Fizz'と、５で割れる数字のときは'Buzz'と、そして15で割れる数字のときは'FizzBuzz'というものです。つまり、プレイヤーはこういった数字の並びを作っていくのです。

```1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 ...```

実際に面接で起こったことは、とても予想できないものでした。その問題を、数行のPythonのコードで*アルゴリズム的に*解くのではなく、その問題をデータを利用して解こうとしたのです。彼は、(3, fizz), (5, buzz), (7, 7), (2, 2), (15, fizzbuzz) といった形式のペアを、このタスクのための分類器を学習するデータとして利用しました。そして、小さなニューラルネットワークを設計して、そのデータを使って学習し、非常に高い精度を得たのです（面接官は完全に当惑し、彼は仕事を得ることができませんでした）。


この面接の状況は、コンピュータサイエンスにおいて、プログラムの設計がデータによるプログラミングによって補完される（ときどき完全に置き換えられる）瞬間であったことは間違いないでしょう。そして、データによるプログラミングが今現在、容易にできることを示した点において、意義深いものといえるでしょう（面接という意味ではそうではなかったですが）。FizzBuzzを上で述べたような方法で真剣に解く人はいないでしょうが、それとは全く異なる方向性で、顔を認識したり、人間の音声やテキストから感情を分類したり、音声を認識したりすることにもつながるのです。たいていのソフトウェアエンジニアは、良いアルゴリズム、たくさんの計算資源とデータ、良いソフトウェアツールによって、洗練されたモデルを構築し、10年前は優秀なサイエンティストにとっても難しいと考えられていた問題さえ、解くことができるようになっています。

この本では、そうした過程に立っているエンジニアを助けることを目的とします。われわれは、数学、コード、実装例をすぐに利用できるパッケージとして組み合わせ、機械学習を実践可能しようとしています。Jupyter notebookはオンラインで利用可能で、ノートパソコンでもクラウドのサーバでも実行できます。新しい世代のプログラマー、アントレプレナー、統計学者、生物学者、機械学習に興味ある人々が、Jupyter notebookを利用して、問題解決のために進歩した機械学習アルゴリズムを実装できるようになることを望みます。

## データを利用してプログラミングする

コードを利用してプログラミングすることと、データを利用してプログラミングすることの違いについてもう少し見てみましょう。それは、思ったよりも深いものかもしれません。たいていの既存のプログラムは機械学習を必要としません。例えば、もし電子レンジのユーザインタフェースを記述しようと思ったら、わずかなボタンで設計することが可能で、ほとんど労力はかかりません。様々な条件を想定して、電子レンジに関する動作を精密に表すロジックやルールを追加すれば完了です。同様に、社会保障番号の妥当性をチェックするプログラムは、たくさんのルールが当てはまるかどうかのテストを必要とします。例えば、その番号は９桁で、000からは始まらない、といったルールです。

上記の二つの例で着目すべきなのは、プログラムのロジックを理解するためにデータを集める必要がないことと、そのデータから特徴を取り出す必要がないことです。十分な時間がある限り、常識やアルゴリズムに関するスキルがあればタスクをこなすことができるのです。

これまで見てきたように、多くの子供や動物がいともたやすく解けるような問題にもかかわらず、それをプログラミングすることは、最も優れたプログラマーであっても難しいという例が多く存在します。画像のなかに猫が写ってるかどうかを判別する問題を考えてみましょう。どこから始めるのが良さそうでしょうか。問題をまずは単純化してみます。つまり、全てのイメージは同じ大きさ(たとえば、400x400 pixels)で、各ピクセルには赤、緑、青のの値が入っており、すなわち画像は480,000の値で表現されます。猫を検出するために関係しそうな情報がどこにあるか、これを決めることはほとんど不可能です。全ての値の平均値、四隅の値、画像内のある一点の値でしょうか。実際、画像の内容を解釈するためには、エッジ、テクスチャ、形、目、鼻などの数千もの値を組み合わせによって現れる特徴を探す必要があります。そのときようやく、その画像に猫が写っているかどうかを決定することができます。

それに変わる手段として、最終的に求めているものにもとづいて、解法を探す方法があります。つまり、画像の例と求めている結果（猫か猫でないか）を出発点とする*データでプログラミングする*ことです。インターネット上では人気な猫の実画像やそれ以上の情報を収集することができます。そして、画像のなかに猫が写っているかどうかを*学習*できる関数を探すというゴールに置き換えます。それは通常、特定の関数の形（例えば多項式の関数）をエンジニアが選んで、そのパラメータをデータから*学習*することを意味します。

一般的に機械学習は、猫の認識のような問題に利用できる幅広い種類の関数を扱います。特に、深層学習はニューラルネットワークにもとづく特定の関数を利用し、特殊な方法でそれらをトレーニング（つまり、それらの関数のパラメータを計算）します。近年、大量のデータと高性能なハードウェアによって、画像、テキスト、音声信号といった複雑で高次元なデータを処理する方法として、深層学習は標準的な選択肢になりつつあります。

## 起源

深層学習は最近の発明ですが、人間は何世紀にも渡って、データの分析や将来の予測をしたいと考えてきました。実際、自然科学の多くが深層学習の起源になっています。例えば、ベルヌーイ分布は[Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli)の名をとっており、ガウス分布は[Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)によって発見されました。彼は、例えば最小二乗法を発見し、今日でも保険の計算や医療診断に至る様々な問題に利用されています。これらのツールは自然科学における実験的なアプローチによってうまれたものです。例えば、抵抗における電流と電圧に関係するオームの法則は、完全に線形モデルとして記述することができます。

中世においても、数学者は予測に対して熱意を注いできました。例えば、[Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)による幾何学の本は、平均的な足の長さを得るために、16人の成人男性の足の長さの平均を計算することを記述しています。

![Estimating the length of a foot](../img/koebel.jpg)

図1.1 はこの予測がどの機能するのかを示しています。16人の成人男性が、教会から出る際に1列に並ぶように言われていました。1フィートの値を見積もるために、彼らの足の長さの総和を16で割ったのでした。そのアルゴリズムは後に、特異な長さを処理するように改善されており、最も短いものと最も長いものを取り除き、残りについて平均をとっています。これはtrimmed mean とよばれる推定手法の始まりといえます。

統計学者は、データを収集して、利用できるようにするところからはじめました。偉大な統計学者の一人、[Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher)は、その理論と遺伝学への応用に関して多大な貢献をしました。彼の多くの線形判別分析のようなアルゴリズムやフィッシャー情報行列といった数式は、今日でも頻繁に利用されています（そして、彼が1936年にリリースしたIrisのデータセットも、機械学習を図解するために利用されています）。

機械学習にとって2番目の影響は、 [(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon)による情報理論と[Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing)による計算理論によるものでしょう。Turingは、[Computing machinery and intelligence](https://www.jstor.org/stable/2251299) (Mind, October 1950)という著名な論文の中で、「機械は考えることができるか?」という質問を投げかけました。彼がTuring testと述べていたものは、もし人間が文章で対話をしているときに、相手からの返答が機械か人間かどちらか判別がつかなければ、機械は知能があるとみなすものでした。今日に至るまで、知的な機械の開発は急速かつ継続的に変わり続けています。

もう一つの影響は神経科学や心理学の分野にみられるでしょう。人間というものは知的な行動をはっきりと示します。そこで、知的な行動を説明したり、その本質を逆解析したりできないかと考えることは当然のように思います。これを達成した最古のアルゴリズムの1つは[Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb)によって定式化されました。

彼の革新的な書籍 [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949)では、ニューロンは正の強化 (Positive reinforcement)によって学習すると彼は断言しています。このことは、Hebbian learning rule として知られるようになりました。。それは、Rosenblatt のパーセプトロンの学習アルゴリズムのプロトタイプであり、今日の深層学習を支えている確率的勾配降下法の多くの基礎を築きました。つまり、ニューラルネットワークにおいて適切な重みを学習するために、求められる行動を強化し、求められない行動を減らすことなのです。

生物学的な観点からの着想という意味では、まず、ニューラルネットワークが名前として与えられている通りです。1世紀以上にわたって （1873年のAlexander Bainや1890年のJames Sherringtonにさかのぼります）、研究者はニューロンを相互作用させるネットワークに類似した計算回路を組み立てようとしていました。時代をこえ、生物学にの解釈はやや薄まってしまいましたが、名前は依然として残っています。その核となる部分には、今日のたいていのネットワークにもみられる重要な原理がわずかに残っています。

* 線形あるいは非線形の演算ユニットについて、しばしば「レイヤー」と呼ばれます。
* 全体のネットワークのパラメータをただちに調整するために連鎖律 (Chain rule, 誤差逆伝播法とも呼ばれる)を使用します。


最初の急速な進歩のあと、ニューラルネットワークの研究は1995年から2005年まで衰えてしまいました。これにはたくさんの理由があります。まず、ネットワークを学習することは膨大な計算量を必要とします。RAMは20世紀の終わりには十分なものとなりましたが、計算力は乏しいものでした。次に、データセットが比較的小さいものでした。実際のところ、1932年から存在するFisherの'Iris dataset'はアルゴリズムの能力をテストするためによく利用されたツールでした。60,000もの手書き数字からなるMNISTは巨大なものと捉えられていました。

データや計算力が乏しいがゆえに、カーネル法や決定木、グラフィカルモデルといった強力な統計的ツールが、経験的に優れていることを示していました。これらのツールは、ニューラルネットワークとは違って、学習に数週間もかかりませんし、強力な理論的保証のもと予測結果をもたらしてくれました。

## 深層学習への道程

World Wide Web、数百万のオンラインユーザをかかえる企業の到来、安価で高性能なセンサー、安価なストレージ (Kryderの法則)、安価な計算機 (Mooreの法則)、特に当初はコンピュータゲーム用に開発されたGPUの普及によって、大量のデータを利用することが可能になり、多くのことが変わりました。突然、これまでは計算量的に実行困難と思われたアルゴリズムやモデルが価値をもつようになったのです（逆もまた然りで、このようなアルゴリズム、モデルがあったからこそ、データが価値をもったのです）。以下の表は、このことを最も良く表しています。

|年代|データセット|メモリ|1秒あたりの浮動小数点演算|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 PF (Nvidia DGX-2)|


非常に明白な点としては、RAMはデータの成長に追いついていないということです。同時に、演算能力は利用可能なデータよりも、速い速度で進歩しています。従って、統計モデルはメモリ効率がより良いものである必要があります（これは通常、非線形性を導入することで実現されます）。また同時に、向上した演算能力によって、より多くの時間をパラメータの最適化に使用することができます。次に、機械学習と統計学の大きな強みは、（一般化）線形モデルやカーネル法から深層学習へと移りました。この理由の一つとしては、深層学習の主力である多層パーセプトロン (例えば、McCulloch & Pitts, 1943)ｍ畳み込みニューラルネットワーク (Le Cun, 1992)、Long Short Term Memory (Hochreiter & Schmidhuber, 1997)、Q-Learning (Watkins, 1989)が、長い間眠っていた後に、ここ10年で根本的に*再発見*されたことが挙げられます。

統計モデル、アプリケーション、そしてアルゴリズムにおける最近の進歩は、カンブリア紀の爆発、つまり生命の種の進化における急速な発展とリンクするかもしれません。実際のところ、現在の最新の技術というものは、利用可能なリソースから成り行きでできあがったものではなく、過去数十年のアルゴリズムにも利用されているのです。

* ネットワークの能力を制御する新しい手法の一例であるDropout [3] は、過学習のリスクをおさえる、つまり大部分の学習データをそのまま記憶してしまうことを避けて、比較的大きなネットワークを学習することを可能にします。これは、ノイズをネットワークに追加 することによって実現され [4]、学習のために重みを確率変数に置き換えます。

* Attentionの仕組みは、1世紀に渡って統計学の分野で悩みとなっていた問題を解決しました。その問題とは、学習可能なパラメータの数を増やすこと無く、システムの記憶量や複雑性を向上させるというものです。Bahdanauらは、学習可能なポインタ構造とみなせる方法を利用した、洗練された解法を発見しました [5]。ある固定次元の表現を利用した機械翻訳を例にとると、全体の文章を記憶するというよりはむしろ、翻訳プロセスにおける中間表現へのポインタを保存すればよいということを意味します。翻訳文のような文章を生成するまえに、翻訳前の文章全体を記憶する必要がもはやなくなり、長文に対して
大きな精度の向上を可能にしました。

* Memory Networks[6]やNeural Programmer-Interpreter[7]のような多段階設計は、統計的なモデル化を行う人々に対して、推論のための反復的なアプローチを記述可能にします。
これらの手法は、繰り返し修正される深層学習の内部状態を想定し、推論の連鎖において以降のステップを実行します。それは、計算のためにメモリの内容が変化するプロセッサに似ています。

* もうひとつの重要な成果は、Generative Adversarial Networks [8]の発明でしょう。従来、確率密度を推定する統計的手法と生成モデルは、妥当な確率分布を発見することと、その確率分布からサンプリングを行う（しばしば近似的な）アルゴリズムに着目してきました。結果としてこれらのアルゴリズムは、統計的モデルにみられるように柔軟性が欠けており、大きく制限されたものになっていました。GANにおける重要な発明は、そのサンプリングを行うsamplerを、微分可能なパラメータをもつ任意のアルゴリズムで置き換えたことにあります。これらのパラメータは、discriminator (真偽を判別するテスト)が、真のデータと偽のデータの区別がつかないように調整されます。任意のアルゴリズムでそのデータを生成できるため、様々な種類の技術において、確率密度の推定が可能になりました。人が乗って走るシマウマ[9]や偽の有名人の顔[10]の例は、両方とも、この技術の進歩のおかげです。

* 多くの場合において、学習に利用可能な大量のデータを処理するにあたって、単一のGPUは不十分です。過去10年にわたって、並列分散学習を実装する能力は非常に進歩しました。スケーラブルなアルゴリズムを設計する際に、最も重要な課題のうちの一つとなったのは、確率的勾配降下法のような深層学習の最適化の能力が、処理される比較的小さなミニバッチに依存するという点です。同時に、小さなバッチはGPUの効率を制限します。それゆえに、ミニバッチのサイズが、例えば1バッチあたり32画像で、それを1024個のGPUで学習することは、32,000枚の画像の集約されたミニバッチと等しくなります。最近の研究として、まずはじめにLi [11]、次にYouら[12]、そしてJiaら[13]が、そのサイズを64,000の観測データにまで向上させ、ImageNetを利用したResNet50の学習時間を7分以下にまで短縮しました。比較のため、当初の学習時間は日単位で計測されるようなものでした。

* 計算を並列化する能力は、強化学習の発展、少なくともシミュレーションが必要となる場面において、大きな貢献を果たしています。これによって、碁、Atariのゲーム、Starcraftにおいてコンピュータが人間の能力を超え、物理シミュレーション(例えば、MuJoCoの利用)においても大きな発展をもたらしました。AlphaGoがこれをどのように達成したかを知りたい場合は、Silverらの文献[18]を見てください。強化学習が最も上手くいくのは、(状態, 行動, 報酬)の3つ組が十分に利用できるとき、つまり、それらの3つ組が互いにどう関係し合うかを学習するために多くの試行が可能なときなのです。シミュレーションはそのための手段を与えてくれます。

* 深層学習のフレームワークは、考えを広める上で重要な役割を果たしてきました。容易なモデリングを可能にする最初の世代のフレームワークとしては、[Caffe](https://github.com/BVLC/caffe)、[Torch](https://github.com/torch)、  [Theano](https://github.com/Theano/Theano)があります。
多くの影響力のある論文はこれらのフレームワークを用いて記述されました。今までに、こうしたフレームワークは[TensorFlow](https://github.com/tensorflow/tensorflow)や、そのハイレベルなAPIとして利用される[Keras](https://github.com/keras-team/keras)、 [CNTK](https://github.com/Microsoft/CNTK)、[Caffe 2](https://github.com/caffe2/caffe2)、そして[Apache MXNet](https://github.com/apache/incubator-mxnet)によって置き換えられてきました。第3世代のフレームワークは、主に深層学習のために必要な指示を記述していくimperativeな手法で、おそらく[Chainer](https://github.com/chainer/chainer)によって普及し、それはPython NumPyを利用してモデルを記述するのに似た文法を使用しました。この考え方は[PyTorch](https://github.com/pytorch/pytorch)やMXNetの[Gluon API](https://github.com/apache/incubator-mxnet)にも採用されています。後者のGluon APIは、このコースで深層学習を教えるために利用します。

学習のためにより良い手法を構築するシステム研究者と、より良いネットワークを構築する統計的なモデル化を行う人々の仕事を分けることは、ものごとを非常にシンプルにしてきました。例えば、線形ロジスティック回帰モデルを学習することは簡単な宿題とはいえず、2014年のカーネギーメロン大学における、機械学習を専門とする博士課程の新しい学生に与えるようなものでした。今では、このタスクは10行以下のコードで実装でき、このことはプログラマーによって確かに理解されるようになりました。

## サクセスストーリー

人工知能は、他の方法では実現困難だったものに対して、成果を残してきた長い歴史があります。例えば、郵便は光学式文字読取装置(OCR)によって並び替えらています。これらのシステムは（有名なMNISTやUSPSといった手書き数字のデータセットをもとにして）、90年台に登場したものです。同様のことが、預金額の読み取りや、申込者の信用力の評価にt形容されています。金融の取引は不正検知のために自動でチェックされています。これによって、PayPal、 Stripe、AliPay、WeChat、Apple、Visa、MasterCardといった多くの電子商取引の支払いシステムの基幹ができあがりました。機械学習は、インターネット上の検索、レコメンデーション、パーソナライゼーション、ランキングも動かしています。言い換えれば、人工知能と機械学習はあらゆるところに浸透していて、しばしば見えないところで動いているのです。

最近になって、以前は解くことが困難と思われた問題へのソリューションとして、AIが脚光を浴びています。

* AppleのSiri、AmazonのAlexa、Google assistantといった知的なアシスタントは、利用可能なレベルの精度で話し言葉の質問に回答することができます。これらは、証明のスイッチをONにする（身体障害者にとってはありがたい）、理髪店の予約をする、電話サポートにおける会話を提供する、といった単純なタスクをこなします。これは、AIがわれわれの生活に影響をもたらしている最も顕著なサインであるように思えます。

* A key ingredient in digital assistants is the ability to recognize speech accurately. Gradually the accuracy of such systems has increased to the point where they reach human parity [14] for certain applications.
* Object recognition likewise has come a long way. Estimating the object in a picture was a fairly challenging task in 2010. On the ImageNet benchmark Lin et al. [15] achieved a top-5 error rate of 28%. By 2017 Hu et al. [16] reduced this error rate to 2.25%. Similarly stunning results have been achieved for identifying birds, or diagnosing skin cancer.
* Games used to be a bastion of human intelligence. Starting from TDGammon [23], a program for playing Backgammon using temporal difference (TD) reinforcement learning, algorithmic and computational progress has led to algorithms for a wide range of applications. Unlike Backgammon, chess has a much more complex state space and set of actions. DeepBlue beat Gary Kasparov, Campbell et al. [17], using massive parallelism, special purpose hardware and efficient search through the game tree. Go is more difficult still, due to its huge state space. AlphaGo reached human parity in 2015,  Silver et al. [18] using Deep Learning combined with Monte Carlo tree sampling. The challenge in Poker was that the state space is large and it is not fully observed (we don't know the opponents' cards). Libratus exceeded human performance in Poker using efficiently structured strategies; Brown and Sandholm [19]. This illustrates the impressive progress in games and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars and trucks. While full autonomy is not quite within reach yet, excellent progress has been made in this direction, with companies such as [Momenta](https://www.momenta.ai/en), [Tesla](http://www.tesla.com), [NVIDIA](http://www.nvidia.com), [MobilEye](http://www.mobileye.com) and [Waymo](http://www.waymo.com) shipping products that enable at least partial autonomy. What makes full autonomy so challenging is that proper driving requires the ability to perceive, to reason and to incorporate rules into a system. At present, Deep Learning is used primarily in the computer vision aspect of these problems. The rest is heavily tuned by engineers.

Again, the above list barely scratches the surface of what is considered intelligent and where machine learning has led to impressive progress in a field. For instance, robotics, logistics, computational biology, particle physics and astronomy owe some of their most impressive recent advances at least in parts to machine learning. ML is thus becoming a ubiquitous tool for engineers and scientists.

Frequently the question of the AI apocalypse, or the AI singularity has been raised in non-technical articles on AI. The fear is that somehow machine learning systems will become sentient and decide independently from their programmers (and masters) about things that directly affect the livelihood of humans. To some extent AI already affects the livelihood of humans in an immediate way - creditworthiness is assessed automatically, autopilots mostly navigate cars safely, decisions about whether to grant bail use statistical data as input. More frivolously, we can ask Alexa to switch on the coffee machine and she will happily oblige, provided that the appliance is internet enabled.

Fortunately we are far from a sentient AI system that is ready to enslave its human creators (or burn their coffee). Firstly, AI systems are engineered, trained and deployed in a specific, goal oriented manner. While their behavior might give the illusion of general intelligence, it is a combination of rules, heuristics and statistical models that underlie the design. Second, at present tools for general Artificial Intelligence simply do not exist that are able to improve themselves, reason about themselves, and that are able to modify, extend and improve their own architecture while trying to solve general tasks.

A much more realistic concern is how AI is being used in our daily lives. It is likely that many menial tasks fulfilled by truck drivers and shop assistants can and will be automated. Farm robots will likely reduce the cost for organic farming but they will also automate harvesting operations. This phase of the industrial revolution will have profound consequences on large swaths of society (truck drivers and shop assistants are some of the most common jobs in many states). Furthermore, statistical models, when applied without care can lead to racial, gender or age bias. It is important to ensure that these algorithms are used with great care. This is a much bigger concern than to worry about a potentially malevolent superintelligence intent on destroying humanity.

## Key Components

Machine learning uses data to learn transformations between examples. For instance, images of digits are transformed to integers between 0 and 9, audio is transformed into text (speech recognition), text is transformed into text in a different language (machine translation), or mugshots are transformed into names (face recognition). In doing so, it is often necessary to represent data in a way suitable for algorithms to process it. This degree of feature transformations is often used as a reason for referring to deep learning as a means for representation learning (in fact, the International Conference on Learning Representations takes its name from that). At the same time, machine learning equally borrows from statistics (to a very large extent questions rather than specific algorithms) and data mining (to deal with scalability).

The dizzying set of algorithms and applications makes it difficult to assess what *specifically* the ingredients for deep learning might be. This is as difficult as trying to pin down required ingredients for pizza - almost every component is substitutable. For instance one might assume that multilayer perceptrons are an essential ingredient. Yet there are computer vision models that use only convolutions. Others only use sequence models.

Arguably the most significant commonality in these methods is the use of end-to-end training. That is, rather than assembling a system based on components that are individually tuned, one builds the system and then tunes their performance jointly. For instance, in computer vision scientists used to separate the process of feature engineering from the process of building machine learning models.
The Canny edge detector [20] and Lowe's SIFT feature extractor [21]  reigned supreme for over a decade as algorithms for mapping images into feature vectors. Unfortunately, there is only so much that humans can accomplish by ingenuity relative to a consistent evaluation over thousands or millions of choices, when carried out automatically by an algorithm. When deep learning took over, these feature extractors were replaced by automatically tuned filters, yielding superior accuracy.

Likewise, in Natural Language Processing the bag-of-words model of Salton and McGill [22] was for a long time the default choice. In it, words in a sentence are mapped into a vector, where each coordinate corresponds to the number of times that particular word occurs. This entirely ignores the word order ('dog bites man' vs. 'man bites dog') or punctuation ('let's eat, grandma' vs. 'let's eat grandma'). Unfortunately, it is rather difficult to engineer better features *manually*. Algorithms, conversely, are able to search over a large space of possible feature designs automatically. This has led to tremendous progress. For instance, semantically relevant word embeddings allow reasoning of the form 'Berlin - Germany + Italy = Rome' in vector space. Again, these results are achieved by end-to-end training of the entire system.

Beyond end-to-end training, the second most relevant part is that we are experiencing a transition from parametric statistical descriptions to fully nonparametric models. When data is scarce, one needs to rely on simplifying assumptions about reality (e.g. via spectral methods) in order to obtain useful models. When data is abundant, this can be replaced by nonparametric models that fit reality more accurately. To some extent, this mirrors the progress that physics experienced in the middle of the previous century with the availability of computers. Rather than solving parametric approximations of how electrons behave by hand, one can now resort to numerical simulations of the associated partial differential equations. This has led to much more accurate models, albeit often at the expense of explainability.

A case in point are Generative Adversarial Networks, where graphical models were replaced by data generating code without the need for a proper probabilistic formulation. This has led to models of images that can look deceptively realistic, something that was considered too difficult for a long time.

Another difference to previous work is the acceptance of suboptimal solutions, dealing with nonconvex nonlinear optimization problems, and the willingness to try things before proving them. This newfound empiricism in dealing with statistical problems, combined with a rapid influx of talent has led to rapid progress of practical algorithms (albeit in many cases at the expense of modifying and re-inventing tools that existed for decades).

Lastly, the Deep Learning community prides itself of sharing tools across academic and corporate boundaries, releasing many excellent libraries, statistical models and trained networks as open source. It is in this spirit that the notebooks forming this course are freely available for distribution and use. We have worked hard to lower the barriers of access for everyone to learn about Deep Learning and we hope that our readers will benefit from this.

## Summary

* Machine learning studies how computer systems can use data to improve performance. It combines ideas from statistics, data mining, artificial intelligence and optimization. Often it is used as a means of implementing artificially intelligent solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. This is often accomplished by a progression of learned transformations.
* Much of the recent progress has been triggered by an abundance of data arising from cheap sensors and internet scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining good performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

## Problems

1. Which parts of code that you are currently writing could be 'learned', i.e. improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using Deep Learning.
1. Viewing the development of Artificial Intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
1. Where else can you apply the end-to-end training approach? Physics? Engineering? Econometrics?
1. Why would you want to build a Deep Network that is structured like a human brain? What would the advantage be? Why would you not want to do that (what are some key differences between microprocessors and neurons)?

## References

[1] Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433.

[2] Hebb, D. O. (1949). The organization of behavior; a neuropsychological theory. A Wiley Book in Clinical Psychology. 62-78.

[3] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.

[4] Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7(1), 108-116.

[5] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[6] Sukhbaatar, S., Weston, J., & Fergus, R. (2015). End-to-end memory networks. In Advances in neural information processing systems (pp. 2440-2448).

[7] Reed, S., & De Freitas, N. (2015). Neural programmer-interpreters. arXiv preprint arXiv:1511.06279.

[8] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

[9] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint.

[10] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196.

[11] Li, M. (2017). Scaling Distributed Machine Learning with System and Algorithm Co-design (Doctoral dissertation, PhD thesis, Intel).

[12] You, Y., Gitman, I., & Ginsburg, B. Large batch training of convolutional networks. ArXiv e-prints.

[13] Jia, X., Song, S., He, W., Wang, Y., Rong, H., Zhou, F., … & Chen, T. (2018). Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes. arXiv preprint arXiv:1807.11205.

[14] Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., … & Zweig, G. (2017, March). The Microsoft 2016 conversational speech recognition system. In Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on (pp. 5255-5259). IEEE.

[15] Lin, Y., Lv, F., Zhu, S., Yang, M., Cour, T., Yu, K., … & Huang, T. (2010). Imagenet classification: fast descriptor coding and large-scale svm training. Large scale visual recognition challenge.

[16] Hu, J., Shen, L., & Sun, G. (2017). Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 7.

[17] Campbell, M., Hoane Jr, A. J., & Hsu, F. H. (2002). Deep blue. Artificial intelligence, 134 (1-2), 57-83.

[18] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., … & Dieleman, S. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529 (7587), 484.

[19] Brown, N., & Sandholm, T. (2017, August). Libratus: The superhuman ai for no-limit poker. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence.

[20] Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.

[21] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[22] Salton, G., & McGill, M. J. (1986). Introduction to modern information retrieval.

[23] Tesauro, G. (1995), Transactions of the ACM, (38) 3, 58-68

## Discuss on our Forum

<div id="discuss" topic_id="2310"></div>

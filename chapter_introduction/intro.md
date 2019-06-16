# はじめに

日々利用しているほとんどすべてのコンピュータのプログラムは、最近になるまで、ソフトウェア開発者によって第一原理にもとづいて開発されています。e-commerceのぷらっとをフォームを管理するアプリケーションを作成したい場合を考えましょう。その問題を考えるために数時間、ホワイトボードに集まって、以下のような上手くいきそうなソリューションを書いたとします。

(i) ユーザはウェブブラウザやモバイルのアプリケーションからインターフェースを介して、そのアプリケーションとやり取りする

(ii) そのアプリケーションは、ユーザの状態を追跡したり、すべての処理履歴を管理するために商用のデータベースを利用する

(iii) アプリケーションの中心に、複数のサーバ上で並列実行される、*ビジネスロジック* (*ブレーン*と呼ぶかもしれません)が、考えられる状況に応じた適切なアクションを緻密に計画する

そうした*ブレーン*をもつアプリケーションを構築するためには、起こりうるすべての特別なケースを考慮して、適切なルールを作成しなければなりません。顧客が商品を買い物かごに入れるためにクリックするたびに、買い物かごデータベースにエントリを追加し、購入する商品のIDと顧客IDを関連付けます。これを一度で完全に理解できる開発者はほとんどいないでしょう（問題を解決するために、何度かテストを実行する必要があるでしょう）。そして、多くの場合、このようなプログラムを第一原理 (First Principles)に従って書くことができ、*実際のお客様を見る前に*構築することができるでしょう。人間には全く新しい状況においても、製品やシステムを動かす第一原理から、自動的に動くシステムを設計する能力があり、これは素晴らしい認知能力です。そして、$100\%$動くようなソリューションを開発できるのであれば、*機械学習を使わないほうがよいでしょう*。

ますます成長している機械学習サイエンティストのコミュニティにとっては幸運なことですが、自動化における多くの問題は、そうした人間の創意工夫だけではまだ簡単に解決されていません。ホワイトボードにあなたが思いつく賢いプログラムを書き出してみましょう。例えば、以下のようなプログラムを書くことを考えます。

* 地理情報、衛星画像、一定間隔の過去の天気データから、明日の天気を予測するプログラムを書く

* フリーフォーマットのテキストで書かれた質問を理解して、適切に回答するプログラムを書く

* 画像に含まれる全ての人間を認識して、それらを枠で囲むプログラムを書く

* ユーザが楽しむような製品で、通常の閲覧の過程ではまだ見ていない製品を提示するプログラムを書く

これらの場合では、優秀なプログラマーであってもゼロからプログラムを書くことはできないでしょう。その理由としては様々なものが考えられます。求められるプログラムが時間によって変化するようなパターンをもつこともあり、それに適応するプログラムも必要になります。また、関係性 (例えば、ピクセル値と抽象的なカテゴリとの関係性) が複雑すぎて、私達の意識的な理解を超える数千、数百万の計算を必要とすることもあります (私達の目はこのようなタスクを難なくこなしますが)。機械学習 (Machine Learning, ML) は*経験*から*挙動を学習*できる協力な技術の学問です。MLのアルゴリズムは、観測データや環境とのインタラクションから経験を蓄積することで、性能が改善します。さきほどの決定的な電子商取引のプラットフォームは、どれだけ経験が蓄積されても、開発者自身が*学習して*、ソフトウェアを改修することを決めないかぎり、同じビジネスロジックに従って実行するため、これとは対照的です。この書籍では、機械学習の基礎について説明し、特に深層学習に焦点を当てます。深層学習は、コンピュータビジョン、自然言語処理、ヘルスケア、ゲノミクスのような幅広い領域において、イノベーションを実現している強力な技術です。

## 身近な例

この書籍の著者は、これを書き始めるようとするときに、多くの仕事を始める場合と同様に、カフェインを摂取していました。そして車に飛び乗って運転を始めたのです。iPhoneをもって、Alexは"Hey Siri"と言って、スマートフォンの音声認識システムを起動します。
Muは「ブルーボトルのコーヒーショップへの行き方」と言いました。するとスマートフォンは、すぐに彼の指示内容を文章で表示しました。そして、行き方に関する要求を認識して、意図に合うようにマップのアプリを起動したのです。起動したマップはたくさんのルートを提示しました。この本で伝えたいことのために、この物語をでっちあげてみましたが、たった数秒の間における、スマートフォンとの日々のやりとりに、様々な機械学習モデルが利用されているのです。

"Alexa"、"Okay Google"、"Siri"といった起動のための言葉に反応するコードを書くことを考えてみます。コンピュータとコードを書くエディタだけをつかって部屋でコードを書いてみましょう。どうやって第一原理に従ってコードを書きますか？考えてみましょう。問題は難しいです。だいたい44,000/秒でマイクからサンプルを集めます。音声の生データに対して、起動の言葉が含まれているかを調べて、YesかNoを正確に対応付けるルールとは何でしょうか？行き詰まってしまうと思いますが心配ありません。このようなプログラムを、ゼロから書く方法はだれもわかりません。これこそが機械学習を使う理由なのです。


ひとつの考え方を紹介したいと思います。私達が、入力と出力を対応付ける方法をコンピュータに陽に伝えることができなくても、私達自身はそのような認識を行う素晴らしい能力を持っています。言い換えれば、たとえ"Alexa"といった言葉を認識するようにコンピュータのプログラムを書けなくても、あなた自身は"Alexa"の言葉を認識できます。従って、私達人間は、音声のデータと起動の言葉を含んでいるか否かのラベルをサンプルとして含む巨大なデータセットをつくることができます。
機械学習のアプローチでは、起動の言葉を認識するようなシステムを陽に実装しません。
代わりに、膨大なパラメータによって挙動を決められるような、柔軟なプログラムを実装します。そして、興味あるタスクの性能を測る指標に関して、プログラムの性能を改善するような、もっともらしいパラメータを決定するため、データセットを利用します。

パラメータは、プログラムの挙動を決めるために、私達が回すことのできるノブのようなものと考えられるでしょう。パラメータを確定すると、そのプログラムは*モデル*と呼ばれます。異なるプログラム (入力と出力のマッピング) をパラメータを変更するだけで生成可能な場合、それらをモデルの*ファミリ (familiy)* と呼ばれます。パラメータを選ぶために、データセットを利用する*メタなプログラム*を*学習アルゴリズム*と呼びます。

機械学習アルゴリズムに進んで取り組んでいく前に、問題を正確に定義する必要があり、つまり、入力と出力の性質を正確にはっきりさせ、モデルのファミリを適切に選ぶことが必要です。この場合、モデルは*入力*として音声の一部を受け取って、*出力*として``{はい、いいえ}``からの選択を生成します。そして、このあとの書籍の内容通りに進むのであれば、音声の一部が起動語を含んでいそうか（そうでないか）を近似的に決定すること、といえます。

もし私達が、正しいモデルを選んだのであれば、そのモデルは'Alexa'という言葉を聞いたときにyesを起動するような、1つの設定を切り替えるノブが存在するでしょう。起動する言葉は任意なので、'Apricot'ということばで起動するような別のノブもありえます。入出力が変われば根本的に別のモデルを必要とする場合もあります。例えば、画像とラベルを対応付けるタスクと、英語と中国語を対応付けるタスクには、異なるモデルを利用する必要があるでしょう。

想像できるとは思いますが、"Alexa"と"Yes"のようなペアがランダムなものであれば、モデルは"Alexa"も"Apricot"も他の英語も何も認識できないでしょう。Deep Learningという言葉の中のLearningというのは、学習の期間において、複数のペアを上手く使って、モデルの挙動を更新していくことを指します。その学習のプロセスというのは以下のようなものです。

1. まず最初にランダムに初期化されたモデルから始めます。このモデルは最初は使い物になりません
1. ラベルデータを取り込みます（例えば、部分的な音声データと対応するYes/Noのラベルです）
1. そのペアを利用してモデルを改善します
1. モデルが良いものになるまで繰り返します

![](../img/ml-loop.svg)

まとめると、起動の言葉を認識できるようなコードを直接書くよりも、*もしたくさんのラベル付きのデータで認識機能を表現できるなら*、その認識機能を*学習*するようなコードを書くべきです。これは、プログラムの挙動をデータセットで決める、つまり*データでプログラムを書く*ようなものだと考えることができます。

私達は、以下に示すようなネコとイヌの画像サンプルを大量に集めて、機械学習を利用することで、ネコ認識器をプログラムすることができます。


|![](../img/cat1.png)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
|:---------------:|:---------------:|:---------------:|:---------------:|
|ネコ|ネコ|イヌ|イヌ|

このケースでは、認識器はネコであれば非常に大きな正の値、イヌであれば非常に大きな負の値、どちらかわからない場合はゼロを出力するように学習するでしょう。そしてこれは、MLが行っていることを、ゼロからプログラミングしているわけではありません。

深層学習は、機械学習の問題を解く最も有名なフレームワークのたった一つです。ここまで、機械学習について広く話をしてきましたが、深層学習については話をしていません。ここで、少しだけ紹介しておきたい点があります。まず、これまで議論した問題として、音声情報の生データ、画像のピクセル値、任意の長さの文章や言語間の関係からの学習を議論しましたが、これらの問題では深層学習が優れており、古くからの機械学習ツールは影響力がなくなりつつあります。深層学習モデルは、多くの計算レイヤーを学習することで高い精度を実現しています。これらの多くのレイヤーからなるモデル (もしくは階層的なモデル) は、これまでのツールではできない方法で、低レベルな認識に関わるデータを取り扱うことが可能です。

過去には、これらの問題にMLを適用する際の重要な部分として、データを浅いモデルでも扱える形式に変換する方法を人手で構築することがありました。深層学習の1つの重要な利点として、古くからの機械学習パイプラインの最後に利用されていた浅いモデルを置き換えただけでなく、負荷の大きい特徴作成（feature engineering) も置き換えた点があります。また、ドメインに特化した前処理の多くを置き換え、コンピュータビジョン、音声認識、自然言語処理、医療情報学、それ以外にも様々な分野を分割していた境界を多く取り除き、多様な問題を扱える統一的なツールとなりつつあります。

## 機械学習の核となる要素: データ・モデル・アルゴリズム

In our wake-word example, we described a dataset consisting of audio snippets and binary labels gave a hand-wavy sense of how we might train a model to approximate a mapping from snippets to classifications. This sort of problem, where we try to predict a designated unknown label given known inputs (also called features or covariates), and examples of both is called supervised learning, and it's just one among many kinds of machine learning problems. In the next section, we'll take a deep dive into the different ML problems. First, we'd like to shed more light on some core components that will follow us around, no matter what kind of ML problem we take on:

<!-- 起動のための言葉を認識するタスクを考えたとき、音声データとラベルからなるデータセットを準備します。そこで、音声からラベルを推定する機械学習モデルをどうやって学習させるかを記述するかもしれません。サンプルからラベルを推定するこのセットアップは機械学習の一種で、*教師あり学習*と呼ばれるものです。Deep Learningにおいても多くのアプローチがありますが、それについては以降の章で述べたいと思います。機械学習を進めるために、以下の４つのことが必要になります。 -->

1. 学習に利用する**データ**
1. データを変換して推定するための**モデル**
1. そのモデルの悪さを定量化する**ロス関数**
1. ロス関数を最小化するような、モデルのパラメータを探す**アルゴリズム**


### データ

It might go without saying that you cannot do data science without data.
We could lose hundreds of pages pondering the precise nature of data
but for now we'll err on the practical side and focus on the key properties
to be concerned with.
Generally we are concerned with a collection of *examples*
(also called *data points*, *samples*, or *instances*).
In order to work with data usefully, we typically
need to come up with a suitable numerical representation.
Each *example* typically consists of a collection
of numerical attributes called *features* or *covariates*.

If we were working with image data,
each individual photograph might constitute an *example*,
each represented by an ordered list of numerical values
corresponding to the brightness of each pixel.
A $200\times200$ color photograph would consist of $200\times200\times3=120000$
numerical values, corresponding to the brightness
of the red, green, and blue channels corresponding to each spatial location.
In a more traditional task, we might try to predict
whether or not a patient will survive,
given a standard set of features such as age, vital signs, diagnoses, etc.

When every example is characterized by the same number of numerical values,
we say that the data consists of *fixed-length* vectors
and we describe the (constant) length of the vectors
as the *dimensionality* of the data.
As you might imagine, fixed length can be a convenient property.
If we wanted to train a model to recognize cancer in microscopy images,
fixed-length inputs means we have one less thing to worry about.

However, not all data can easily be represented as fixed length vectors.
While we might expect microscrope images to come from standard equipment,
we can't expect images mined from the internet to all show up in the same size.
While we might imagine cropping images to a standard size,
text data resists fixed-length representations even more stubbornly.
Consider the product reviews left on e-commerce sites like Amazon or TripAdvisor. Some are short: "it stinks!". Others ramble for pages.
One major advantage of deep learning over traditional methods
is the comparative grace with which modern models
can handle *varying-length* data.


一般的に、多くのデータを持っていれば持っているほど、問題を簡単に解くことができます。多くのデータを持っているなら、より性能の良いモデルを構築することができるからです。比較的小規模なデータからビッグデータと呼ばれる時代への移り変わりによって、現代の深層学習は成り立っているといえます。話をもとに戻しますが、深層学習における最も素晴らしいモデルの多くは大規模なデータセットなしには機能しません。
いくつかは小規模なデータの時代においても有効でしたが、従来からのアプローチと大差はありません。


Finally it's not enough to have lots of data and to process it cleverly.
We need the *right* data.
If the data is full of mistakes, or if the chosen features are not predictive of the target quantity of interest, learning is going to fail.
The situation is well captured by the cliché: *garbage in, garbage out*.
Moreover, poor predictive performance isn't the only potential consequence.
In sensitive applications of machine learning,
like predictive policing, resumé screening, and risk models used for lending,
we must be especially alert to the consequences of garbage data.
One common failure mode occurs in datasets where some groups of people
are unrepresented in the training data.
Imagine applying a skin cancer recognition system in the wild
that had never seen black skin before.
Failure can also occur when the data doesn't merely under-represent some groups,
but reflects societal prejudices.
For example if past hiring decisions are used to train a predictive model
that will be used to screen resumes, then machine learning models could inadvertently capture and automate historical injustices.
Note that this can all happen without the data scientist being complicit,
or even aware.

<!--
* **画像** スマートフォンで撮影されたり、Webで収集された画像、衛星画像、超音波やCTやMRIなどのレントゲン画像など

* **テキスト** Eメール、学校でのエッセイ、tweet、ニュース記事、医者のメモ、書籍、翻訳文のコーパスなど

* **音声** Amazon Echo、iPhone、Androidのようなスマートデバイスに送られる音声コマンド、音声つき書籍、通話、音楽など

* **動画** テレビや映画、Youtubeのビデオ、携帯電話の撮影、自宅の監視カメラ映像、複数カメラによる追跡、など

* **構造化データ** ウェブページ、電子カルテ、レンタカーの記録、デジタルな請求書など -->


### モデル

Most machine learning involves *transforming* the data in some sense.
We might want to build a system that ingests photos and predicts *smiley-ness*.
Alternatively, we might want to ingest a set of sensor readings
and predict how *normal* vs *anomalous* the readings are.
By *model*, we denote the computational machinery for ingesting data
of one type, and spitting out predictions of a possibly different type.
In particular, we are interested in statistical models
that can be estimated from data.
While simple models are perfectly capable of addressing
appropriately simple problems the problems
that we focus on in this book stretch the limits of classical methods.
Deep learning is differentiated from classical approaches
principally by the set of powerful models that it focuses on.
These models consist of many successive transformations of the data
that are chained together top to bottom, thus the name *deep learning*.
On our way to discussing deep neural networks, we'll discuss some more traditional methods.

###  目的関数

モデルが良いかどうかを評価するためには、モデルの出力と実際の正解を比較する必要があります。ロス関数は、その出力が*悪い*ことを評価する方法です。例えば、画像から患者の心拍数を予測するモデルを学習する場合を考えます。そのモデルが心拍数は100bpmだと推定して、実際は60bpmが正解だったときには、そのモデルに対して推定結果が悪いことを伝えなくてはなりません。

同様に、Eメールがスパムである確率を予測するモデルを作りたいとき、その予測が上手く行っていなかったら、そのモデルに伝える方法が必要になります。一般的に、機械学習の*学習*と呼ばれる部分はロス関数を最小化することです。通常、モデルはたくさんのパラメータをもっています。
パラメータの最適な値というのは、私達が学習で必要とするものであり、観測したデータのなかの*学習データ*の上でロスを最小化することによって得られます。残念ながら、学習データ上でいくら上手くロス関数を最小化しても、いまだ見たことがないテストデータにおいて、学習したモデルがうまくいくという保証はありません。従って、利用できるデータを、学習データ (モデルパラメータ決定用) とテストデータ (評価用)に分けることが一般的で、それらから以下の2種類の値を求めることができます。

* **学習誤差**: 学習済みモデルにおける学習データの誤差です。これは、実際の試験に備えるための練習問題に対する学生の得点のようなものです。その結果は、実際の試験の結果が良いことを期待させますが、最終試験での成功を保証するものではありません。
* **テスト誤差**: たことのないテストデータに対する誤差で、学習誤差とは大きく異なる場合があります。見たことのないデータに対して、モデルが対応（汎化）できないとき、その状態を *overfitting* (過適合)と呼びます。実生活でも、練習問題に特化して準備したにもかかわらず、本番の試験で失敗するというのと似ています。

### 最適化アルゴリズム

元データとその数値的な表現、モデル、上手く定義された目的関数があれば、ロスを最小化するような最適なパラメータを探索するアルゴリズムが必要になります。
ニューラルネットワークにおける最も有名な最適化アルゴリズムは、最急降下法と呼ばれる方法にもとづいています。端的に言えば、パラメーラを少しだけ動かしたとき、学習データに対するロスがどのような方向に変化するかを、パラメータごとに見ることです。ロスが小さくなる方向へパラメータを更新します。

## さまざまな機械学習

In the following sections, we will discuss a few types of machine learning in some more detail. We begin with a list of *objectives*, i.e. a list of things that machine learning can do. Note that the objectives are complemented with a set of techniques of *how* to accomplish them, i.e. training, types of data, etc. The list below is really only sufficient to whet the readers' appetite and to give us a common language when we talk about problems. We will introduce a larger number of such problems as we go along.


### 教師あり学習

教師あり学習は、入力データが与えられて、何らかの*対象*を予測するものです。その対象というのは、*ラベル*と呼ばれていて、*y*で表現することが多いです。入力データ点は、*example* や *instance* と呼ばれることがあり、$\boldsymbol{x}$で表現されることが多いです。
教師あり学習の目的は、入力$\boldsymbol{x}$から予測$f_{\theta}(\boldsymbol{x})$を得るようなモデル$f_\theta$を生成することです。

この説明を例を使って具体的に説明したいと思います。ヘルスケアの分野で仕事をしているとき、患者が心臓発作を起こすかどうかを予測したいと思います。実際に観測した内容、この場合、*心臓発作*か*心臓発作でない*かがラベル$y$です。入力データ$\boldsymbol{x}$は、心拍数や、最高、最低の血圧といった、いわゆるバイタルサインというものになるでしょう。

教師あり学習における、教師するというのは、パラメータ$\theta$を選ぶためのもので、私達が教師となって、ラベルの付いた入力データ($\boldsymbol{x}_i, y_i$)とモデルを与えます。
各データ$\boldsymbol{x}_i$は正しいラベルと対応しています。




確率の専門用語を用いれば、私達は条件付き確率$P(y|x)$を求めようとしています。機械学習には、いくつかのアプローチがありますが、教師あり学習は実際に利用されている機械学習の大半を占めると言って良いでしょう。重要なタスクは、与えられた事実から未知の事象の確率を推測すること、と大雑把に言うことができるでしょう。例えば、

* **CT画像からガンかガンでないかを予測する**
* **英語の文章から正しいフランス語の翻訳を予測する**
* **今月の財務データから翌月の株価を予測する**

「入力から対象を予測する」というシンプルな説明をしましたが、教師あり学習は、入出力データのタイプ、サイズ、数に応じて、様々な形式やモデル化の方法をとることができます。例えば、文字列や時系列データなどの系列データを処理する場合と、固定長のベクトルで表現されるデータを処理する場合で、異なるモデルを利用することができます。このコンテンツの序盤では、これらの問題の多くについて深く説明したいと思います。


簡潔に言うと、機械学習のプロセスというのは、こういったものです。たくさんの入力データ点をランダムに選んで手に入れます。それぞれのデータ点に対して正しいラベルを付与します。入力データ点と対応するラベル（欲しい出力）で学習データセットを作ります。その学習データセットを教師あり学習アルゴリズムに入力します。*教師あり学習アルゴリズム*は一種の関数で、データセットを入力として受け取り、*学習モデル*とよばれる関数を出力します。その学習モデルを利用して、未知の入力データから対応するラベルを予測することができます。

![](../img/supervised-learning.svg)



#### 回帰

最も単純な教師あり学習のタスクとして頭に思い浮かぶものは、おそらく回帰ではないかと思います。例えば、住宅の売上に関するデータベースから、一部のデータセットが得られた場合を考えてみます。各列が異なる住居に、各列は関連する属性、例えば、住宅の面積、寝室の数、トイレの数、中心街まで徒歩でかかる時間に対応するような表を構成するでしょう。形式的に、このようなデータセットの1行を*特徴ベクトル*、関連する対象（今回の事例では1つの家）を*データ例*と呼びます。

もしニューヨークやサンフランシスコに住んでいて、Amazon、Google、Microsoft、FacebookなどのCEOでなければ、その特徴ベクトル（面積、寝室数、トイレ数、中心街までの距離）は$[100, 0, .5, 60]$といった感じでしょう。一方、もしピッツバーグに住んでいれば、その特徴ベクトルは$[3000, 4, 3, 10]$のようになると思います。 このような*特徴ベクトル*は、伝統的な機械学習のあらゆる問題において必要不可欠なものでした。あるデータ例に対する特徴ベクトルを$\mathbf{x_i}$で、全てのテータ例の特徴ベクトルを$X$として表します。

何が問題を*回帰*させるかというと、実はその出力なのです。もしあなたが、新居を購入しようと思っていて、上記のような特徴量を用いて、家の適正な市場価値を推定したいとしましょう。目標値は、販売価格で、これは*実数*です。あるデータ例$\mathbf{x_i}$に対する個別の目標値を$y_i$とし、すべてのデータ例$\mathbf{X}$に対応する全目標を$\mathbf{y}$とします。目標値が、ある範囲内の任意の実数をとるとき、この問題を回帰問題と呼びます。ここで作成するモデルの目的は、実際の目標値に近い予測値(いまの例では、価格の推測値)を生成することです。この予測値を$\hat{y}_i$とします。もしこの表記になじみがなければ、いまのところ無視しても良いです。以降の章では、中身をより徹底的に解説していく予定です。

多くの実践的な問題は、きちんと説明されて、わかりやすい回帰問題となるでしょう。ユーザがある動画につけるレーティングを予測する問題は回帰問題です。もし2009年にその功績をあげるような偉大なアルゴリズムを設計できていれば、[Netflixの100万ドルの賞](https://en.wikipedia.org/wiki/Netflix_Prize)を勝ち取っていたかも知れません。病院での患者の入院日数を予測する問題もまた回帰問題です。ある1つの法則として、*どれくらい?*という問題は回帰問題を示唆している、と判断することは良いかも知れません。

* 'この手術は何時間かかりますか?' - *回帰*
* 'この写真にイヌは何匹いますか?' - *回帰*.

しかし、「これは__ですか?」のような問題として簡単に扱える問題であれば、それはおそらく分類問題で、次に説明する全く異なる種類の問題です。たとえ、機械学習をこれまで扱ったことがなかったとしても、形式に沿わない方法で、おそらく回帰問題を扱うことができるでしょう。例えば、下水装置を直すことを考えましょう。工事業者は下水のパイプから汚れを除くのに$x_1=3$時間かかって、あなたに$y_1=350$の請求をしました。友人が同じ工事業者を雇い、$x_2 = 2$時間かかったとき、友人に$y_2=250$の請求をしました。もし、これから汚れを除去する際に、どれくらいの請求が発生するかを尋ねられたら、作業時間が長いほど料金が上がるといった、妥当な想定をすると思います。そして、基本料金があって、１時間当たりの料金もかかるという想定もするでしょう。これらの想定のもと、与えられた2つのデータを利用して、工事業者の値付け方法を特定できるでしょう。1時間当たり\$100で、これに加えて\$50の出張料金です。もし、読者がこの内容についてこれたのであれば、線形回帰のハイレベルな考え方をすでに理解できているでしょう (そして、バイアス項のついた線形モデルを暗に設計したことになります)。


この場合では、工事業者の価格に完全に一致するようなパラメータを作ることができましたが、場合によってはそれが不可能なことがあります。例えば、2つの特徴に加えて、いくらかの分散がある要因によって発生する場合です。このような場合は、予測値と観測値の差を最小化するようにモデルを学習します。本書の章の多くでは、以下の2種類の一般的なロスのうち1つに着目します。
以下の式で定義される[L1ロス](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)と

$$l(y,y') = \sum_i |y_i-y_i'|$$

以下の式で定義される最小二乗ロス、別名[L2ロス](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)です。


$$l(y,y') = \sum_i (y_i - y_i')^2.$$

のちほど紹介しますが、$L_2$ロスはガウスノイズによってばらついているデータを想定しており、$L_1$ロスはラプラス分布のノイズによってばらついていることを想定しています。

#### 分類

回帰モデルは「*どれくらい多い?*」という質問に回答する上で、優れたものではありますが、多くの問題はこのテンプレートに都合よく当てはめられるとは限りません。例えば、ある銀行ではモバイルアプリに、領収書をスキャンする機能を追加したいと思っています。これには、スマートフォンのカメラで領収書の写真を撮る顧客と、画像内のテキストを自動で理解できる必要のある機械学習モデルが関係するでしょう。より頑健に手書きのテキストを理解する必要もあるでしょう。この種のシステムは光学文字認識 (Optical Character Recognition, OCR) と呼ばれており、OCRが解くこの種の問題は
分類と呼ばれています。分類の問題は、回帰に利用されたアルゴリズムとは全く異なるアルゴリズムが利用されます。

分類において、ある画像データのピクセル値のような特徴ベクトルにもとづいて、いくつかの候補の中から、そのデータがどのカテゴリ（正式には*クラス*)を予測したいと思うでしょう。例えば、手書きの数字に対しては、0から9までの数値に対応する10クラスになるでしょう。最も単純な分類というのは、2クラスだけを対象にするとき、すなわち2値分類と呼ばれる問題です。例えば、データセット$X$が動物の画像で構成されていて、*ラベル*$Y$が$\mathrm{\{ネコ, イヌ\}}$のクラスであるとします。回帰の場合、実数$\hat{y}$を出力するような回帰式を求めていたでしょう。分類では、出力$\hat{y}$は予測されるクラスとなるような分類器を求めることになるでしょう。

この本が目的としているように、より技術的な内容に踏み込んでいきましょう。*ネコ*や*イヌ*といったカテゴリに対して、0か1かで判定するようなモデルを最適化することは非常に難しいです。その代わりに、確率を利用してモデルを表現するほうがはるかに簡単です。あるデータ$x$に対して、モデルは、ラベル$k$となる確率$\hat{y}_k$を決定します。これらは確率であるため、正の値であり、その総和は$1$になります。したがって、$K$個のカテゴリの確率を決めるためには、$K-1$の数だけ必要であることがわかります。2クラス分類の場合がわかりやすいです。表を向く確率が0.6 (60%) の不正なコインでは、0.4 (40%)の確率で裏を向くでしょう。動物を認識する例に戻ると、分類器は画像を見てネコである確率 $\Pr(y=\mathrm{cat}| x) = 0.9$を出力します。その数値は、画像がネコを表していることを、90%の自信をもって分類器が判定したものだと解釈できるでしょう。ある予測されたクラスに対する確率の大きさは信頼度の一つです。そしてこれは、唯一の信頼度というわけではなく、より発展的な章において、異なる不確実性について議論する予定です。

二つより多いクラスが考えられるときは、その問題を*多クラス分類*と呼びます。一般的な例として、 `[0, 1, 2, 3 ... 9, a, b, c, ...]`のような手書き文字の認識があります。L1やL2のロス関数を最小化することで回帰問題を解こうとしてきましたが、分類問題に対する一般的なロス関数はcross-entropyと呼ばれるものです。MXNet Gluonにおいては、対応する関数が[ここ](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss)に記載されています

分類器が最も可能性が高いと判断したクラスが、行動を決定づけるものになるとは限りません。例えば、裏庭で美しいキノコを発見したとします。


|![](../img/death_cap.jpg)|
|:-------:|
|タマゴテングタケという毒キノコ - 決して食べてはいけません!|

ここで分類器を構築して、画像のキノコが毒キノコかどうかを判定するように分類器を学習させます。そして、その毒キノコ分類器が毒キノコ(death cap)である確率$\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2$を出力したとします。言い換えれば、そのキノコは80%の信頼度で毒キノコではないと判定されているのです。しかし、それを食べるような人はいないでしょう。このキノコでおいしい夕食をとる利益が、これを食べて20%の確率で死ぬリスクに見合わないからです。言い換えれば、*不確実なリスク*が利益よりもはるかに重大だからです。数学でこのことを見てみましょう。基本的には、私たちが引き起こすリスクの期待値を計算する必要があります。つまり、それに関連する利益や損失と、起こりうる確率を掛け合わせます。

$$L(\mathrm{action}| x) = \mathbf{E}_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)]$$

そして、キノコを食べることによるロス$L$は$L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$で、これに対して、キノコを食べない場合のロス$L$は$L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$です。

私たちの注意は正しかったのです。キノコの学者はおそらく、上のキノコは毒キノコであるというでしょう。分類は、そうした2クラス分類、多クラス分類または、マルチラベル分類よりもずっと複雑になることもあります。例えば、階層構造のような分類を扱うものもあります。階層では、多くのクラスの間に関係性があることを想定しています。そして、すべての誤差は同じではありません。つまり、関係性のない離れたクラスを誤分類することは良くないですが、関係性のあるクラスを誤分類することはまだ許容されます。通常、このような分類は*階層分類*と呼ばれます。初期の例は[リンネ](https://en.wikipedia.org/wiki/Carl_Linnaeus)によるもので、彼は動物を階層的に分類・組織しました。

![](../img/sharks.png)

動物の階層的分類の場合、プードルをシュナウザーに間違えることはそこまで悪いことでないでしょうが、プードルを恐竜と間違えるようであれば、その分類モデルはペナルティを受けるでしょう。階層における関連性は、そのモデルをどのように使うかに依存します。例えば、ガラガラヘビとガータースネークの2種類のヘビは系統図のなかでは近いかもしれませんが、猛毒をもつガラガラヘビを、そうでないガータースネークと間違えることは命取りになるかもしれません。


#### タグ付け

いくつかの分類問題は、2クラス分類や多クラスのような設定にうまく合わないことがあります。例えば、イヌをネコと区別するための標準的な2クラス分類器を学習するとしましょう。コンピュータービジョンの最新の技術をもって、既存のツールで、これを実現することができます。それにも関わらず、モデルがどれだけ精度が良くなったとしても、ブレーメンの音楽隊のような画像が分類器に入力されると、問題が起こることに気づきませんか。

![](../img/stackedanimals.jpg)

画像から確認できる通り、画像にはネコ、オンドリ、イヌ、ロバが、背景の木とともに写っています。最終的には、モデルを使ってやりたいことに依存はするのですが、この画像を扱うのに2クラス分類を利用することは意味がないでしょう。代わりに、その画像において、ネコとイヌとオンドリとロバのすべて写っていることを、モデルに判定させたいと考えるでしょう。

どれか1つを選ぶのではなく、*相互に排他的でない*クラスを予測するように学習する問題は、マルチラベル分類と呼ばれています。ある技術的なブログにタグを付与する場合を考えましょう。例えば、'機械学習'、'技術'、'ガジェット'、'プログラミング言語'、'linux'、'クラウドコンピューティング'、'AWS'です。典型的な記事には5から10のタグが付与されているでしょう。なぜなら、これらの概念は互いに関係しあっているからです。'クラウドコンピューティング'に関する記事は、'AWS'について言及する可能性が高く、'機械学習'に関する記事は'プログラミング言語'も扱う可能性があるでしょう。

また、生体医学の文献を扱うときにこの種の問題を処理する必要があります。論文に正確なタグ付けをすることは重要で、これによって、研究者は文献を徹底的にレビューすることができます。アメリカ国立医学図書館では、たくさんの専門的なタグ付けを行うアノテータが、PubMedにインデックスされる文献一つ一つを見て、2万8千のタグの集合であるMeSHから適切な用語を関連付けます。これは時間のかかる作業で、文献が保管されてからタグ付けが終わるまで、アノテータは通常1年の時間をかけます。個々の文献が人手による正式なレビューを受けるまで、機械学習は暫定的なタグを付与することができるでしょう。実際のところ、この数年間、BioASQという組織がこの作業を正確に実行するための[コンペティションを行っていました](http://bioasq.org/)。

#### 検索とランキング

上記の回帰や分類のように、データをある実数値やカテゴリに割り当てない場合もあるでしょう。情報検索の分野では、ある商品の集合に対するランキングを作成することになります。Web検索を例にとると、その目的はクエリに関係する特定のページを決定するだけでは十分ではなく、むしろ、大量の検索結果からユーザに提示すべきページを決定することにあります。私達はその検索結果の順番を非常に気にします。学習アルゴリズムは、大きな集合から取り出した一部の集合について、要素に順序をつける必要があります。言い換えれば、アルファベットの最初の5文字を対象としたときに、``A B C D E`` と ``C A B E D``には違いがあるということです。たとえ、その集合が同じであっても、その集合の中の順序は重要です。

この問題に対する1つの解決策としては、候補となる集合のすべての要素に関連スコアをつけて、そのスコアが高い要素を検索することでしょう。 [PageRank](https://en.wikipedia.org/wiki/PageRank)はその関連スコアの初期の例です。変わった点の1つとしては、それが実際のクエリに依存しないということです。代わりに、そのクエリの単語を含む結果に対して、単純に順序をつけています。現在の機械学習エンジンは、クエリに依存した関連スコアを得るために、機械学習と行動モデルを利用しています。このトピックのみを扱うようなカンファレンスも存在します。

<!-- Add / clean up-->

#### 推薦システム

推薦システムは、検索とランキングに関係するもう一つの問題です。その問題は、ユーザに関連する商品群を提示するという目的においては似ています。主な違いは、推薦システムにおいて特定のユーザの好みに合わせる(*Personalization*)に重きをおいているところです。例えば、映画の推薦の場合、SFのファン向けの推薦結果は、Woody Allenのコメディに詳しい人向けの推薦結果とは大きく異なるでしょう。

そのような問題は、映画、製品、音楽の推薦において見られます。ときどき、顧客はその商品がその程度好きなのかということを陽に与える場合があります(例えば、Amazonの商品レビュー)。また、プレイリストでタイトルをスキップするように、結果に満足しない場合に、そのフィードバックを送ることもあります。一般的に、こうしたシステムでは、あるユーザ$u_i$と商品$p_i$があたえられたとき、商品に付与する評価や購入の確率などを、スコア$y_i$として見積もることに力を入れています。

そのようなモデルが与えられると、あるユーザに対して、スコア$y_{ij}$が最も大きい商品群を検索することできるでしょう。そして、それが推薦として利用されるのです。実際に利用されているシステムはもっと先進的で、スコアを計算する際に、ユーザの行動や商品の特徴を詳細に考慮しています。次の画像は、著者の好みに合わせて調整したpersonalizationのアルゴリズムを利用して、Amazonで推薦される深層学習の書籍の例を表しています。

![](../img/deeplearning_amazon.png)

#### Sequence Learning

So far we've looked at problems where we have some fixed number of inputs
and produce a fixed number of outputs.
Before we considered predicting home prices from a fixed set of features:
square footage, number of bedrooms, number of bathrooms, walking time to downtown.
We also discussed mapping from an image (of fixed dimension),
to the predicted probabilities that it belongs to each of a fixed number of classes,
or taking a user ID and a product ID, and predicting a star rating.
In these cases, once we feed our fixed-length input into the model to generate an output,
the model immediately forgets what it just saw.

This might be fine if our inputs truly all have the same dimensions
and if successive inputs truly have nothing to do with each other.
But how would we deal with video snippets?
In this case, each snippet might consist of a different number of frames.
And our guess of what's going on in each frame
might be much stronger if we take into account
the previous or succeeding frames.
Same goes for language.
One popular deep learning problem is machine translation:
the task of ingesting sentences in some source language
and predicting their translation in another language.

These problems also occur in medicine.
We might want a model to monitor patients in the intensive care unit and to fire off alerts
if their risk of death in the next 24 hours exceeds some threshold.
We definitely wouldn't want this model to throw away everything it knows about the patient history each hour,
and just make its predictions based on the most recent measurements.

These problems are among the more exciting applications of machine learning
and they are instances of *sequence learning*.
They require a model to either ingest sequences of inputs
or to emit sequences of outputs (or both!).
These latter problems are sometimes referred to as ``seq2seq`` problems.
Language translation is a ``seq2seq`` problem.
Transcribing text from spoken speech is also a ``seq2seq`` problem.
While it is impossible to consider all types of sequence transformations,
a number of special cases are worth mentioning:

##### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities.

|Tom | has | dinner | in | Washington | with | Sally.|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Ent | - | - | - | Ent | - | Ent|


##### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker,
and the output $y$ is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are ``seq2seq`` problems where the output is much shorter than the input.

|`-D-e-e-p- L-ea-r-ni-ng-`|
|:--------------:|
|![Deep Learning](../img/speech.png)|

##### Text to Speech

Text-to-Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this isn't quite so trivial for computers.

##### Machine Translation

Unlike the case of speech recognition, where corresponding inputs and outputs occur in the same order (after alignment),
in machine translation, order inversion can be vital.
In other words, while we are still converting one sequence into another,
neither the number of inputs and outputs
nor the order of corresponding data points
are assumed to be the same.
Consider the following illustrative example of the obnoxious tendency of Germans
(*Alex writing here*)
to place the verbs at the end of sentences.

|German |Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?|
|:------|:---------|
|English|Did you already check out this excellent tutorial?|
|Wrong alignment |Did you yourself already this excellent tutorial looked-at?|

A number of related problems exist.
For instance, determining the order in which a user reads a webpage
is a two-dimensional layout analysis problem.
Likewise, for dialogue problems,
we need to take world-knowledge and prior state into account.
This is an active area of research.


### Unsupervised learning

All the examples so far were related to *Supervised Learning*,
i.e. situations where we feed the model
a bunch of examples and a bunch of *corresponding target values*.
You could think of supervised learning as having an extremely specialized job and an extremely anal boss.
The boss stands over your shoulder and tells you exactly what to do in every situation until you learn to map from situations to actions.
Working for such a boss sounds pretty lame.
On the other hand, it's easy to please this boss. You just recognize the pattern as quickly as possible and imitate their actions.

In a completely opposite way,
it could be frustrating to work for a boss
who has no idea what they want you to do.
However, if you plan to be a data scientist, you'd better get used to it.
The boss might just hand you a giant dump of data and tell you to *do some data science with it!*
This sounds vague because it is.
We call this class of problems *unsupervised learning*,
and the type and number of questions we could ask
is limited only by our creativity.
We will address a number of unsupervised learning techniques in later chapters. To whet your appetite for now, we describe a few of the questions you might ask:

* Can we find a small number of prototypes that accurately summarize the data? Given a set of photos, can we group them into landscape photos, pictures of dogs, babies, cats, mountain peaks, etc.? Likewise, given a collection of users' browsing activity, can we group them into users with similar behavior? This problem is typically known as **clustering**.
* Can we find a small number of parameters that accurately capture the relevant properties of the data? The trajectories of a ball are quite well described by velocity, diameter, and mass of the ball. Tailors have developed a small number of parameters that describe human body shape fairly accurately for the purpose of fitting clothes. These problems are referred to as **subspace estimation** problems. If the dependence is linear, it is called **principal component analysis**.
* Is there a representation of (arbitrarily structured) objects in Euclidean space (i.e. the space of vectors in $\mathbb{R}^n$) such that symbolic properties can be well matched? This is called **representation learning** and it is used to describe entities and their relations, such as Rome - Italy + France = Paris.
* Is there a description of the root causes of much of the data that we observe? For instance, if we have demographic data about house prices, pollution, crime, location, education, salaries, etc., can we discover how they are related simply based on empirical data? The field of **directed graphical models** and **causality** deals with this.
* An important and exciting recent development is **generative adversarial networks**. They are basically a procedural way of synthesizing data. The underlying statistical mechanisms are tests to check whether real and fake data are the same. We will devote a few notebooks to them.


### Interacting with an Environment

So far, we haven't discussed where data actually comes from,
or what actually *happens* when a machine learning model generates an output.
That's because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data up front,
then do our pattern recognition without ever interacting with the environment again.
Because all of the learning takes place after the algorithm is disconnected from the environment,
this is called *offline learning*.
For supervised learning, the process looks like this:

![](../img/data-collection.svg)


This simplicity of offline learning has its charms.
The upside is we can worry about pattern recognition in isolation without these other problems to deal with,
but the downside is that the problem formulation is quite limiting.
If you are more ambitious, or if you grew up reading Asimov's Robot Series,
then you might imagine artificially intelligent bots capable not only of making predictions,
but of taking actions in the world.
We want to think about intelligent *agents*, not just predictive *models*.
That means we need to think about choosing *actions*, not just making *predictions*.
Moreover, unlike predictions, actions actually impact the environment.
If we want to train an intelligent agent,
we must account for the way its actions might
impact the future observations of the agent.


Considering the interaction with an environment opens a whole set of new modeling questions. Does the environment:

* remember what we did previously?
* want to help us, e.g. a user reading text into a speech recognizer?
* want to beat us, i.e. an adversarial setting like spam filtering (against spammers) or playing a game (vs an opponent)?
* not  care (as in most cases)?
* have shifting dynamics (steady vs. shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different).
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs.
We'll briefly describe reinforcement learning and adversarial learning,
two settings that explicitly consider interaction with an environment.


### Reinforcement learning

If you're interested in using machine learning to develop an agent that interacts with an environment and takes actions, then you're probably going to wind up focusing on *reinforcement learning* (RL).
This might include applications to robotics, to dialogue systems,
and even to developing AI for video games.
*Deep reinforcement learning* (DRL), which applies deep neural networks
to RL problems, has surged in popularity.
The breakthrough [deep Q-network that beat humans at Atari games using only the visual input](https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/) ,
and the [AlphaGo program that dethroned the world champion at the board game Go](https://www.wired.com/2017/05/googles-alphago-trounces-humans-also-gives-boost/) are two prominent examples.

Reinforcement learning gives a very general statement of a problem,
in which an agent interacts with an environment over a series of *time steps*.
At each time step $t$, the agent receives some observation $o_t$ from the environment,
and must choose an action $a_t$ which is then transmitted back to the environment.
Finally, the agent receives a reward $r_t$ from the environment.
The agent then receives a subsequent observation, and chooses a subsequent action, and so on.
The behavior of an RL agent is governed by a *policy*.
In short, a *policy* is just a function that maps from observations (of the environment) to actions.
The goal of reinforcement learning is to produce a good policy.

![](../img/rl-environment.svg)

It's hard to overstate the generality of the RL framework.
For example, we can cast any supervised learning problem as an RL problem.
Say we had a classification problem.
We could create an RL agent with one *action* corresponding to each class.
We could then create an environment which gave a reward
that was exactly equal to the loss function from the original supervised problem.

That being said, RL can also address many problems that supervised learning cannot.
For example, in supervised learning we always expect
that the training input comes associated with the correct label.
But in RL, we don't assume that for each observation,
the environment tells us the optimal action.
In general, we just get some reward.
Moreover, the environment may not even tell us which actions led to the reward.

Consider for example the game of chess.
The only real reward signal comes at the end of the game when we either win, which we might assign a reward of 1,
or when we lose, which we could assign a reward of -1.
So reinforcement learners must deal with the *credit assignment problem*.
The same goes for an employee who gets a promotion on October 11.
That promotion likely reflects a large number of well-chosen actions over the previous year.
Getting more promotions in the future requires figuring out what actions along the way led to the promotion.

Reinforcement learners may also have to deal with the problem of partial observability.
That is, the current observation might not tell you everything about your current state.
Say a cleaning robot found itself trapped in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observations before entering the closet.

Finally, at any given point, reinforcement learners might know of one good policy,
but there might be many other better policies that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.


#### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems.

When the environment is fully observed,
we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*.


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

* AppleのSiri、AmazonのAlexa、Google assistantといった知的なアシスタントは、利用可能なレベルの精度で話し言葉の質問に回答することができます。これらは、証明のスイッチをONにする（身体障害者にとってはありがたい）、理髪店の予約をする、電話サポートにおける会話を提供する、といった技術を必要としないタスクをこなします。これは、AIがわれわれの生活に影響をもたらしている最も顕著なサインであるように思えます。

* デジタルなアシスタントなかで重要な構成要素となっているのは、音声を精度よく認識する能力です。音声認識の精度は、特定のアプリケーションにおいて、人間と同等の精度にまで徐々に改善してきました [14]。

* 同様に物体の認識も長い道のりを経ています。画像内の物体を推定することは2010年においては全く困難なタスクでした。ImageNetのベンチーマークにおいては、Linら[15]がtop-5のエラー率 (推定結果の上位5件に正解がない割合)は28%でした。2017年までには、Huら[16]がこのエラー率を2.25%まで低減しました。同様に素晴らしい結果が、鳥の認識や皮膚がんの診断においても達成されています。

* 以前はゲームは人間の知性の砦だったでしょう。Temporal difference (TD)を利用した強化学習でBackgrammonをプレイするTDGammon[23]から始まり、アルゴリズムや演算能力の進歩は、幅広いアプリケーションを対象としたアルゴリズムをリードしてきました。Backgammonとは異なり、チェスはさらに複雑な状態空間と行動集合をもっています。DeepBlueは、Campbellら[17]らが巨大な並列処理、特別なハードウェア、ゲーム木による効率な探索を利用し、Gary Kasparovを破りました。囲碁はその巨大な状態空間のためにさらに困難です。AlphaGoは2015年に人間と肩を並べるようになり、Silverら[18]はモンテカルロ木のサンプリングと深層学習を組み合わせて利用しました。ポーカーにおいては、その状態空間は大規模で完全に観測されない（相手のカードを知ることができない）という難しさがあります。Libratusは、BrownとSandholm[19]による、効率的で構造化された戦略を用いることで、ポーカーにおいて人間の能力を超えました。このことは、ゲームにおける優れた進歩と、先進的なアルゴリズムがゲームにおける重要な部分で機能した事実を表しています。

* AIの進歩における別の兆候としては、車やトラックの自動運転の登場でしょう。完全自動化には全く及んでいないとはいえ、この方向性には重要な進歩がありました。例えば、[Momenta](https://www.momenta.ai/en)、[Tesla](http://www.tesla.com), [NVIDIA](http://www.nvidia.com)、[MobilEye](http://www.mobileye.com)、[Waymo](http://www.waymo.com)は、少なくとも部分的な自動化を可能にした商品を販売しています。完全自動化がなぜ難しいかというと、正しい運転にはルールを認識して論理的に考え、システムに組み込むことが必要だからです。現在は、深層学習はこれらの問題のうち、コンピュータビジョンの側面で主に利用されています。残りはエンジニアによって非常に調整されているのです。

繰り返しますが、上記のリストは知性というものがとういうもので、機械学習がある分野において素晴らしい進歩をもたらしたということについて、表面的な内容を走り書きしたに過ぎません。例えば、ロボティクス、ロジスティクス、計算生物学、粒子物理学、天文学においては、それらの優れた近年の進歩に関して、部分的にでも機械学習によるところがあります。機械学習は、こうしてエンジニアやサイエンティストにとって、広く利用されるツールになったのです。

しばしば、AIの黙示録や特異点に関する質問が、AIに関する技術的でない
記事に取り上げられることがあります。機械学習システムが知覚を持ち、プログラマー（や管理者）の意図にかかわらず、人間の生活に直接的に影響を及ぼすものごとを決定することについて、恐れられているのです。ある程度は、AIはすでに人間の生活に直接的に影響を与えています。つまり、信用力は自動で評価されえいますし、自動パイロットは車をほとんど安全にナビゲーションしますし、保釈をしてよいかどうかを決める際には統計データが利用されています。取るに足らない例ではありますが、われわれは、インターネットにつながってさえいれば、AlexaにコーヒーマシンのスイッチをONにするようお願いでき、Alexaは要望に応えることができるでしょう。

幸運にも人間を奴隷として扱う（もしくはコーヒーを焦がしてしまう）ような知覚のあるAIシステムはまだ遠い未来にあります。まず、AIシステムは、ある目的に特化した方法で開発され、学習され、展開されます。AIシステムの行動はあたかも汎用的な知性をもっているかのように幻想をみせるかもしれませんが、それらは設計されたルールや経験則、統計モデルの組み合わせになっています。次に現時点では、あらゆるタスクを解決するために、自分自身を改善し、それらを論理的に考え、アーキテクチャを修正、拡張、改善するような、汎用人工知能の方法というのは存在していません。

さらに現実的な心配ごととしては、AIがどのように日々の生活に利用されるかです。トラックの運転手や店舗のアシスタントが担っている、技術を必要としないたくさんのタスクは自動化されるでしょう。農業ロボットは有機栽培のコストを下げるでしょうが、収穫作業も自動化してしまうしょう。この産業革命となるフェーズでは、社会における広い範囲に重大な結果を及ぼすでしょう（トラックの運転手や店舗のアシスタントは、多くの州において最も広く行われている仕事です）。さらに、統計モデルは注意せずに利用されれば、人種、性別、年齢による差別を生じる可能性があります。これらのアルゴリズムを、必ず注意して利用ことは重要です。このことは、人類を滅亡させるような悪意ある超人的な知能や意思を心配するよりもずっと、懸念されることなのです。


## まとめ

* 機械学習は、コンピュータのシステムが性能を改善するために、データの利用方法を研究するものです。それは、統計学、データマイニング、人工知能、最適化の考え方を組み合わせています。そして、人工知能的なソリューションを実装するためによく利用されます。

* 機械学習の一種で、表現学習というものは、データの最適な表現を自動的に探す方法に焦点を当てています。多くの場合、これは学習によるデータ変換の発展によって実現されました。

* 最近の進歩の多くは、安価なセンサー、インターネット上に広がるアプリケーションから得られる豊富なデータと、GPUを主にする演算能力の素晴らしい進歩がきっかけになっています。

* 全体のシステム最適化は、良い性能を得るために重要な構成要素です。効率的な深層学習フレームワークによって、最適化を非常に簡単に設計して、実装することができます。

## 課題

1. あなたが現在書いているコードの中で、学習可能な部分はどこでしょうか。言い換えれば、コードの中で設計に関する選択をしている部分で、学習によって改善できて、自動的に決定できる部分がありますか?あなたのコードは、経験則にもとづいて設計に関する選択する部分を含んでいますか?

1. あなたが直面した問題で、その問題を解くのに十分なデータがあるけども、自動化する手法がないような問題はどういうものでしょうか？これらは、深層学習を適用できる最初の候補になるかもしれません。

1. 人工知能の発展を新たな産業革命としてみたとき、アルゴリズムとデータの関係は何にあたるでしょうか?それは蒸気機関と石炭に似ています。根本的な違いは何でしょうか。

1. End-to-endの学習を適用できる他の分野はどこでしょうか? 物理学?
工学?経済学?

## 参考文献

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


## [議論](https://discuss.mxnet.io/t/2310)のためのQRコードをスキャン

![](../img/qr_intro.svg)

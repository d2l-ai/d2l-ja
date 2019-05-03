# Introduction

Until recently, nearly all of the computer programs
that we interacted with every day were coded
by software developers from first principles.
Say that we wanted to write an application to manage an e-commerce platform.
After huddling around a whiteboard for a few hours to ponder the problem,
we would come up with the broad strokes of a working solution
that would probably look something like this:
(i) users would interact with the application
through an interface running in a web browser or mobile application
(ii) our application would rely on a commerical database engine
to keep track of each user's state and maintain records
of all historical transactions
(ii) at the heart of our application, running in parallel across many servers, the *business logic* (you might say, the *brains*)
would map out in methodical details the appropriate action to take
in every conceivable circumstance.

To build the *brains* of our application,
we'd have to step through every possible corner case
that we anticipate encountering, devising appropriate rules.
Each time a customer clicks to add an item to their shopping cart,
we add an entry to the shopping cart database table,
associating that user's ID with the requested product’s ID.
While few developers ever get it completely right the first time
(it might take some test runs to work out the kinks),
for the most part, we could write such a program from first principles
and confidently launch it *before ever seeing a real customer*.
Our ability to design automated systems from first principles
that drive functioning products and systems,
often in novel situations, is a remarkable cognitive feat.
And when you're able to devise solutions that work $100\%$ of the time.
*you should not be using machine learning*.

Fortunately—for the growing community of ML scientists—many
problems in automation don't bend so easily to human ingenuity.
Imagine huddling around the whiteboard with the smartest minds you know,
but this time you are tackling any of the following problems:
 * Write a program that predicts tomorrow's weather
given geographic information, satellite images,
and a trailing window of past weather.
 * Write a program that takes in a question,
 expressed in free-form text, and answers it correctly.
 * Write a program that given an image
 can identify all the people it contains,
 drawing outlines around each.
  * Write a program that presents users with products
  that they are likely to enjoy but unlikely,
  in the natural course of browsing, to encounter.

In each of these cases, even elite programmers
are incapable of coding up solutions from scratch.
The reasons for this can vary.
Sometimes the program that we are looking for
follows a pattern that changes over time,
and we need our programs to adapt.
In other cases, the relationship
(say between pixels, and abstract categories)
may be too complicated, requiring thousands or millions of computations
that are beyond our conscious understanding
(even if our eyes manage the task effortlessly).
Machine learning (ML) is the study of powerful techniques
that can *learn behavior* from *experience*.
As ML algorithm accumulates more experience,
typically in the form of observational data
or interactions with an environment, their performance improves.
Contrast this with our deterministic e-commerce platform,
which performs according to the same business logic,
no matter how much experience accrues,
until the developers themselves *learn* and decide
that it's time to update the software.
In this book, we will teach you the fundamentals of machine learning,
and focus in particular on deep learning,
a powerful set of techniques driving innovations
in areas as diverse as computer vision, natural language processing,
healthcare, and genomics.


## 身近な例

この書籍の著者は、これを書き始めるようとするときに、多くの仕事を始める場合と同様に、カフェインを摂取していました。そして車に飛び乗って運転を始めたのです。iPhoneをもって、Alexは"Hey Siri"と言って、スマートフォンの音声認識システムを起動します。
Muは「ブルーボトルのコーヒーショップへの行き方」と言いました。するとスマートフォンは、すぐに彼の指示内容を文章で表示しました。そして、行き方に関する要求を認識して、意図に合うようにマップのアプリを起動したのです。起動したマップはたくさんのルートを提示しました。この本で伝えたいことのために、この物語をでっちあげてみましたが、たった数秒の間における、スマートフォンとの日々のやりとりに、様々な機械学習モデルが利用されているのです。

"Alexa"、"Okay Google"、"Siri"といった起動のための言葉に反応するコードを書くことを考えてみます。コンピュータとコードを書くエディタだけをつかって部屋でコードを書いてみましょう。どうやって第一原理に従ってコードを書きますか？考えてみましょう。問題は難しいです。だいたい44,000/秒でマイクからサンプルを集めます。音声の生データに対して、起動の言葉が含まれているかを調べて、YesかNoを正確に対応付けるルールとは何でしょうか？行き詰まってしまうと思いますが心配ありません。このようなプログラムを、ゼロから書く方法はだれもわかりません。これこそが機械学習を使う理由なのです。


ひとつの考え方を紹介したいと思います。私達が、入力と出力を対応付ける方法をコンピュータに陽に伝えることができなくても、私達自身はそのような認識を行う素晴らしい能力を持っています。言い換えれば、たとえ"Alexa"といった言葉を認識するようにコンピュータのプログラムを書けなくても、あなた自身は"Alexa"の言葉を認識できます。従って、私達人間は、音声のデータと起動の言葉を含んでいるか否かのラベルをサンプルとして含む巨大なデータセットをつくることができます。
機械学習のアプローチでは、起動の言葉を認識するようなシステムを陽に実装しません。
代わりに、膨大なパラメータによって挙動を決められるような、柔軟なプログラムを実装します。Then we use the dataset to determine
the best possible set of parameters,
those that improve the performance of our program
with respect to some measure of performance on the task of interest.


You can think of the parameters as knobs that we can turn,
manipulating the behavior of the program.
Fixing the parameters, we call the program a *model*.
The set of all distinct programs (input-output mappings)
that we can produce just by manipulating the parameters
is called a *family* of models.
And the *meta-program* that uses our dataset
to choose the parameters is called a *learning algorithm*.

Before we can go ahead and engage the learning algorithm,
we have to define the problem precisely,
pinning down the exact nature of the inputs and outputs,
and choosing an appropriate model family.
In this case, our model receives a snippet of audio as *input*,
and it generates a selection among ``{yes, no}`` as *output*—which,
if all goes according to plan,
will closely approximate whether (or not)
the snippet contains the wake word.


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

Deep learning is just one among many popular frameworks for solving machine learning problems. While thus far, we've only talked about machine learning broadly and not deep learning, there's a couple points worth sneaking in here: First, the problems that we've discussed thus far: learning from raw audio signal, directly from the pixels in images, and mapping between sentences of arbitrary lengths and across languages are problems where deep learning excels and traditional ML tools faltered. Deep models are deep in precisely the sense that they learn many layers of computation. It turns out that these many-layered (or hierarchical) models are capable of addressing low-level perceptual data in a way that previous tools could not. In bygone days, the crucial part of applying ML to these problems consisted of coming up with manually engineered ways of transforming the data into some form amenable to shallow models. One key advantage of deep learning is that it replaces not only the shallow models at the end of traditional learning pipelines, but also the labor-intensive feature engineering. Secondly, by replacing much of the domain-specific preprocessing, deep learning has eliminated many of the boundaries that previously separated computer vision, speech recognition, natural language processing, medical informatics, and other application areas, offering a unified set of tools for tackling diverse problems.


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



## Roots

Although deep learning is a recent invention, humans have held the desire to analyze data and to predict future outcomes for centuries. In fact, much of natural science has its roots in this. For instance, the Bernoulli distribution is named after [Jacob Bernoulli (1655-1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli), and the Gaussian distribution was discovered by [Carl Friedrich Gauss (1777-1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss). He invented for instance the least mean squares algorithm, which is still used today for a range of problems from insurance calculations to medical diagnostics. These tools gave rise to an experimental approach in natural sciences - for instance, Ohm's law relating current and voltage in a resistor is perfectly described by a linear model.

Even in the middle ages mathematicians had a keen intuition of estimates. For instance, the geometry book of [Jacob Köbel (1460-1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) illustrates averaging the length of 16 adult men's feet to obtain the average foot length.

![Estimating the length of a foot](../img/koebel.jpg)

Figure 1.1 illustrates how this estimator works. 16 adult men were asked to line up in a row, when leaving church. Their aggregate length was then divided by 16 to obtain an estimate for what now amounts to 1 foot. This 'algorithm' was later improved to deal with misshapen feet - the 2 men with the shortest and longest feet respectively were sent away, averaging only over the remainder. This is one of the earliest examples of the trimmed mean estimate.

Statistics really took off with the collection and availability of data. One of its titans, [Ronald Fisher (1890-1962)](https://en.wikipedia.org/wiki/Ronald_Fisher), contributed significantly to its theory and also its applications in genetics. Many of his algorithms (such as Linear Discriminant Analysis) and formulae (such as the Fisher Information Matrix) are still in frequent use today (even the Iris dataset that he released in 1936 is still used sometimes to illustrate machine learning algorithms).

A second influence for machine learning came from Information Theory [(Claude Shannon, 1916-2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the Theory of computation via [Alan Turing (1912-1954)](https://en.wikipedia.org/wiki/Alan_Turing). Turing posed the question "can machines think?” in his famous paper [Computing machinery and intelligence](https://www.jstor.org/stable/2251299) (Mind, October 1950). In what he described as the Turing test, a machine can be considered intelligent if it is difficult for a human evaluator to distinguish between the replies from a machine and a human being through textual interactions. To this day, the development of intelligent machines is changing rapidly and continuously.

Another influence can be found in neuroscience and psychology. After all, humans clearly exhibit intelligent behavior. It is thus only reasonable to ask whether one could explain and possibly reverse engineer these insights. One of the oldest algorithms to accomplish this was formulated by [Donald Hebb (1904-1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).

In his groundbreaking book [The Organization of Behavior](http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf) (John Wiley & Sons, 1949) he posited that neurons learn by positive reinforcement. This became known as the Hebbian learning rule. It is the prototype of Rosenblatt's perceptron learning algorithm and it laid the foundations of many stochastic gradient descent algorithms that underpin deep learning today: reinforce desirable behavior and diminish undesirable behavior to obtain good weights in a neural network.

Biological inspiration is what gave Neural Networks its name. For over a century (dating back to the models of Alexander Bain, 1873 and James Sherrington, 1890) researchers have tried to assemble computational circuits that resemble networks of interacting neurons. Over time the interpretation of biology became more loose but the name stuck. At its heart lie a few key principles that can be found in most networks today:

* The alternation of linear and nonlinear processing units, often referred to as 'layers'.
* The use of the chain rule (aka backpropagation) for adjusting parameters in the entire network at once.

After initial rapid progress, research in Neural Networks languished from around 1995 until 2005. This was due to a number of reasons. Training a network is computationally very expensive. While RAM was plentiful at the end of the past century, computational power was scarce. Secondly, datasets were relatively small. In fact, Fisher's 'Iris dataset' from 1932 was a popular tool for testing the efficacy of algorithms. MNIST with its 60,000 handwritten digits was considered huge.

Given the scarcity of data and computation, strong statistical tools such as Kernel Methods, Decision Trees and Graphical Models proved empirically superior. Unlike Neural Networks they did not require weeks to train and provided predictable results with strong theoretical guarantees.

## The Road to Deep Learning

Much of this changed with the ready availability of large amounts of data, due to the World Wide Web, the advent of companies serving hundreds of millions of users online, a dissemination of cheap, high quality sensors, cheap data storage (Kryder's law), and cheap computation (Moore's law), in particular in the form of GPUs, originally engineered for computer gaming. Suddenly algorithms and models that seemed computationally infeasible became relevant (and vice versa). This is best illustrated in the table below:

|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 PF (Nvidia DGX-2)|

It is quite evident that RAM has not kept pace with the growth in data. At the same time, the increase in computational power has outpaced that of the data available. This means that statistical models needed to become more memory efficient (this is typically achieved by adding nonlinearities) while simultaneously being able to spend more time on optimizing these parameters, due to an increased compute budget. Consequently the sweet spot in machine learning and statistics moved from (generalized) linear models and kernel methods to deep networks. This is also one of the reasons why many of the mainstays of deep learning, such as Multilayer Perceptrons (e.g. McCulloch & Pitts, 1943), Convolutional Neural Networks (Le Cun, 1992), Long Short Term Memory (Hochreiter & Schmidhuber, 1997), Q-Learning (Watkins, 1989), were essentially 'rediscovered' in the past decade, after laying dormant for considerable time.

The recent progress in statistical models, applications, and algorithms, has sometimes been likened to the Cambrian Explosion: a moment of rapid progress in the evolution of species. Indeed, the state of the art is not just a mere consequence of available resources, applied to decades old algorithms. Note that the list below barely scratches the surface of the ideas that have helped researchers achieve tremendous progress over the past decade.

* Novel methods for capacity control, such as Dropout [3] allowed for training of relatively large networks without the danger of overfitting, i.e. without the danger of merely memorizing large parts of the training data. This was achieved by applying noise injection [4] throughout the network, replacing weights by random variables for training purposes.
* Attention mechanisms solved a second problem that had plagued statistics for over a century: how to increase the memory and complexity of a system without increasing the number of learnable parameters. [5] found an elegant solution by using what can only be viewed as a learnable pointer structure. That is, rather than having to remember an entire sentence, e.g. for machine translation in a fixed-dimensional representation, all that needed to be stored was a pointer to the intermediate state of the translation process. This allowed for significantly increased accuracy for long sentences, since the model no longer needed to remember the entire sentence before beginning to generate sentences.
* Multi-stage designs, e.g. via the Memory Networks [6] and the Neural Programmer-Interpreter [7] allowed statistical modelers to describe iterative approaches to reasoning. These tools allow for an internal state of the deep network to be modified repeatedly, thus carrying out subsequent steps in a chain of reasoning, similar to how a processor can modify memory for a computation.
* Another key development was the invention of Generative Adversarial Networks [8]. Traditionally statistical methods for density estimation and generative models focused on finding proper probability distributions and (often approximate) algorithms for sampling from them. As a result, these algorithms were largely limited by the lack of flexibility inherent in the statistical models. The crucial innovation in GANs was to replace the sampler by an arbitrary algorithm with differentiable parameters. These are then adjusted in such a way that the discriminator (effectively a two-sample test) cannot distinguish fake from real data. Through the ability to use arbitrary algorithms to generate data it opened up density estimation to a wide variety of techniques. Examples of galloping Zebras [9] and of fake celebrity faces [10] are both testimony to this progress.
* In many cases a single GPU is insufficient to process the large amounts of data available for training. Over the past decade the ability to build parallel distributed training algorithms has improved significantly. One of the key challenges in designing scalable algorithms is that the workhorse of deep learning optimization, stochastic gradient descent, relies on relatively small minibatches of data to be processed. At the same time, small batches limit the efficiency of GPUs. Hence, training on 1024 GPUs with a minibatch size of, say 32 images per batch amounts to an aggregate minibatch of 32k images. Recent work, first by Li [11],  and subsequently by You et al. [12] and Jia et al. [13] pushed the size up to 64k observations, reducing training time for ResNet50 on ImageNet to less than 7 minutes. For comparison - initially training times were measured in the order of days.
* The ability to parallelize computation has also contributed quite crucially to progress in reinforcement learning, at least whenever simulation is an option. This has led to significant progress in computers achieving superhuman performance in Go, Atari games, Starcraft, and in physics simulations (e.g. using MuJoCo). See e.g. Silver et al. [18] for a description of how to achieve this in AlphaGo. In a nutshell, reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e. whenever it is possible to try out lots of things to learn how they relate to each other. Simulation provides such an avenue.
* Deep Learning frameworks have played a crucial role in disseminating ideas. The first generation of frameworks allowing for easy modeling encompassed [Caffe](https://github.com/BVLC/caffe), [Torch](https://github.com/torch), and [Theano](https://github.com/Theano/Theano). Many seminal papers were written using these tools. By now they have been superseded by [TensorFlow](https://github.com/tensorflow/tensorflow), often used via its high level API [Keras](https://github.com/keras-team/keras), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), and [Apache MxNet](https://github.com/apache/incubator-mxnet). The third generation of tools, namely imperative tools for deep learning, was arguably spearheaded by [Chainer](https://github.com/chainer/chainer), which used a syntax similar to Python NumPy to describe models. This idea was adopted by [PyTorch](https://github.com/pytorch/pytorch) and the [Gluon API](https://github.com/apache/incubator-mxnet) of MxNet. It is the latter that this course uses to teach Deep Learning.

The division of labor between systems researchers building better tools for training and statistical modelers building better networks has greatly simplified things. For instance, training a linear logistic regression model used to be a nontrivial homework problem, worthy to give to new Machine Learning PhD students at Carnegie Mellon University in 2014. By now, this task can be accomplished with less than 10 lines of code, putting it firmly into the grasp of programmers.

## Success Stories

Artificial Intelligence has a long history of delivering results that would be difficult to accomplish otherwise. For instance, mail is sorted using optical character recognition. These systems have been deployed since the 90s (this is, after all, the source of the famous MNIST and USPS sets of handwritten digits). The same applies to reading checks for bank deposits and scoring creditworthiness of applicants. Financial transactions are checked for fraud automatically. This forms the backbone of many e-commerce payment systems, such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, MasterCard. Computer programs for chess have been competitive for decades. Machine learning feeds search, recommendation, personalization and ranking on the internet. In other words, artificial intelligence and machine learning are pervasive, albeit often hidden from sight.

It is only recently that AI has been in the limelight, mostly due to solutions to problems that were considered intractable previously.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, or Google's assistant are able to answer spoken questions with a reasonable degree of accuracy. This includes menial tasks such as turning on light switches (a boon to the disabled) up to making barber's appointments and offering phone support dialog. This is likely the most noticeable sign that AI is affecting our lives.

* A key ingredient in digital assistants is the ability to recognize speech accurately. Gradually the accuracy of such systems has increased to the point where they reach human parity [14] for certain applications.
* Object recognition likewise has come a long way. Estimating the object in a picture was a fairly challenging task in 2010. On the ImageNet benchmark Lin et al. [15] achieved a top-5 error rate of 28%. By 2017 Hu et al. [16] reduced this error rate to 2.25%. Similarly stunning results have been achieved for identifying birds, or diagnosing skin cancer.
* Games used to be a bastion of human intelligence. Starting from TDGammon [23], a program for playing Backgammon using temporal difference (TD) reinforcement learning, algorithmic and computational progress has led to algorithms for a wide range of applications. Unlike Backgammon, chess has a much more complex state space and set of actions. DeepBlue beat Gary Kasparov, Campbell et al. [17], using massive parallelism, special purpose hardware and efficient search through the game tree. Go is more difficult still, due to its huge state space. AlphaGo reached human parity in 2015,  Silver et al. [18] using Deep Learning combined with Monte Carlo tree sampling. The challenge in Poker was that the state space is large and it is not fully observed (we don't know the opponents' cards). Libratus exceeded human performance in Poker using efficiently structured strategies; Brown and Sandholm [19]. This illustrates the impressive progress in games and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars and trucks. While full autonomy is not quite within reach yet, excellent progress has been made in this direction, with companies such as [Momenta](https://www.momenta.ai/en), [Tesla](http://www.tesla.com), [NVIDIA](http://www.nvidia.com), [MobilEye](http://www.mobileye.com) and [Waymo](http://www.waymo.com) shipping products that enable at least partial autonomy. What makes full autonomy so challenging is that proper driving requires the ability to perceive, to reason and to incorporate rules into a system. At present, Deep Learning is used primarily in the computer vision aspect of these problems. The rest is heavily tuned by engineers.

Again, the above list barely scratches the surface of what is considered intelligent and where machine learning has led to impressive progress in a field. For instance, robotics, logistics, computational biology, particle physics and astronomy owe some of their most impressive recent advances at least in parts to machine learning. ML is thus becoming a ubiquitous tool for engineers and scientists.

Frequently the question of the AI apocalypse, or the AI singularity has been raised in non-technical articles on AI. The fear is that somehow machine learning systems will become sentient and decide independently from their programmers (and masters) about things that directly affect the livelihood of humans. To some extent AI already affects the livelihood of humans in an immediate way - creditworthiness is assessed automatically, autopilots mostly navigate cars safely, decisions about whether to grant bail use statistical data as input. More frivolously, we can ask Alexa to switch on the coffee machine and she will happily oblige, provided that the appliance is internet enabled.

Fortunately we are far from a sentient AI system that is ready to enslave its human creators (or burn their coffee). Firstly, AI systems are engineered, trained and deployed in a specific, goal oriented manner. While their behavior might give the illusion of general intelligence, it is a combination of rules, heuristics and statistical models that underlie the design. Second, at present tools for general Artificial Intelligence simply do not exist that are able to improve themselves, reason about themselves, and that are able to modify, extend and improve their own architecture while trying to solve general tasks.

A much more realistic concern is how AI is being used in our daily lives. It is likely that many menial tasks fulfilled by truck drivers and shop assistants can and will be automated. Farm robots will likely reduce the cost for organic farming but they will also automate harvesting operations. This phase of the industrial revolution will have profound consequences on large swaths of society (truck drivers and shop assistants are some of the most common jobs in many states). Furthermore, statistical models, when applied without care can lead to racial, gender or age bias. It is important to ensure that these algorithms are used with great care. This is a much bigger concern than to worry about a potentially malevolent superintelligence intent on destroying humanity.

## Summary

* Machine learning studies how computer systems can use data to improve performance. It combines ideas from statistics, data mining, artificial intelligence and optimization. Often it is used as a means of implementing artificially intelligent solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. This is often accomplished by a progression of learned transformations.
* Much of the recent progress has been triggered by an abundance of data arising from cheap sensors and internet scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining good performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

## Exercises

1. Which parts of code that you are currently writing could be 'learned', i.e. improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using Deep Learning.
1. Viewing the development of Artificial Intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
1. Where else can you apply the end-to-end training approach? Physics? Engineering? Econometrics?

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


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2310)

![](../img/qr_intro.svg)

# はじめに

この書籍の著者は、これを書き始めるようとするときに、多くの仕事を始める場合と同様に、カフェインを摂取していました。そして車に飛び乗って運転を始めたのです。iPhoneをもって、Alexは"Hey Siri"と言って、スマートフォンの音声認識システムを起動します。
Muは「ブルーボトルのコーヒーショップへの行き方」と言いました。するとスマートフォンは、すぐに彼の指示内容を文章で表示しました。そして、行き方に関する要求を認識して、意図に合うようにマップのアプリを起動したのです。起動したマップはたくさんのルートを提示しました。この本で伝えたいことのために、この物語をでっちあげてみましたが、たった数秒の間における、スマートフォンとの日々のやりとりに、様々な機械学習モデルが利用されているのです。

もし、これまで機械学習に取り組んだことがなければ、著者が何を言っているのか不思議に思うかもしれません。「ただのプログラミングとは違うの？」とか「機械学習ってどういう意味？」といった疑問をもつと思います。まずはじめに、明確にしておきますが、機械学習のアルゴリズムはすべてコンピュータのプログラミングという形で実装されます。ここでは、他のコンピュータサイエンスの分野でも利用されるようなプログラミング言語とハードウェアを利用しますが、すべてのコンピュータプログラムが機械学習と関係しているわけではありません。2つ目の質問に関しては、機械学習のような広範囲に渡る学問分野を定義することは難しいです。それは、例えば、「数学って何?」といった質問に答えるようなものです。しかし、始めるにあたって、直感的でわかりやすい説明を心がけようと思います。


## 機械学習の適用例

私達が普段利用しているコンピュータプログラムの多くは、根本的に成立する第一原理にもとづいて実装されます。ショッピングカートに商品を追加するとき、e-commerceのアプリを実行して、shopping cartという、ユーザIDと商品IDが結びついたデータベースのテーブルにエントリを追加すると思います。このようなプログラムは、実際のお客さんに会ったことがなくても成立する第一原理から実装します。もし、このような簡単なアプリケーションを実装しようとするなら、機械学習を使わないほうが良いです。

これは機械学習のサイエンティストにとって幸運なこともかもしれませんが、多くの問題に対するソリューションは簡単なものではありません。 コーヒーを買いに行くという嘘の話に戻りましょう。"Alexa"、"Okay Google"、"Siri"といった起動のための言葉に反応するコードを書くことを考えてみます。コンピュータとコードを書くエディタだけをつかって部屋でコードを書いてみましょう。どうやって第一原理に従ってコードを書きますか？考えてみましょう。問題は難しいです。だいたい44,000/秒でマイクからサンプルを集めます。音声の生データに対して、起動の言葉が含まれているかを調べて、YesかNoを正確に対応付けるルールとは何でしょうか？行き詰まってしまうと思いますが心配ありません。このようなプログラムを、ゼロから書く方法はだれもわかりません。これこそが機械学習を使う理由なのです。

![](../img/wake-word.svg)

ちょっとした考え方を紹介したいと思います。私達が、入力と出力を対応付ける方法をコンピュータに陽に伝えることができなくても、私達自身はそのような認識を行う素晴らしい能力を持っています。言い換えれば、たとえ"Alexa"といった言葉を認識するようにコンピュータのプログラムを書けなくても、あなた自身は"Alexa"の言葉を認識できます。従って、私達人間は、音声のデータと起動の言葉を含んでいるか否かのラベルをサンプルとして含む巨大なデータセットをつくることができます。機械学習を使うと、起動の言葉を直ちに認識するようなシステムを陽に実装する必要がありません。代わりに、膨大なパラメータで柔軟なプログラムを実装します。パラメータは、プログラムの挙動を柔軟に変更・調整するためのものです。私達はこのプログラムのことをモデルとよんでいます。一般的に、モデルは入力を出力に変換するような機械に過ぎません。上記の例では、モデルは一部の音声を入力として受け取って、それが起動の言葉を含んでいるかどうかを判定できることを期待し、YesかNoを回答として返します。

もし私達が、正しいモデルを選ぶことができれば、そのモデルは"Alexa"という言葉を聞いたときに"Yes"と出力するペアが必要になります。"Yes"という言葉は"Apricot"に置き換えることもできるでしょう。"Alexa"を認識する機能は"Apricot"を認識する機能は非常に似たようなものであり、同じモデルが適用できると思うかもしれません。しかし、入出力が変われば根本的に別のモデルを必要とする場合もあります。例えば、画像とラベルを対応付けるタスクと、英語と中国語を対応付けるタスクには、異なるモデルを利用する必要があるでしょう。

想像できるとは思いますが、"Alexa"と"Yes"のようなペアがランダムなものであれば、モデルは"Alexa"も"Apricot"も他の英語も何も認識できないでしょう。Deep Learningという言葉の中のLearningというのは、学習の期間において、複数のペアを上手く使って、モデルの挙動を更新していくことを指します。その学習のプロセスというのは以下のようなものです。

1. まず最初にランダムに初期化されたモデルから始めます。このモデルは最初は使い物になりません
1. ラベルデータを取り込みます（例えば、部分的な音声データと対応するYes/Noのラベルです）
1. そのペアを利用してモデルを改善します
1. モデルが良いものになるまで繰り返します

![](../img/ml-loop.svg)

まとめると、起動の言葉を認識できるようなコードを直接書くよりも、*もしラベル付きのデータをたくさん集められるなら*、その認識機能を*学習*するようなコードを書くべきです。これは、プログラムの挙動をデータセットで決める、つまり*データでプログラムを書く*ようなものだと考えることができます。

私達は、以下に示すようなネコとイヌの画像サンプルを大量に集めて、機械学習を利用することで、ネコ認識器をプログラムすることができます。


|![](../img/cat1.png)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
|:---------------:|:---------------:|:---------------:|:---------------:|
|ネコ|ネコ|イヌ|イヌ|

このケースでは、認識器はネコであれば非常に大きな正の値、イヌであれば非常に大きな負の値、どちらかわからない場合はゼロを出力するように学習するでしょう。しかし、機械学習が行うようなイヌとネコを識別するための境界を、ゼロからプログラミングするわけではありません。

## 機械学習の多機能性

これは機械学習を支える核となる考え方ですが、ある特定の挙動に関して直接プログラムとして書くよりは、まるで経験を獲得していくように、挙動を改善する能力をプログラミングします。この基本的な考え方は様々な形式で実装されます。機械学習は、多くの異なる分野で利用されており、多様なモデルに関係し、異なるアルゴリズムに応じてモデルを更新してきました。この例として、音声認識の問題に対する*教師あり学習*について述べたいと思います。

単純なルールベースのシステムが上手く行かなかったり、構築が非常に難しかったり、といった様々な状況において、私達がデータを利用して作業するための、多数のツール群が機械学習と言えます。例えば、機械学習の技術は検索エンジン、自動運転、機械翻訳、医療診断、スパムフィルタ、ゲームプレイ(チェスや囲碁)、顔認識、データマッチング、保険料の計算、写真の加工フィルタなど、すでに幅広く利用されています。

これらの問題は表面上は違いますが、多くのものは共通の問題構造をもっていて、Deep Learningで扱えるものもあります。挙動を直接的にコードで記述できませんが、*データでプログラムする*、という点においては似通っています。このようなプログラムは*数学*という共通言語によってつながっています。この書籍では、数学の記述に関して最小限にとどめつつ、他の機械学習やニューラルネットワークの書籍とは違って、実際の例とコードにもとづいて説明をしたいと思います。


## 機械学習の基礎
起動のための言葉を認識するタスクを考えたとき、音声データとラベルからなるデータセットを準備します。そこで、音声からラベルを推定する機械学習モデルをどうやって学習させるかを記述するかもしれません。サンプルからラベルを推定するこのセットアップは機械学習の一種で、*教師あり学習*と呼ばれるものです。Deep Learningにおいても多くのアプローチがありますが、それについては以降の章で述べたいと思います。機械学習を進めるために、以下の４つのことが必要になります。

1. データ
1. データを変換して推定するためのモデル
1. そのモデルが上手く行っているかどうかを測るためのロス関数
1. ロス関数を最小化するような、モデルのパラメータを探すアルゴリズム

### データ

一般的に、多くのデータを持っていれば持っているほど、問題を簡単に解くことができます。多くのデータを持っているなら、より性能の良いモデルを構築することができるからです。データはDeep Learningの発達に大きく貢献し、現在の多くのDeep Learningモデルは大規模なデータセットなしには動きません。以下では、機械学習を実践するうえで扱うことが多いデータの例を示します。

* **画像** スマートフォンで撮影されたり、Webで収集された画像、衛星画像、超音波やCTやMRIなどのレントゲン画像など

* **テキスト** Eメール、学校でのエッセイ、tweet、ニュース記事、医者のメモ、書籍、翻訳文のコーパスなど

* **音声** Amazon Echo、iPhone、Androidのようなスマートデバイスに送られる音声コマンド、音声つき書籍、通話、音楽など

* **動画** テレビや映画、Youtubeのビデオ、携帯電話の撮影、自宅の監視カメラ映像、複数カメラによる追跡、など

* **構造化データ** ウェブページ、電子カルテ、レンタカーの記録、デジタルな請求書など


### モデル

通常、データは私達が達成しようとすることとは大きく異なっています。例えば、人間の写真を持っていて、そこに映る人たちが幸せかどうかを知りたいとします。そのために、高解像度の画像から幸福度を出力するようなモデルを必要とすると思います。簡単な問題がシンプルなモデルで解決できるかもしれないのに対し、このケースではいくつかの問いを投げかけることになります。幸福度を求めるためには、その検出器が数百、数千の低レベルな特徴（画像の場合はピクセル単位）をかなり抽象的な幸福度に変換する必要があります。そして、正しいモデルを選ぶのは難しく、異なるデータセットには異なるモデルが適しています。このコンテンツでは、モデルとしてDeep Neural Networksに着目します。これらのモデルは最初（入力）から最後（出力）までつながった、たくさんのデータ変換で構成されています。従って、*Deep Learning*と呼ばれているのです。Deep Nets、つまりDeep Learningのモデルについて議論するときは、比較的シンプルで浅いモデルを議論するようにしたいと思います。


###  ロス関数

モデルが良いかどうかを評価するためには、モデルの出力と実際の正解を比較する必要があります。ロス関数は、その出力が*悪い*ことを評価する方法です。例えば、画像から患者の心拍数を予測するモデルを学習する場合を考えます。そのモデルが心拍数は100bpmだと推定して、実際は60bpmが正解だったときには、そのモデルに対して推定結果が悪いことを伝えなくてはなりません。

同様に、Eメールがスパムである確率を予測するモデルを作りたいとき、その予測が上手く行っていなかったら、そのモデルに伝える方法が必要になります。一般的に、機械学習の*学習*と呼ばれる部分はロス関数を最小化することです。通常、モデルはたくさんのパラメータをもっています。パラメータの最適な値というのは、私達が学習で必要とするものであり、観測したデータのなかの*学習データ*の上でロスを最小化することによって得られます。残念ながら、学習データ上でいくら上手くロス関数を最小化しても、いまだ見たことがないテストデータにおいて、学習したモデルがうまくいくという保証はありません。従って、以下の2つの指標をチェックする必要があります。

* **学習誤差**: これは、学習データ上でロスを最小化してモデルを学習したときの、学習データにおける誤差です。これは、実際の試験に備えて、学生が練習問題をうまく説いているようなものです。その結果は、実際の試験の結果が良いかどうかを期待させますが、最終試験での成功を保証するものではありません。
* **テスト誤差**: これは、見たことのないテストデータに対する誤差で、学習誤差とは少し異なっています。見たことのないデータに対して、モデルが対応（汎化）できないとき、その状態を *overfitting* (過適合)と呼びます。実生活でも、練習問題に特化して準備したにもかかわらず、本番の試験で失敗するというのと似ています。

### 最適化アルゴリズム
最終的にロスを最小化するということは、モデルとそのロス関数に対して、ロスを最小化するようなモデルのパラメータを探索するということになります。ニューラルネットワークにおける最も有名な最適化アルゴリズムは、最急降下法と呼ばれる方法にもとづいています。端的に言えば、パラメーラを少しだけ動かしたとき、学習データに対するロスがどのような方向に変化するかを、パラメータごとに見ることです。ロスが小さくなる方向へパラメータを更新します。

次の節では、機械学習のいくつかの種類について詳細を議論します。まず、機械学習の*目的*、言い換えれば機械学習ができることに関して、そのリストを紹介します。その目的は、目的を達成するための*手段*、いわば学習やデータの種類と補完的な位置づけであることに気をつけてください。以下のリストは、ひとまず、読者の好奇心をそそり、問題について話し合うための共通言語を理解することを目的としています。より多くの問題については、追って紹介をしていきたいと思います。

## 教師あり学習

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



### 回帰

最も単純な教師あり学習のタスクとして頭に思い浮かぶものは、おそらく回帰ではないかと思います。例えば、住宅の売上に関するデータベースから、一部のデータセットが得られた場合を考えてみます。各列が異なる住居に、各列は関連する属性、例えば、住宅の面積、寝室の数、トイレの数、中心街まで徒歩でかかる時間に対応するような表を構成するでしょう。形式的に、このようなデータセットの1行を*特徴ベクトル*、関連する対象（今回の事例では1つの家）を*データ例*と呼びます。

もしニューヨークやサンフランシスコに住んでいて、Amazon、Google、Microsoft、FacebookなどのCEOでなければ、その特徴ベクトル（面積、寝室数、トイレ数、中心街までの距離）は$[100, 0, .5, 60]$といった感じでしょう。一方、もしピッツバーグに住んでいれば、その特徴ベクトルは$[3000, 4, 3, 10]$のようになると思います。 このような特徴ベクトルは、伝統的な機械学習のあらゆる問題において必要不可欠なものでした。あるデータ例に対する特徴ベクトルを$\mathbf{x_i}$で、全てのテータ例の特徴ベクトルを$X$として表します。

何が問題を回帰させるかというと、実はその出力なのです。もしあなたが、新居を購入しようと思っていて、上記のような特徴量を用いて、家の適正な市場価値を推定したいとしましょう。目標値は、販売価格で、これは*実数*です。あるデータ例$\mathbf{x_i}$に対する個別の目標値を$y_i$とし、すべてのデータ例$\mathbf{X}$に対応する全目標を$\mathbf{y}$とします。目標値が、ある範囲内の任意の実数をとるとき、この問題を回帰問題と呼びます。ここで作成するモデルの目的は、実際の目標値に近い予測値(いまの例では、価格の推測値)を生成することです。この予測値を$\hat{y}_i$とします。もしこの表記になじみがなければ、いまのところ無視しても良いです。以降の章では、中身をより徹底的に解説していく予定です。

多くの実践的な問題は、きちんと説明されて、わかりやすい回帰問題となるでしょう。ユーザがある動画につけるレーティングを予測する問題は回帰問題です。もし2009年にその功績をあげるような偉大なアルゴリズムを設計できていれば、[Netflixの100万ドルの賞](https://en.wikipedia.org/wiki/Netflix_Prize)を勝ち取っていたかも知れません。病院での患者の入院日数を予測する問題もまた回帰問題です。ある1つの法則として、*どれくらい?*という問題は回帰問題を示唆している、と判断することは良いかも知れません。

* '手術は何時間かかりますか?' - *回帰*
* 'この写真にイヌは何匹いますか?' - *回帰*.

しかし、「これは__ですか?」のような問題として簡単に扱える問題であれば、それはおそらく分類問題で、次に説明する全く異なる種類の問題です。たとえ、機械学習をこれまで扱ったことがなかったとしても、形式に沿わない方法で、おそらく回帰問題を扱うことができるでしょう。例えば、下水装置を直すことを考えましょう。工事業者は下水のパイプから汚れを除くのに$x_1=3$時間かかって、あなたに$y_1=350$の請求をしました。友人が同じ工事業者を雇い、$x_2 = 2$時間かかったとき、友人に$y_2=250$の請求をしました。もし、これから汚れを除去する際に、どれくらいの請求が発生するかを尋ねられたら、作業時間が長いほど料金が上がるといった、妥当な想定をすると思います。そして、基本料金があって、１時間当たりの料金もかかるという想定もするでしょう。これらの想定のもと、与えられた2つのデータを利用して、工事業者の値付け方法を特定できるでしょう。1時間当たり\$100で、これに加えて\$50の出張料金です。もし、読者がこの内容についてこれたのであれば、線形回帰のハイレベルな考え方をすでに理解できているでしょう (そして、バイアス項のついた線形モデルを暗に設計したことになります)。


この場合では、工事業者の価格に完全に一致するようなパラメータを作ることができましたが、場合によってはそれが不可能なことがあります。例えば、2つの特徴に加えて、いくらかの分散がある要因によって発生する場合です。このような場合は、予測値と観測値の差を最小化するようにモデルを学習します。本書の章の多くでは、以下の2種類の一般的なロスのうち1つに着目します。
以下の式で定義される[L1ロス](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L1Loss)と

$$l(y,y') = \sum_i |y_i-y_i'|$$

以下の式で定義される最小二乗ロス、別名[L2ロス](http://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.L2Loss)です。


$$l(y,y') = \sum_i (y_i - y_i')^2.$$

のちほど紹介しますが、$L_2$ロスはガウスノイズによってばらついているデータを想定しており、$L_1$ロスはラプラス分布のノイズによってばらついていることを想定しています。

### Classification

While regression models are great for addressing *how many?* questions,
lots of problems don't bend comfortably to this template. For example,
a bank wants to add check scanning to their mobile app.
This would involve the customer snapping a photo of a check with their smartphone's camera
and the machine learning model would need to be able to automatically understand text seen in the image.
It would also need to understand hand-written text to be even more robust.
This kind of system is referred to as optical character recognition (OCR),
and the kind of problem it solves is called a classification.
It's treated with a distinct set of algorithms than those that are used for regression.

In classification, we want to look at a feature vector, like the pixel values in an image,
and then predict which category (formally called *classes*),
among some set of options, an example belongs.
For hand-written digits, we might have 10 classes,
corresponding to the digits 0 through 9.
The simplest form of classification is when there are only two classes,
a problem which we call binary classification.
For example, our dataset $X$ could consist of images of animals
and our *labels* $Y$ might be the classes $\mathrm{\{cat, dog\}}$.
While in regression, we sought a regressor to output a real value $\hat{y}$,
in classification, we seek a *classifier*, whose output $\hat{y}$ is the predicted class assignment.

For reasons that we'll get into as the book gets more technical, it's pretty hard to optimize a model that can only output a hard categorical assignment, e.g. either *cat* or *dog*.
It's a lot easier instead to express the model in the language of probabilities.
Given an example $x$, the model assigns a probability $\hat{y}_k$ to each label $k$.
Because these are probabilities, they need to be positive numbers and add up to $1$.
This means that we only need $K-1$ numbers to give the probabilities of $K$ categories.
This is easy to see for binary classification.
If there's a 0.6 (60%) probability that an unfair coin comes up heads,
then there's a 0.4 (40%) probability that it comes up tails.
Returning to our animal classification example, a classifier might see an image
and output the probability that the image is a cat $\Pr(y=\mathrm{cat}| x) = 0.9$.
We can interpret this number by saying that the classifier is 90% sure that the image depicts a cat.
The magnitude of the probability for the predicted class is one notion of confidence.
It's not the only notion of confidence and we'll discuss different notions of uncertainty in more advanced chapters.

When we have more than two possible classes, we call the problem *multiclass classification*.
Common examples include hand-written character recognition `[0, 1, 2, 3 ... 9, a, b, c, ...]`.
While we attacked regression problems by trying to minimize the L1 or L2 loss functions,
the common loss function for classification problems is called cross-entropy.
In MXNet Gluon, the corresponding loss function can be found [here](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss).

Note that the most likely class is not necessarily the one that you're going to use for your decision. Assume that you find this beautiful mushroom in your backyard:

|![](../img/death_cap.jpg)|
|:-------:|
|Death cap - do not eat!|

Now, assume that you built a classifier and trained it
to predict if a mushroom is poisonous based on a photograph.
Say our poison-detection classifier outputs $\Pr(y=\mathrm{death cap}|\mathrm{image}) = 0.2$.
In other words, the classifier is 80% confident that our mushroom *is not* a death cap.
Still, you'd have to be a fool to eat it.
That's because the certain benefit of a delicious dinner isn't worth a 20% risk of dying from it.
In other words, the effect of the *uncertain risk* by far outweighs the benefit.
Let's look at this in math. Basically, we need to compute the expected risk that we incur, i.e. we need to multiply the probability of the outcome with the benefit (or harm) associated with it:

$$L(\mathrm{action}| x) = \mathbf{E}_{y \sim p(y| x)}[\mathrm{loss}(\mathrm{action},y)]$$

Hence, the loss $L$ incurred by eating the mushroom is $L(a=\mathrm{eat}| x) = 0.2 * \infty + 0.8 * 0 = \infty$, whereas the cost of discarding it is $L(a=\mathrm{discard}| x) = 0.2 * 0 + 0.8 * 1 = 0.8$.

Our caution was justified: as any mycologist would tell us, the above actually *is* a death cap.
Classification can get much more complicated than just binary, multiclass, of even multi-label classification.
For instance, there are some variants of classification for addressing hierarchies.
Hierarchies assume that there exist some relationships among the many classes.
So not all errors are equal - we prefer to misclassify to a related class than to a distant class.
Usually, this is referred to as *hierarchical classification*.
One early example is due to [Linnaeus](https://en.wikipedia.org/wiki/Carl_Linnaeus),
who organized the animals in a hierarchy.

![](../img/sharks.png)

In the case of animal classification, it might not be so bad to mistake a poodle for a schnauzer,
but our model would pay a huge penalty if it confused a poodle for a dinosaur.
What hierarchy is relevant might depend on how you plan to use the model.
For example, rattle snakes and garter snakes might be close on the phylogenetic tree,
but mistaking a rattler for a garter could be deadly.

### Tagging

Some classification problems don't fit neatly into the binary or multiclass classification setups.
For example, we could train a normal binary classifier to distinguish cats from dogs.
Given the current state of computer vision,
we can do this easily, with off-the-shelf tools.
Nonetheless, no matter how accurate our model gets, we might find ourselves in trouble when the classifier encounters an image of the Bremen Town Musicians.

![](../img/stackedanimals.jpg)


As you can see, there's a cat in the picture, and a rooster, a dog and a donkey, with some trees in the background.
Depending on what we want to do with our model ultimately,
treating this as a binary classification problem
might not make a lot of sense.
Instead, we might want to give the model the option
of saying the image depicts a cat *and* a dog *and* a donkey *and* a rooster.

The problem of learning to predict classes
that are *not mutually exclusive*
is called multi-label classification.
Auto-tagging problems are typically best described
as multi-label classification problems.
Think of the tags people might apply to posts on a tech blog,
e.g., 'machine learning', 'technology', 'gadgets',
'programming languages', 'linux', 'cloud computing', 'AWS'.
A typical article might have 5-10 tags applied
because these concepts are correlated.
Posts about 'cloud computing' are likely to mention 'AWS'
and posts about 'machine learning' could also deal with 'programming languages'.

We also have to deal with this kind of problem when dealing with the biomedical literature,
where correctly tagging articles is important
because it allows researchers to do exhaustive reviews of the literature.
At the National Library of Medicine, a number of professional annotators
go over each article that gets indexed in PubMed
to associate each with the relevant terms from MeSH,
a collection of roughly 28k tags.
This is a time-consuming process and the annotators typically have a one year lag between archiving and tagging. Machine learning can be used here to provide provisional tags
until each article can have a proper manual review.
Indeed, for several years, the BioASQ organization has [hosted a competition](http://bioasq.org/)
to do precisely this.


### Search and ranking

Sometimes we don't just want to assign each example to a bucket or to a real value. In the field of information retrieval, we want to impose a ranking on a set of items. Take web search for example, the goal is less to determine whether a particular page is relevant for a query, but rather, which one of the plethora of search results should be displayed for the user. We really care about the ordering of the relevant search results and our learning algorithm needs to produce ordered subsets of elements from a larger set. In other words, if we are asked to produce the first 5 letters from the alphabet, there is a difference between returning ``A B C D E`` and ``C A B E D``. Even if the result set is the same, the ordering within the set matters nonetheless.

One possible solution to this problem is to score every element in the set of possible sets along with a corresponding relevance score and then to retrieve the top-rated elements. [PageRank](https://en.wikipedia.org/wiki/PageRank) is an early example of such a relevance score. One of the peculiarities is that it didn't depend on the actual query. Instead, it simply helped to order the results that contained the query terms. Nowadays search engines use machine learning and behavioral models to obtain query-dependent relevance scores. There are entire conferences devoted to this subject.

<!-- Add / clean up-->

### Recommender systems

Recommender systems are another problem setting that is related to search and ranking. The problems are  similar insofar as the goal is to display a set of relevant items to the user. The main difference is the emphasis on *personalization* to specific users in the context of recommender systems. For instance, for movie recommendations, the results page for a SciFi fan and the results page for a connoisseur of Woody Allen comedies might differ significantly.

Such problems occur, e.g. for movie, product or music recommendation. In some cases, customers will provide explicit details about how much they liked the product (e.g. Amazon product reviews). In some other cases, they might simply provide feedback if they are dissatisfied with the result (skipping titles on a playlist). Generally, such systems strive to estimate some score $y_{ij}$, such as an estimated rating or probability of purchase, given a user $u_i$ and product $p_j$.

Given such a model, then for any given user, we could retrieve the set of objects with the largest scores $y_{ij}$, which are then used as a recommendation. Production systems are considerably more advanced and take detailed user activity and item characteristics into account when computing such scores. The following image is an example of deep learning books recommended by Amazon based on personalization algorithms tuned to the author's preferences.

![](../img/deeplearning_amazon.png)


### Sequence Learning

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

#### Tagging and Parsing

This involves annotating a text sequence with attributes. In other words, the number of inputs and outputs is essentially the same. For instance, we might want to know where the verbs and subjects are. Alternatively, we might want to know which words are the named entities. In general, the goal is to decompose and annotate text based on structural and grammatical assumptions to get some annotation. This sounds more complex than it actually is. Below is a very simple example of annotating a sentence with tags indicating which words refer to named entities.

|Tom | has | dinner | in | Washington | with | Sally.|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Ent | - | - | - | Ent | - | Ent|


#### Automatic Speech Recognition

With speech recognition, the input sequence $x$ is the sound of a speaker,
and the output $y$ is the textual transcript of what the speaker said.
The challenge is that there are many more audio frames (sound is typically sampled at 8kHz or 16kHz) than text, i.e. there is no 1:1 correspondence between audio and text,
since thousands of samples correspond to a single spoken word.
These are seq2seq problems where the output is much shorter than the input.

|`-D-e-e-p- L-ea-r-ni-ng-`|
|:--------------:|
|![Deep Learning](../img/speech.png)|

#### Text to Speech

Text to Speech (TTS) is the inverse of speech recognition.
In other words, the input $x$ is text
and the output $y$ is an audio file.
In this case, the output is *much longer* than the input.
While it is easy for *humans* to recognize a bad audio file,
this isn't quite so trivial for computers.

#### Machine Translation

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


## Unsupervised learning

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
However, if you plan to be a data scientist, you had better get used to it.
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


## Interacting with an environment

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
* have shifting dynamics (steady vs shifting over time)?

This last question raises the problem of *covariate shift*,
(when training and test data are different).
It's a problem that most of us have experienced when taking exams written by a lecturer,
while the homeworks were composed by his TAs.
We'll briefly describe reinforcement learning, and adversarial learning,
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


### MDPs, bandits, and friends

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of *special cases* of reinforcement learning problems.

When the environment is fully observed, we call the RL problem a *Markov Decision Process* (MDP).
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions with initially unknown rewards,
this problem is the classic *multi-armed bandit problem*.

## Conclusion

Machine Learning is vast. We cannot possibly cover it all. On the other hand, neural networks are simple and only require elementary mathematics. So let's get started (but first, let's install MXNet).

## Discuss on our Forum

<div id="discuss" topic_id="2314"></div>

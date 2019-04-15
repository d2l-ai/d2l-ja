# まえがき

ほんの2、３年前は、大きな企業やスタートアップにおいて知的な製品やサービスを開発するような、深層学習の科学者のチームは存在しませんでした。われわれ著者のうち、最も若い世代がこの分野に入ったときも、日々の新聞で機械学習が新聞の見出しにでることはありませんでした。われわれの両親は、われわれが医薬や法律といった関係の職業よりも機械学習を好んでいることはもちろん、機械学習が何なのかということについても知りません。機械学習は、狭い領域における実世界の応用に向けた、先進的な学習的領域だったのです。例えば音声認識やコンピュータビジョンといった応用では、機械学習というのは非常に小さな構成要素の1つで、それとは別に、多くのドメイン知識を必要としたのでした。ニューラルネットワークは、われわれがこの本で着目する深層学習モデルの元になるものですが、時代遅れなツールだと考えられていました。

5年ほど前、深層学習はコンピュータビジョン、自然言語処理、自動音声認識、強化学習、統計的モデリングといった多様な分野において急速な進歩をみせ、世界を驚かせました。このような発展を手に、われわれは自動運転を開発したり（車の自動性を高めたり）、日常の応答を予測する賢い応答システムを開発したり、山のようなメールを探すことを助けたり、碁のようなボードゲームで世界のトッププレイヤーに打ち勝つソフトウェアエージェントを開発したり、数十年はかかると思われていたものを開発したりできるようになりました。すでに、これらのツールはそのインパクトを拡大し続けており、映画の作成方法を変え、病気の診断方法を変え、天体物理学から生物学に至るまでの基礎科学における役割を大きくしています。この本では、深層学習をより身近なものとするための試みを紹介し、読者に*概念*、*具体的な内容*、*コード*のすべてをお伝えします。

## この書籍について

### コード、数学、HTMLを結びつける手段

どのようなコンピュータの技術もそのインパクトを十分に発揮するためには、十分に理解され、文書化され、成熟して十分に保守されたツールによって支援される必要があります。
キーとなるアイデアを明確な形で切り出して、新たに深層学習に取り組む人が最新の技術を身につけるための時間を最小化すべきです。成熟したライブラリは共通のタスクを自動化し、お手本となるコードは、取り組む人が必要とするものにあわせて、アプリケーションを簡単に修正、応用、拡張する手助けとなるべきです。動的なWebアプリケーションを例にあげましょう。
Amazonのような多くの企業が1990年代にデータベースを利用したWebアプリケーションの開発に成功しましたが、創造的な事業家を支援する潜在的な技術は、ここ10年で大きく実現されたものであり、それは強力で十分に文書化されたフレームワークの開発のおかげだったのです。

どんなアプリケーションも様々な学問を集めて成り立っているので、深層学習を理解することは他にはない挑戦を意味するでしょう。深層学習を適用するためには以下を理解する必要があります。

(i) ある特定の手段によって問題を投げかけるための動機  
(ii) 与えられたモデリングアプローチを構成する数学  
(iii) モデルをデータに適合させるための最適化アルゴリズム  
(iv) モデルを効率的に学習させるため工学、つまり数値計算の落とし穴にはまらないようにしたり、利用可能なハードウェアを最大限生かすこと

問題を定式化するために必要なクリティカルシンキングのスキル、その問題を解くために必要な数学、その解法を実装するためのソフトウェアについて、1つの場所で
教えることは恐ろしいほどの挑戦です。この本における、われわれのゴールは、将来の実践者に対して必要な情報を提供するための統一的なリソースを提示することです。

われわれは、この本のプロジェクトを2017年7月に開始し、そのころ、MXNetの新しいGluonのインターフェースをユーザに説明する必要がありました。当時、(1) 常に最新で、(2) 技術的な深さをもちあわせて、現代の機械学習を幅広くカバーし、(3) 魅力的な教科書に期待される説明文の中に、ハンズオンの資料で求められるような整備された実行可能なコードが含まれるようなリソースは存在しませんでした。世の中には、深層学習のフレームワークの利用方法（例えば、Tensorflowにおける行列の基本的な数値計算）や、特定の技術を実装する方法（たとえば、LeNet、AlexeNet、ResNetなどのコードスニペット）に関する十分な数のコード例が、ブログの記事やGitHubに存在しています。しかし、これらの例は典型的には与えられたアプローチを*どのように*実装するかに重点が置かれていて、*なぜ*そのアルゴリズムに決定したのかという議論はなされていません。ブログの記事は散発的にこのような内容をカバーしていて、例えば[Distill](http://distill.pub)というウェブサイトや、個人のブログなどがありますが、深層学習における限定的な内容をカバーするだけで、しばしば関連するコードがありません。一方で、いくつかの教科書が登場し、最も著名な[Goodfellow, Bengio and Courville, 2016](https://www.deeplearningbook.org/)は、深層学習を支える概念について丁寧に調査、説明しているが、これらのリソースは説明文とコードによる実装を合わせておらず、それを実装する方法について、ときどき読者に手がかかりを与えないままになっています。加えて、非常に多くのリソースは、商用の教育コースの提供者によって、課金者のみにアクセスできるようになっており隠れてしまっています。

そこで、以下のことが可能なリソースの作成に着手しました。

(1) あらゆる人にとって無料で利用できる  
(2) 実際に機械学習の応用ができるサイエンティストになるための、起点となるような十分な技術的な深さを提供する  
(3) 読者にどうやっても問題を解くかを示すような実行可能なコードを含む  
(4) 我々や大部分はコミュニティによって、すばやくアップデートされる  
(5) 技術的な詳細についての対話的な議論や質問に回答するための[forum](http://discuss.mxnet.io)によって補足される  

これらのゴールはしばしば衝突してしまいます。数式、理論、そして引用は、最善な形で管理されて、LaTeXによって組版されます。コードはPythonで最善の形で記述される。そして、ウェブページはもともとHTMLやJavaScriptによって成り立っている。さらに、実行可能なコードとして、物理的な書籍として、ダウンロード可能なPDFとして、インターネット上のWeb Siteとして、これらのいずれの場合でもアクセス可能なコンテンツを欲しいと考えていました。現在もこれらの要求に完全に応えられるツールやワークフローは存在しません。そこでわれわれは、われわれ自信でこれを構築する必要がありました。われわれは、このアプローチの詳細について [appendix](../chapter_appendix/how-to-contribute.md)に記載しています。そのソースの共有や編集の許可のためのGithub、コードと数式と文章を調和させるJupyter ノートブック、複数の出力(Webページ、PDFなど)を生成するためのSphinx、フォーラムのためのDisclosure、これらを利用することを決めました。われわれのシステムはまだ完全ではない一方で、これらの選択は互いに競合してきた関心事を互いに歩み寄らせているでしょう。この場がこのような統合的なワークフローを利用して最初に出版される本になるだろうと、われわれは信じています。


## やってみて学ぶ

多くの教科書は、一連のトピックについて網羅的に詳細を交えて説明します。例えば、 Chris Bishopの素晴らしい書籍
[Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)は、各トピックを徹底的に説明し、線形回帰の章でもとてつもない量の内容を理解する必要があります。専門家は、その書籍の完全さゆえに非常に気に入っていますが、初心者にとっての導入の書籍としてはその有用性を活かしきれません。

この本では、たいていの考え方を *just in time*、つまり必要なときに教えたいと思います。言い換えれば、いくつかの実践的な目標を達成するために、ある考え方が必要になったときに、それを教えていきます。基礎となる予備知識、例えば、線形代数や確率といったものを伝える最初の段階では少し時間がかかると思います。風変わりな確率分布について心配する前に、最初のモデルを学習することに満足さを感じてほしいと思っています。

基礎的な数学の知識に関する集中講義を提供するための、一部の初歩的なノートブックは別として、それ以降のノートブックはほどよい量の新しいコンセプトを紹介し、全てがそろった単一で実行可能な例を実際のデータセットとともに提供します。これは、１つの構成上のチャレンジと呼んでいいと思います。いくつかのモデルは、単一のノートブックの中に論理的にひとまとめにすることもできるでしょう。そして、そのいくつかのモデルを連続的に実行することによって、考え方を教えたほうが良いかもしれません。一方、* 1つの実行可能な例と1つのノートブック* のポリシーに固執するのには大きな利点があります。われわれのコードを活用して、読者のみなさんの研究プロジェクトを、なるべく簡単に立ち上げるのを助けることができます。1つのノートブックをコピーしてから、それを修正していけばよいのです。

われわれは、必要に応じて実行可能なコードなかに、それを理解するための情報を含めます。一般的に、ツールを十分に説明する前に、そのツールを使えるようにしようとして失敗しすることもあるでしょう （そして、あとになってフォローアップの説明をします）。例えば、*stochastic gradient descent* について、それがなぜ有用で、なぜうまくいくのかを十分に説明する前に使うことがあると思います。この方法は、短時間のうちのいくつかの判断だけで、読者にわれわれを信用してもらうことになりますが、実際に機械学習を行う人に対して、問題をすばやく解くための武器を与えることができます。

ここでは最初から最後まで、MXNetのライブラリを利用して進める予定です。MXNetのライブラリは研究用途にも十分に柔軟で、本番環境の用途にも十分高速であるという得がたい特徴をもっています。深層学習の考え方についてゼロから伝えていく予定で、ときどき、``Gluon``の先進的な特徴によってユーザに隠蔽されたモデルについて、詳細な部分を掘り下げたいと思います。与えられた深層学習のレイヤーの中で起こっているすべてを理解してほしいので、特に基礎的なチュートリアルにおいて掘り下げを行います。この場合に、われわれは次の2種類の例を一般的に提示します。1つはNDArray(多次元配列）や自動微分を利用してゼロか全てを実装するもので、もう1つは``Gluon``によって同じことを簡潔に実装するものです。レイヤーがどのように動くかを伝えたら、以降のチュートリアルでは、``Gluon``を利用したものを使います。

### コンテンツと構成

The book can be roughly divided into three sections:

* The first part covers prerequisites and basics.
The first chapter offers an [Introduction to Deep Learning](../chapter_introduction/index.md).
In [Crashcourse](../chapter_crashcourse/index.md),
we'll quickly bring you up to speed on the prerequisites required for hands-on deep learning,
such as how to acquire and run the codes covered in the book.
[Deep Learning Basics](../chapter_deep-learning-basics/index.md)
covers the most basic concepts and techniques of deep learning,
such as multi-layer perceptrons and regularization.
<!--If you are short on time or you only want to learn only
about the most basic concepts and techniques of deep learning,
it is sufficient to read the first section only.-->
* The next three chapters focus on modern deep learning techniques.
[Deep Learning Computation](../chapter_deep-learning-computation/index.md)
describes the various key components of deep learning calculations
and lays the groundwork for the later implementation of more complex models.
Next we explain [Convolutional Neural Networks](../chapter_convolutional-neural-networks/index.md),
powerful tools that form the backbone of most modern computer vision systems in recent years.
Subsequently, we introduce [Recurrent Neural Networks](../chapter_recurrent-neural-networks/index.md),
models that exploit temporal or sequential structure in data,
and are commonly used for natural language processing and time series prediction.
These sections will get you up to speed on the basic tools behind most modern deep learning.

* Part three discusses scalability, efficiency and applications.
First we discuss several common [Optimization Algorithms](../chapter_optimization/index.md)
used to train deep learning models.
The next chapter, [Performance](../chapter_computational-performance/index.md),
examines several important factors that affect the computational performance of your deep learning code.
Chapters 9 and 10  illustrate major applications of deep learning
in computer vision and natural language processing, respectively.

An outline of the book together with possible flows for navigating it is given below.
The arrows provide a graph of prerequisites:

![Book structure](../img/book-org.svg)


### Code

Most sections of this book feature executable code.
We recognize the importance of an interactive learning experience in deep learning.
At present certain intuitions can only be developed through trial and error,
tweaking the code in small ways and observing the results.
Ideally, an elegant mathematical theory might tell us
precisely how to tweak our code to achieve a desired result.
Unfortunately, at present such elegant theories elude us.
Despite our best attempts, our explanations for of various techniques might be lacking,
sometimes on account of our shortcomings,
and equally often on account of the nascent state of the science of deep learning.
We are hopeful that as the theory of deep learning progresses,
future editions of this book will be able to provide insights in places the present edition cannot.

Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under MXNet 1.2.0.
However, due to the rapid development of deep learning,
some code *in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version remain up-to-date.
In case of such problems, please consult the section
["Installation and Running"](../chapter_prerequisite/install.md)
to update the code and runtime environment.
At times, to avoid unnecessary repetition,
we encapsulate the frequently-imported and referred-to functions, classes, etc.
in this book in the `gluonbook` package, version number 1.0.0.
We give a detailed overview of these functions and classes in the appendix [“gluonbook package index”](../chapter_appendix/gluonbook.md)


### Target Audience

This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we'll only assume that you come in with some basics,
including (the very basics of) linear algebra, calculus, probability,
and Python programming.
Moreover, this book's appendix provides a refresher
on most of the mathematics covered in this book.
Most of the time, we will prioritize intuition and ideas
over mathematical rigor.
There are many terrific books which can lead the interested reader further. For instance [Linear Analysis](https://www.amazon.com/Linear-Analysis-Introductory-Cambridge-Mathematical/dp/0521655773) by Bela Bollobas covers linear algebra and functional analysis in great depth. [All of Statistics](https://www.amazon.com/All-Statistics-Statistical-Inference-Springer/dp/0387402721) is a terrific guide to statistics.
And if you have not used Python before,
you may want to peruse the [Python tutorial](http://learnpython.org/).


### Forum

Associated with this book, we've launched a discussion forum,
located at [discuss.mxnet.io](https://discuss.mxnet.io/).
When you have questions on any section of the book,
you can find the associated discussion page by scanning the QR code
at the end of the section to participate in its discussions.
The authors of this book and broader MXNet developer community
frequently participate in forum discussions.


## 謝辞

われわれは、英語と日本語のドラフト版の作成に貢献した数百の人に恩があります。彼らは、内容を改善するための手助けをしてくれ、価値あるフィードバックを提供してくれました。特に、あらゆる人のために英語のドラフト版を改善した人々に感謝したいです。かれらのGithubのIDや名前を順不同で記載します。

alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, mohamed-ali,
mstewart141, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, vishwesh5, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, IgorDzreyev, trungha-ngx, pmuens, alukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, vfdev-5, bowen0701, arush15june, prasanth5reddy.

さらに、われわれはAmazon Wen Services、特に、Swami Sivasubramanian、Raju Gulabani、Charlie Bell、Andrew Jassyには、この書籍を執筆するための惜しみないサポートしてくれたことに感謝します。

費やした時間、リソース、同僚との議論、継続的な取り組みなくして、この書籍は生まれなかったでしょう。



## Summary

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.mxnet.io/.
* Apache MXNet is a powerful library for coding up deep learning models and running them in parallel across GPU cores.
* Gluon is a high level library that makes it easy to code up deep learning models using Apache MXNet.
* Conda is a Python package manager that ensures that all software dependencies are met.
* All notebooks are available for download on GitHub and  the conda configurations needed to run this book's code are expressed in the `environment.yml` file.
* If you plan to run this code on GPUs, don't forget to install the necessary drivers and update your configuration.


## Exercises

1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum,
where you'll be able to seek out help and discuss the book and find answers to your questions
by engaging the authors and broader community.
1. Create an account on the forum and introduce yourself.


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2311)

![](../img/qr_preface.svg)

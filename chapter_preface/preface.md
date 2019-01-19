# まえがき

以前は、大きな企業やスタートアップにおいて知的な製品やサービスを開発するような、深層学習の科学者のチームは存在しませんでした。われわれ著者のうち、最も若い世代がこの分野に入ったときも、日々の新聞で機械学習が新聞の見出しにでることはありませんでした。われわれの両親は、われわれが医薬や法律といった関係の職業よりも機械学習を好んでいることはもちろん、機械学習が何なのかということについても知りません。機械学習は、狭い領域における実世界の応用に向けた、先進的な学習的領域だったのです。例えば音声認識やコンピュータビジョンといった応用では、機械学習というのは非常に小さな構成要素の1つで、それとは別に、多くのドメイン知識を必要としたのでした。ニューラルネットワークは、われわれがこの本で着目する深層学習モデルの元になるものですが、時代遅れなツールだと考えられていました。

5年ほど前、深層学習はコンピュータビジョン、自然言語処理、自動音声認識、強化学習、統計的モデリングといった多様な分野において急速な進歩をみせ、世界を驚かせました。このような発展を手に、われわれは自動運転を開発したり（車の自動性を高めたり）、日常の応答を予測する賢い応答システムを開発したり、山のようなメールを探すことを助けたり、碁のようなボードゲームで世界のトッププレイヤーに打ち勝つソフトウェアエージェントを開発したり、数十年はかかると思われていたものを開発したりできるようになりました。すでに、これらのツールはそのインパクトを拡大し続けており、映画の作成方法を変え、病気の診断方法を変え、天体物理学から生物学に至るまでの基礎科学における役割を大きくしています。この本では、深層学習をより身近なものとするための試みを紹介し、読者に*概念*、*具体的な内容*、*コード*のすべてをお伝えします。


## コード、数学、HTMLを結びつける手段

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

われわれは、以下のことが可能なリソースの作成に着手しました。

(1) あらゆる人にとって無料で利用できる  
(2) 実際に機械学習の応用ができるサイエンティストになるための、起点となるような十分な技術的な深さを提供する  
(3) 読者にどうやっても問題を解くかを示すような実行可能なコードを含む  
(4) 我々や大部分はコミュニティによって、すばやくアップデートされる  
(5) 技術的な詳細についての対話的な議論や質問に回答するための[forum](http://discuss.mxnet.io)によって補足される  

これらのゴールはしばしば衝突を起こしていました。数式、理論、そして引用は、最善な形で管理されて、LaTeXによって組版されます。コードはPythonで最善の形で記述される。そして、ウェブページはもともとHTMLやJavaScriptによって成り立っている。さらに、実行可能なコードとして、物理的な書籍として、ダウンロード可能なPDFとして、インターネット上のWeb Siteとして、これらのいずれの場合でもアクセス可能なコンテンツを欲しいと考えていました。現在もこれらの要求に完全に応えられるツールやワークフローは存在しません。そこでわれわれは、われわれ自信でこれを構築する必要がありました。われわれは、このアプローチの詳細について [appendix](../chapter_appendix/how-to-contribute.md)に記載しています。そのソースの共有や編集の許可のためのGithub、コードと数式と文章を調和させるJupyter ノートブック、複数の出力(Webページ、PDFなど)を生成するためのSphinx、フォーラムのためのDisclosure、これらを利用することを決めました。われわれのシステムはまだ完全ではない一方で、これらの選択は互いに競合してきた関心事を互いに歩み寄らせているでしょう。この場がこのような統合的なワークフローを利用して最初に出版される本になるだろうと、われわれは信じています。

## Organization

Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background, each subsequent notebook introduces
both a reasonable number of new concepts and provides a single
self-contained working example -- using a real dataset.
This presents an organizational challenge. Some models might logically
be grouped together in a single notebook.  And some ideas might be
best taught by executing several models in succession.  On the other
hand, there's a big advantage to adhering to a policy of *1 working
example, 1 notebook*: This makes it as easy as possible for you to
start your own research projects by leveraging our code. Just copy a
single notebook and start modifying it.

We will interleave the runnable code with background material as needed.
In general, we will often err on the side of making tools
available before explaining them fully (and we will follow up by
explaining the background later).  For instance, we might use
*stochastic gradient descent* before fully explaining why it is useful
or why it works.  This helps to give practitioners the necessary
ammunition to solve problems quickly, at the expense of requiring the
reader to trust us with some decisions, at least in the short term.

Throughout, we'll be working with the MXNet library, which has the
rare property of being flexible enough for research while being fast
enough for production.  This book will teach deep learning concepts
from scratch.  Sometimes, we want to delve into fine details about the
models that are hidden from the user by ``Gluon``'s advanced features.
This comes up especially in the basic tutorials, where we want you to
understand everything that happens in a given layer.  In these cases,
we generally present two versions of the example: one where we
implement everything from scratch, relying only on NDArray and
automatic differentiation, and another where we show how to do things
succinctly with ``Gluon``.  Once we've taught you how a layer works,
we can just use the ``Gluon`` version in subsequent tutorials.

## Learning by Doing

Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook,
[Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738),
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.

In this book, we'll teach most concepts *just in time*.
In other words, you'll learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability.
We want you to taste the satisfaction of training your first model
before worrying about more exotic probability distributions.


## Acknowledgments

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft for making it better for everyone.
Their GitHub IDs or names are (in no particular order): alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat, cryptonaut, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, mohamed-ali, mstewart141, Mike Müller, NRauschmayr, prakhar1989, sad-, sfermigier, sundeepteki, topecongiro, vishaalkapoor, vishwesh5, YaYaB.
Moreover, we thank Amazon Web Services, especially Swami Sivasubramanian, Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book.
Without the available time, resources, discussions with colleagues,
and continuous encouragement this book would not have happened.

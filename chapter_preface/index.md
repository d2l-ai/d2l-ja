# まえがき

ほんの2、３年前は、大きな企業やスタートアップにおいて知的な製品やサービスを開発するような、深層学習の科学者のチームは存在しませんでした。われわれ著者のうち、最も若い世代がこの分野に入ったときも、日々の新聞で機械学習が新聞の見出しにでることはありませんでした。われわれの両親は、われわれが医薬や法律といった関係の職業よりも機械学習を好んでいることはもちろん、機械学習が何なのかということについても知りません。機械学習は、狭い領域における実世界の応用に向けた、先進的な学習的領域だったのです。例えば音声認識やコンピュータビジョンといった応用では、機械学習というのは非常に小さな構成要素の1つで、それとは別に、多くのドメイン知識を必要としたのでした。ニューラルネットワークは、われわれがこの本で着目する深層学習モデルの元になるものですが、時代遅れなツールだと考えられていました。

5年ほど前、深層学習はコンピュータビジョン、自然言語処理、自動音声認識、強化学習、統計的モデリングといった多様な分野において急速な進歩をみせ、世界を驚かせました。このような発展を手に、われわれは自動運転を開発したり（車の自動性を高めたり）、日常の応答を予測する賢い応答システムを開発したり、山のようなメールを探すことを助けたり、碁のようなボードゲームで世界のトッププレイヤーに打ち勝つソフトウェアエージェントを開発したり、数十年はかかると思われていたものを開発したりできるようになりました。すでに、これらのツールはそのインパクトを拡大し続けており、映画の作成方法を変え、病気の診断方法を変え、天体物理学から生物学に至るまでの基礎科学における役割を大きくしています。

## この書籍について
この本では、深層学習をより身近なものとするための試みを紹介し、読者に*概念*、*具体的な内容*、*コード*のすべてをお伝えします。


### コード、数学、HTMLを結びつける手段

どのようなコンピュータの技術もそのインパクトを十分に発揮するためには、十分に理解され、文書化され、成熟して十分に保守されたツールによって支援される必要があります。
キーとなるアイデアを明確な形で切り出して、新たに深層学習に取り組む人が最新の技術を身につけるための時間を最小化すべきです。成熟したライブラリは共通のタスクを自動化し、お手本となるコードは、取り組む人が必要とするものにあわせて、アプリケーションを簡単に修正、応用、拡張する手助けとなるべきです。動的なWebアプリケーションを例にあげましょう。
Amazonのような多くの企業が1990年代にデータベースを利用したWebアプリケーションの開発に成功しましたが、創造的な事業家を支援する潜在的な技術は、ここ10年で大きく実現されたものであり、それは強力で十分に文書化されたフレームワークの開発のおかげだったのです。

どんなアプリケーションも様々な学問によって成り立っているので、深層学習の潜在的な能力を試すことは、他にはない挑戦となるでしょう。深層学習を適用するためには以下を理解する必要があります。

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

多くの教科書は、一連のトピックについて網羅的に詳細を交えて説明します。例えば、 Chris Bishopの素晴らしい書籍 :cite:`Bishop.2006` は、各トピックを徹底的に説明し、線形回帰の章でもとてつもない量の内容を理解する必要があります。専門家は、その書籍の完全さゆえに非常に気に入っていますが、初心者にとっての導入の書籍としてはその有用性を活かしきれません。

この本では、たいていの考え方を *just in time*、つまり必要なときに教える方針です。言い換えれば、いくつかの実践的な目標を達成するために、ある考え方が必要になったときに、それを教えていきます。基礎となる予備知識、例えば、線形代数や確率といったものを伝える最初の段階では少し時間がかかると思います。風変わりな確率分布について心配する前に、最初のモデルを学習することに満足さを感じてほしいと思っています。

基礎的な数学の知識に関する集中講義を提供するための、一部の初歩的なノートブックは別として、それ以降のノートブックはほどよい量の新しいコンセプトを紹介し、全てがそろった単一で実行可能な例を実際のデータセットとともに提供します。これは、１つの構成上のチャレンジと呼んでいいと思います。いくつかのモデルは、単一のノートブックの中に論理的にひとまとめにすることもできるでしょう。そして、そのいくつかのモデルを連続的に実行することによって、考え方を教えたほうが良いかもしれません。一方、* 1つの実行可能な例と1つのノートブック* のポリシーに固執するのには大きな利点があります。われわれのコードを活用して、読者のみなさんの研究プロジェクトを、なるべく簡単に立ち上げるのを助けることができます。1つのノートブックをコピーしてから、それを修正していけばよいのです。

われわれは、必要に応じて実行可能なコードなかに、それを理解するための情報を含めます。一般的に、ツールを十分に説明する前に、そのツールを使えるようにしようとして失敗しすることもあるでしょう （そして、あとになってフォローアップの説明をします）。例えば、*stochastic gradient descent* について、それがなぜ有用で、なぜうまくいくのかを十分に説明する前に使うことがあると思います。この方法は、短時間のうちのいくつかの判断だけで、読者にわれわれを信用してもらうことになりますが、実際に機械学習を行う人に対して、問題をすばやく解くための武器を与えることができます。

ここでは最初から最後まで、MXNetのライブラリを利用して進める予定です。MXNetのライブラリは研究用途にも十分に柔軟で、本番環境の用途にも十分高速であるという得がたい特徴をもっています。深層学習の考え方についてゼロから伝えていく予定で、ときどき、``Gluon``の先進的な特徴によってユーザに隠蔽されたモデルについて、詳細な部分を掘り下げたいと思います。与えられた深層学習のレイヤーの中で起こっているすべてを理解してほしいので、特に基礎的なチュートリアルにおいて掘り下げを行います。この場合に、われわれは次の2種類の例を一般的に提示します。1つはNDArray(多次元配列）や自動微分を利用してゼロか全てを実装するもので、もう1つは``Gluon``によって同じことを簡潔に実装するものです。レイヤーがどのように動くかを伝えたら、以降のチュートリアルでは、``Gluon``を利用したものを使います。


## 内容と構成

この本は、おおまかに次の3つのパートにわかれています。

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* 最初のパートは前提条件や基礎の部分を扱います。:numref:`chap_introduction` では深層学習の紹介をします。そして、:numref:`chap_preliminaries` では、ハンズオン形式の深層学習に必要な準備について、例えば、データの保存・加工の方法や、線形代数、微分・積分、確率などの基本的な考え方にもとづいた、演算の適用方法などを紹介します。:numref:`chap_linear` と :numref:`chap_perceptrons` は、深層学習における最も基本的な考え方と技術、例えば、多層パーセプトロンや正則化について説明します。

* 次の5つの章は現代の深層学習の技術に焦点を当てています。:numref:`chap_computation` は、
深層学習における計算に関する重要な要素をいくつか説明し、後に複雑なモデルを実装するための土台を築きます。:numref:`chap_cnn` と :numref:`chap_modern_cnn` では、畳み込みニューラルネットワーク (Convolutional Neural Networks; CNNs)という、現在のコンピュータビジョの土台を築く強力なツールを紹介します。続いて、:numref:`chap_rnn` と :numref:`chap_modern_rnn`
では、再帰ニューラルネットワーク (Recurrent Neural Networks; RNNs) とよばれる、時系列・系列データを扱うモデルで、自然言語処理や時系列データ予測に利用されるモデルを紹介します。:numref:`chap_attention` では、Attention Mechanisms と呼ばれる技術を用いた新しいモデルについて紹介します。これは、自然言語処理における RNN に取って代わりつつあります。これらの章では、深層学習の最近の多くのアプリケーションを支える基本的なツールについて理解してもらうことを目的としています。


* パート3は、スケーラビリティ、効率性、アプリケーションについて考察します。まず、:numref:`chap_optimization` では、深層学習モデルの学習に利用される、いくつかの最適化アルゴリズムについて考察します。次の章である :numref:`chap_performance` では、深層学習の性能に影響する重要な要素について調査します。 :numref:`chap_cv` と :numref:`chap_nlp` では、コンピュータビジョンと自然言語処理のそれぞれにおける、深層学習の主要なアプリケーションを説明します。

## コード
:label:`sec_code`

この本のほとんどの章では、深層学習におけるインタラクティブな学習体験が重要であることを信じて、実行可能コードを取り上げています。そして、コードを細かく調整し、結果を観察するという試行錯誤を通して確かな感覚を開発することができます。理想的には、洗練された数学的理論が、コードを微調整して目的の結果を達成するための方法を、正確に教えてくれる可能性がなくはありません。残念ながら、現在、そのような洗練された理論は私たちにはありません。私たちがここで最善を尽くしたとしても、これらのモデルを特徴付ける数学は非常に難しく、これらのトピックに関する重要な調査が最近始まったばかりであるため、さまざまな手法の正式な説明はまだ不十分なままです。
深層学習の理論が進むにつれて、現在の版では提供できない洞察を、将来の版で提供できるようになることを期待しています。


この書籍のたいていのコードはApache MXNetを利用しています。MXNetは、AWS (Amazon Web Services)が選んだ、深層学習のためのオープンソースフレームワークで、多くの大学と企業で利用されています。この書籍の全てのコードはMXNet1.2.0にもとづくテストをパスしています。しかし、深層学習の急速な発展にともなって、*印刷版*のコードは、MXNetの将来のバージョンでは動かない可能性があります。一方、オンライン版は常に最新版に維持されるようにします。もしなにか問題が生じたら、:ref:`chap_installation` を見て、コードや実行環境をアップデートしまししょう。

現在、不必要な繰り返し作業を避けるため、この書籍で頻繁にimportされたり、参照されたりする関数、クラス、その他については、`d2l`のパッケージにカプセル化します。パッケージに保存されている関数、クラス、複数のインポートなどのブロックは、`# Saved in the d2l package for later use`といった注記をします。`d2l`のパッケージは軽量で、次のパッケージやモジュールのみを依存関係として必要とします。

```{.python .input  n=1}
# Saved in the d2l package for later use
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
import os
import pandas as pd
import random
import re
import sys
import tarfile
import time
import zipfile
```

これらの関数やクラスについての詳細は :numref:`sec_d2l` で説明します。


## 対象とする読者

この本は、深層学習の実践的な技術を学びたい大学生 (学部生、大学院生)、エンジニア、研究者を対象にしています。
著者らはゼロからすべての考え方を説明するつもりですので、深層学習や機械学習に関する事前知識をもっている必要はありません。
深層学習のモデルを完全に説明するためには、いくらかの数学やプログラミングが必要ですが、(非常に基本的な)線形代数、微積分、確率、pythonのプログラミングを含む基礎を理解できるようになれば良いと考えています。そして、Appendix では、この書籍でカバーされる数学を再思い出すための内容を提供します。この書籍では、多くの場合、数学的な厳密さにもとづく直感的な理解や考え方を重視します。興味がある読者がさらに深く学ぶためのすごい書籍はたくさんあります。例えば、Bela Bollobas の Linear Analysis :cite:`Bollobas.1999` 非常に深い線形代数や関数解析に関してカバーしています。All of Statistics :cite:`Wasserman.2013` は統計に関する素晴らしいガイドです。もし Python を以前に利用したことがなければ、[Python tutorial](http://learnpython.org/) を追ってみるのも良いかもしれません。

### フォーラム

この書籍に関連して、[discuss.mxnet.io](https://discuss.mxnet.io/)に議論のためのフォーラムを立ち上げています。この書籍のどの部分でも、もし質問があれば、各節の最後にあるQRコードをスキャンして関連する議論のページにたどり着き、議論に参加することができます。この書籍の著者と、幅広いMXNet開発者コミュニティは、このフォーラムでの議論によく参加しています。


## 謝辞

われわれは、英語と日本語のドラフト版の作成に貢献した数百の人に恩があります。彼らは、内容を改善するための手助けをしてくれ、価値あるフィードバックを提供してくれました。特に、あらゆる人のために英語のドラフト版を改善した人々に感謝したいです。かれらのGithubのIDや名前を順不同で記載します。

alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, mohamed-ali,
mstewart141, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, vishwesh5, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, IgorDzreyev, trungha-ngx, pmuens, alukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, vfdev-5, bowen0701, arush15june, prasanth5reddy.

さらに、われわれはAmazon Wen Services、特に、Swami Sivasubramanian、Raju Gulabani、Charlie Bell、Andrew Jassyには、この書籍を執筆するための惜しみないサポートしてくれたことに感謝します。

費やした時間、リソース、同僚との議論、継続的な取り組みなくして、この書籍は生まれなかったでしょう。



## まとめ

* 深層学習は、パターン認識を革新し、コンピュータビジョン、自然言語処理、自動音声認識などの幅広い分野に力を与える技術を導入しました。
* うまく深層学習を適用するためには、問題を明らかにする方法、数学的なモデリング、モデルをデータに当てはめるためのアルゴリズム、それらすべてを実装する工学的な技術が必要です。
* この書籍は、文章、図表、数学、コード、これらすべてを1箇所にまとめた、包括的なリソースを提供します。
* この書籍に関する質問に回答するために、https://discuss.mxnet.io/ のフォーラムに参加してください。
* Apache MXNetは深層学習モデルのコードを実装し、複数のGPUコアで並列に実行するための強力なライブラリです。
* Gluonという高レベルなライブラリを利用することで、Apache MXNetを利用した深層学習モデルの実装を簡単に行うことができます。
* Condaは、すべてのソフトウェアの依存関係が満たされることを保証するPythonのパッケージ管理ツールです。
* すべてのノートブックはGithub上でダウンロードすることが可能で、この書籍のコードを実行するために必要なcondaの設定ファイルは`environment.yml`のファイルの中に記述されています。
* もしここでのコードをGPU上で実行しようと考えているのであれば、必要なドライバをインストールして、構成を更新することを忘れないようにしましょう。

## 練習

1. この書籍のフォーラム[discuss.mxnet.io](https://discuss.mxnet.io/)でアカウントを登録しましょう。
1. コンピュータにPythonをインストールしましょう。
1. 各セクションの最後にあるフォーラムへのリンクをたどりましょう。フォーラムでは、著者や広いコミュニティに関わることで、助けを求めたり、書籍の内容を議論したり、疑問に対する答えを探すことができるでしょう。
1. フォーラムのアカウントを作成して自己紹介をしましょう。

## [議論](https://discuss.mxnet.io/t/2311)のためのQRコードをスキャン

![](../img/qr_preface.svg)

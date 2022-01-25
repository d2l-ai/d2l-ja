# 自然言語処理:アプリケーション
:label:`chap_nlp_app`

:numref:`chap_nlp_pretrain` では、テキストシーケンスでトークンを表現し、その表現をトレーニングする方法を見てきました。このような事前学習済みテキスト表現は、さまざまなダウンストリーム自然言語処理タスクのためにさまざまなモデルに供給できます。 

実際、以前の章ではすでにいくつかの自然言語処理アプリケーションについて説明してきました。
*事前トレーニングなし*
ディープラーニングのアーキテクチャを説明するためだけに。例えば :numref:`chap_rnn` では、小説のようなテキストを生成する言語モデルの設計に RNN を頼りにしてきました。:numref:`chap_modern_rnn` と :numref:`chap_attention` では、機械翻訳用のRNNとアテンションメカニズムに基づくモデルも設計しました。 

ただし、本書は、そのようなアプリケーションのすべてを包括的に網羅することを意図したものではありません。その代わり、*言語の (深い) 表現学習を自然言語処理問題への対処にどのように適用するか*に焦点を当てています。この章では、事前学習済みのテキスト表現を考慮して、一般的で代表的なダウンストリーム自然言語処理タスクの 2 つについて説明します。センチメント分析と自然言語推論は、それぞれ単一テキストとテキストペアの関係を分析します。 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

:numref:`fig_nlp-map-app` に示されているように、この章では、MLP、CNN、RNN、アテンションなど、さまざまなタイプのディープラーニングアーキテクチャを使用して自然言語処理モデルを設計する際の基本的な考え方について説明します。:numref:`fig_nlp-map-app` では、事前学習済みのテキスト表現をどのアーキテクチャにも組み合わせることができますが、:numref:`fig_nlp-map-app` ではいくつかの代表的な組み合わせを選択します。具体的には、センチメント分析のためにRNNとCNNをベースにした一般的なアーキテクチャを探ります。自然言語推論では、テキストペアの分析方法を示すために、アテンションとMLPを選択します。最後に、シーケンスレベル (単一テキスト分類とテキストペア分類) やトークンレベル (テキストタグ付けと質問応答) など、さまざまな自然言語処理アプリケーション向けに事前学習済みの BERT モデルを微調整する方法を紹介します。具体的な経験的ケースとして、自然言語推論のためにBERTを微調整する。 

:numref:`sec_bert` で紹介したように、BERT は幅広い自然言語処理アプリケーションに対して最小限のアーキテクチャ変更しか必要としません。ただし、この利点には、ダウンストリームアプリケーション用に膨大な数の BERT パラメータを微調整するという代償が伴います。空間や時間が限られている場合、MLP、CNN、RNN、およびアテンションに基づいて作成されたモデルの方がより実現可能です。以下では、センチメント分析アプリケーションから始めて、RNN と CNN をそれぞれベースにしたモデル設計を説明します。

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

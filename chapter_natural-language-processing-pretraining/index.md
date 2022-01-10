# 自然言語処理:事前訓練
:label:`chap_nlp_pretrain`

人間はコミュニケーションをとる必要があります。人間の状態に対するこの基本的な必要性から、膨大な量の文章が日常的に生成されてきました。ソーシャルメディア、チャットアプリ、電子メール、製品レビュー、ニュース記事、研究論文、書籍にリッチテキストが含まれる場合、コンピューターがそれらを理解して支援を提供したり、人間の言語に基づいた意思決定を行ったりできるようにすることが不可欠になります。 

*自然言語処理*は、コンピュータと人間の相互作用を自然言語を用いて研究する。
実際には、:numref:`sec_language_model` の言語モデルや :numref:`sec_machine_translation` の機械翻訳モデルなど、テキスト (人間の自然言語) データを処理および分析するために、自然言語処理技術を使用することが非常に一般的です。 

テキストを理解するために、その表現を学ぶことから始めることができます。大規模コーパスの既存のテキストシーケンスを活用し、
*自己教師あり学習*
は、周囲のテキストの他の部分を使用してテキストの隠れた部分を予測するなど、テキスト表現の事前トレーニングに広く使用されています。このようにして、モデルは*高額*のラベリング作業なしに、*膨大な*テキストデータから監視を通じて学習します。 

この章で説明するように、各単語またはサブワードを個別のトークンとして扱う場合、各トークンの表現は、word2vec、GLOVE、または大きなコーパス上のサブワード埋め込みモデルを使用して事前学習できます。事前トレーニング後、各トークンの表現はベクトルになる可能性がありますが、コンテキストが何であれ同じままです。例えば、「銀行」のベクトル表現は、「銀行に行ってお金を預ける」と「銀行に行って座る」の両方で同じです。したがって、最近の多くの事前トレーニングモデルでは、同じトークンの表現を異なるコンテキストに適応させています。その中には、トランスエンコーダをベースにしたより深い自己教師ありモデルであるBERTがあります。この章では、:numref:`fig_nlp-map-pretrain` で強調されているように、このようなテキスト表現を事前にトレーニングする方法に焦点を当てます。 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on the upstream text representation pretraining.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`

全体像を見ると、:numref:`fig_nlp-map-pretrain` は、事前学習済みのテキスト表現を、さまざまなダウンストリーム自然言語処理アプリケーション用のさまざまな深層学習アーキテクチャに供給できることを示しています。:numref:`chap_nlp_app` でそれらをカバーします。

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining
```

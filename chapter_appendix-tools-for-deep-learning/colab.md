# グーグル・コラボレーションを使う
:label:`sec_colab`

:numref:`sec_sagemaker` と :numref:`sec_aws` で AWS でこの本を実行する方法を紹介しました。もう 1 つの選択肢として、この本を [Google Colab](https://colab.research.google.com/) で実行する方法があります。Google アカウントをお持ちの場合は、無料の GPU が提供されます。 

Colab でセクションを実行するには、:numref:`fig_colab` のように、そのセクションのタイトルの右側にある `Colab` ボタンをクリックするだけです。  

![Open a section on Colab](../img/colab.png)
:width:`300px`
:label:`fig_colab`

コードセルを初めて実行すると、:numref:`fig_colab2` に示すような警告メッセージが表示されます。「RUN ANYWAY」をクリックして無視してもかまいません。 

![The warning message for running a section on Colab](../img/colab-2.png)
:width:`300px`
:label:`fig_colab2`

次に、Colab がこのノートブックを実行するインスタンスに接続します。具体的には、`d2l.try_gpu()` 関数を呼び出すときなど、GPU が必要な場合、GPU インスタンスに自動的に接続するように Colab にリクエストします。 

## [概要

* Google Colab を使用して、この本の各セクションを GPU で実行できます。

## 演習

1. Google Colab を使用して、この本のコードを編集して実行してみてください。

[Discussions](https://discuss.d2l.ai/t/424)

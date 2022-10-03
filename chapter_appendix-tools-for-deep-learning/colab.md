# グーグル・コラボを使う
:label:`sec_colab`

:numref:`sec_sagemaker` と :numref:`sec_aws` で AWS でこの本を実行する方法を紹介しました。別のオプションは、Googleアカウントを持っている場合、この本を[Google Colab](https://colab.research.google.com/)で実行することです。 

Colabでセクションのコードを実行するには、:numref:`fig_colab`に示すように、`Colab`ボタンをクリックします。  

![Run the code of a section on Colab](../img/colab.png)
:width:`300px`
:label:`fig_colab`

コードセルを初めて実行する場合は、:numref:`fig_colab2`に示すような警告メッセージが表示されます。無視するには、「実行する」をクリックするだけです。 

![Ignore the warning message by clicking "RUN ANYWAY".](../img/colab-2.png)
:width:`300px`
:label:`fig_colab2`

次に、Colab は、このセクションのコードを実行するインスタンスに接続します。具体的には、GPUが必要な場合、ColabはGPUインスタンスへの接続を自動的に要求されます。 

## まとめ

* Google Colab を使用して、この本の各セクションのコードを実行できます。
* 本書のいずれかのセクションでGPUが必要な場合、ColabはGPUインスタンスへの接続を要求されます。

## 演習

1. Google Colab を使用して、この本の任意のセクションを開きます。
1. Google Colab を使用して GPU を必要とするセクションを編集して実行します。

[Discussions](https://discuss.d2l.ai/t/424)

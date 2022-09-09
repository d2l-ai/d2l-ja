# Amazon SageMaker を使う
:label:`sec_sagemaker`

ディープラーニングアプリケーションは、ローカルマシンが提供できるものを簡単に超えるほど多くの計算リソースを必要とする場合があります。クラウドコンピューティングサービスを使用すると、より強力なコンピューターを使用して、この本のGPU集約型コードをより簡単に実行できます。このセクションでは、Amazon SageMaker を使用してこの本のコードを実行する方法を紹介します。 

## サインアップ

まず、https://aws.amazon.com/ でアカウントをサインアップする必要があります。セキュリティを強化するため、二要素認証の使用が推奨されます。また、インスタンスの実行を停止し忘れた場合など、予期せぬ事態を避けるために、請求と支出の詳細なアラートを設定することもお勧めします。AWS アカウントにログインした後、[console](http://console.aws.amazon.com/) に移動して「Amazon SageMaker」(:numref:`fig_sagemaker` を参照) を検索し、それをクリックして SageMaker パネルを開きます。 

![Search for and open the SageMaker panel.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## SageMaker インスタンスを作成する

次に、:numref:`fig_sagemaker-create` の説明に従ってノートブックインスタンスを作成しましょう。 

![Create a SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker は、さまざまな計算能力と価格で複数の [instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/) を提供しています。ノートブックインスタンスを作成するときに、その名前とタイプを指定できます。:numref:`fig_sagemaker-create-2`では、`ml.p3.2xlarge`を選択しました。1つのTesla V100 GPUと8コアCPUを備えたこのインスタンスは、本のほとんどで十分に強力です。 

![Choose the instance type.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
SageMaker で実行するための ipynb フォーマットの本全体は https://github.com/d2l-ai/d2l-en-sagemaker. We can specify this GitHub repository URL (:numref:`fig_sagemaker-create-3` で入手できます。これにより、SageMaker はインスタンスの作成時にクローンを作成できます。
:end_tab:

:begin_tab:`pytorch`
SageMaker で実行するための ipynb フォーマットの本全体は https://github.com/d2l-ai/d2l-pytorch-sagemaker. We can specify this GitHub repository URL (:numref:`fig_sagemaker-create-3` で入手できます。これにより、SageMaker はインスタンスの作成時にクローンを作成できます。
:end_tab:

:begin_tab:`tensorflow`
SageMaker で実行するための ipynb フォーマットの本全体は https://github.com/d2l-ai/d2l-tensorflow-sagemaker. We can specify this GitHub repository URL (:numref:`fig_sagemaker-create-3` で入手できます。これにより、SageMaker はインスタンスの作成時にクローンを作成できます。
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## インスタンスの実行と停止

インスタンスの作成には数分かかる場合があります。インスタンスの準備ができたら、その横にある「Open Jupyter」リンク（:numref:`fig_sagemaker-open`）をクリックして、このインスタンスでこの本のすべてのJupyterノートブックを編集して実行できるようにします（:numref:`sec_jupyter`の手順と同様）。 

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

作業が終了したら、それ以上課金されないようにインスタンスを停止することを忘れないでください (:numref:`fig_sagemaker-stop`)。 

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## ノートブックの更新

:begin_tab:`mxnet`
このオープンソースブックのノートブックは、GitHubの[d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker)リポジトリで定期的に更新されます。最新バージョンに更新するには、SageMaker インスタンス (:numref:`fig_sagemaker-terminal`) でターミナルを開きます。
:end_tab:

:begin_tab:`pytorch`
このオープンソースブックのノートブックは、GitHubの[d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker)リポジトリで定期的に更新されます。最新バージョンに更新するには、SageMaker インスタンス (:numref:`fig_sagemaker-terminal`) でターミナルを開きます。
:end_tab:

:begin_tab:`tensorflow`
このオープンソースブックのノートブックは、GitHubの[d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker)リポジトリで定期的に更新されます。最新バージョンに更新するには、SageMaker インスタンス (:numref:`fig_sagemaker-terminal`) でターミナルを開きます。
:end_tab:

![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

リモートリポジトリから更新をプルする前に、ローカルの変更をコミットしたい場合があります。それ以外の場合は、ターミナルで次のコマンドを実行して、ローカルの変更をすべて破棄します。

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## まとめ

* Amazon SageMaker を使用してノートブックインスタンスを作成し、この本の GPU 集中型コードを実行できます。
* Amazon SageMaker インスタンスのターミナルからノートブックを更新できます。

## 演習

1. Amazon SageMaker を使用して GPU を必要とするセクションを編集して実行します。
1. ターミナルを開いて、この本のすべてのノートブックをホストするローカルディレクトリにアクセスします。

[Discussions](https://discuss.d2l.ai/t/422)

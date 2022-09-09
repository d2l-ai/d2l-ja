# AWS EC2 インスタンスの使用
:label:`sec_aws`

このセクションでは、すべてのライブラリを raw Linux マシンにインストールする方法を説明します。:numref:`sec_sagemaker` で Amazon SageMaker の使用方法について説明しましたが、AWS では自分でインスタンスを構築するほうがコストが安くなることを思い出してください。このチュートリアルには、次の 3 つの手順が含まれます。 

1. AWS EC2 から GPU Linux インスタンスをリクエストします。
1. CUDA をインストールします (または CUDA がプリインストールされた Amazon マシンイメージを使用します)。
1. 本のコードを実行するためのディープラーニングフレームワークとその他のライブラリをインストールします。

このプロセスは、多少の変更はありますが、他のインスタンス (および他のクラウド) にも適用されます。先に進む前に、AWS アカウントを作成する必要があります。詳細については :numref:`sec_sagemaker` を参照してください。 

## EC2 インスタンスの作成と実行

AWS アカウントにログインした後、「EC2」(:numref:`fig_aws` の赤いボックスでマーク) をクリックして EC2 パネルに移動します。 

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` は、機密性の高いアカウント情報がグレー表示された EC2 パネルを示しています。 

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### ロケーションの事前設定レイテンシを減らすために、近くのデータセンターを選択します。例:「オレゴン」(:numref:`fig_ec2`の右上にある赤いボックスでマーク)。中国にお住まいの場合は、ソウルや東京など、近くのアジア太平洋地域を選択できます。一部のデータセンターには GPU インスタンスがない場合があることに注意してください。 

### 上限を増やす

インスタンスを選択する前に、:numref:`fig_ec2`に示すように、左側のバーの「Limits」ラベルをクリックして、数量制限があるかどうかを確認してください。:numref:`fig_limits`はそのような制限の例を示しています。アカウントは現在、リージョンごとに「p2.xlarge」インスタンスを開くことができません。1 つ以上のインスタンスを開く必要がある場合は、[Request limit increase] リンクをクリックして、より高いインスタンスクォータを申請します。通常、申請の処理には1営業日かかります。 

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### インスタンスを起動する

次に、:numref:`fig_ec2` の赤いボックスでマークされている [Launch Instance] ボタンをクリックして、インスタンスを起動します。 

まず、適切な Amazon マシンイメージ (AMI) を選択します。検索ボックスに「Ubuntu」と入力します（:numref:`fig_ubuntu`の赤いボックスでマークされています）。 

![Choose an AMI.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 には、選択できるさまざまなインスタンス構成が用意されています。これは初心者には圧倒されることがあります。:numref:`tab_ec2`には、さまざまな適切なマシンがリストされています。 

:さまざまな EC2 インスタンスタイプ 

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |
:label:`tab_ec2`

これらのサーバーはすべて、使用されているGPUの数を示す複数の種類があります。たとえば、p2.xlarge には 1 GPU があり、p2.16xlarge には 16 GPU とより多くのメモリがあります。詳細については、[AWS EC2 documentation](https732293614) を参照してください。 

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

適切なドライバーと GPU 対応のディープラーニングフレームワークを備えた GPU 対応インスタンスを使用する必要があります。そうしないと、GPU を使用しても何のメリットも得られません。 

ここまで、:numref:`fig_disk` の上部に示されているように、EC2 インスタンスを起動するための 7 つのステップのうち最初の 2 つを完了しました。この例では、ステップ「3.インスタンスの設定」、「5.タグを追加」と「6.セキュリティグループの設定」。「4.ストレージを追加」をクリックし、デフォルトのハードディスクサイズを64 GB（:numref:`fig_disk`の赤いボックスでマーク）に増やします。CUDA自体はすでに4 GBを占めていることに注意してください。 

![Modify the hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

最後に、「7.Review」をクリックし、「Launch」をクリックして、設定したインスタンスを起動します。これで、インスタンスへのアクセスに使用するキーペアを選択するように求められます。キーペアがない場合は、:numref:`fig_keypair`の最初のドロップダウンメニューで [Create a new key pair] を選択してキーペアを生成します。その後、このメニューで「既存のキーペアを選択」を選択し、以前に生成したキーペアを選択できます。「Launch Instances」をクリックして、作成したインスタンスを起動します。 

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

新しいキーペアを生成した場合は、必ずキーペアをダウンロードして安全な場所に保存してください。これは、サーバーに SSH 接続する唯一の方法です。:numref:`fig_launching`に表示されているインスタンスIDをクリックして、このインスタンスのステータスを表示します。 

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### インスタンスに接続する

:numref:`fig_connect`に示すように、インスタンスの状態が緑色に変わったら、インスタンスを右クリックして `Connect` を選択し、インスタンスのアクセス方法を表示します。 

![View instance access method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

これが新しい鍵である場合、SSH が機能するために公開されてはいけません。`D2L_key.pem` を格納するフォルダに移動し、次のコマンドを実行してキーを公開しないようにします。

```bash
chmod 400 D2L_key.pem
```

![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

次に、:numref:`fig_chmod`の下の赤いボックスにsshコマンドをコピーし、コマンドラインに貼り付けます。

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

コマンドラインに「接続を続けますか (はい/いいえ)」というプロンプトが表示されたら、「はい」と入力して Enter キーを押し、インスタンスにログインします。 

これでサーバーの準備が整いました。 

## CUDA のインストール

CUDA をインストールする前に、必ず最新のドライバーでインスタンスを更新してください。

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

ここでは CUDA 10.1 をダウンロードします。NVIDIA の [公式リポジトリ](https://developer.nvidia.com/cuda-toolkit-archive) to find the download link as shown in :numref:`fig_cuda`) にアクセスしてください。 

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

説明をコピーして端末に貼り付け、CUDA 10.1 をインストールします。

```bash
# The link and file name are subject to changes
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

プログラムのインストール後、次のコマンドを実行して GPU を表示します。

```bash
nvidia-smi
```

最後に、CUDA をライブラリパスに追加して、他のライブラリが見つけやすくします。

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## コードを実行するためのライブラリのインストール

この本のコードを実行するには、EC2 インスタンスで Linux ユーザー向け :ref:`chap_installation` の手順を実行し、リモート Linux サーバーでの作業に関する次のヒントを使用します。 

* Minicondaのインストールページでbashスクリプトをダウンロードするには、ダウンロードリンクを右クリックして「リンクアドレスをコピー」を選択し、`wget [copied link address]`を実行します。
* 現在のシェルを閉じて再度開く代わりに `~/miniconda3/bin/conda init`, you may execute `source ~/.bashrc` を実行した後。

## Jupyter ノートブックをリモートで実行する

Jupyter Notebook をリモートで実行するには、SSH ポート転送を使用する必要があります。結局のところ、クラウド内のサーバーにはモニターやキーボードがありません。そのためには、次のようにデスクトップ (またはラップトップ) からサーバーにログインします。

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
```

次に、EC2 インスタンス上のこの本のダウンロード済みコードの場所に移動して、以下を実行します。

```
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter`は、Jupyter Notebookを実行した後の出力を示しています。最後の行はポート 8888 の URL です。 

![Output after running the Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

ポート 8889 へのポート転送を使用したので、:numref:`fig_jupyter` の赤いボックスの最後の行をコピーし、URL の「8888」を「8889」に置き換えて、ローカルブラウザで開きます。 

## 未使用のインスタンスを閉じる

クラウドサービスは使用時間によって請求されるため、使用されていないインスタンスを閉じる必要があります。代替案があることに注意してください。 

* インスタンスを「停止」すると、再び起動できるようになります。これは、通常のサーバーの電源を切るようなものです。ただし、停止したインスタンスには、保持されているハードディスク容量に対して少額の料金が請求されます。 
* インスタンスを「終了」すると、そのインスタンスに関連付けられているすべてのデータが削除されます。これにはディスクも含まれるため、再度起動することはできません。これは、将来必要ないことがわかっている場合にのみ行ってください。

インスタンスをさらに多くのインスタンスのテンプレートとして使用する場合は、:numref:`fig_connect`の例を右クリックし、「Image」$\rightarrow$「Create」を選択してインスタンスのイメージを作成します。これが完了したら、「インスタンスの状態」$\rightarrow$「終了」を選択してインスタンスを終了します。次回このインスタンスを使用するときは、このセクションの手順に従って、保存したイメージに基づいてインスタンスを作成できます。唯一の違いは、「1.:numref:`fig_ubuntu` に表示されている「AMI」を選択します。保存した画像を選択するには、左側の「My AMI」オプションを使用する必要があります。作成されたインスタンスは、イメージハードディスクに保存された情報を保持します。たとえば、CUDA やその他のランタイム環境を再インストールする必要はありません。 

## まとめ

* 独自のコンピューターを購入して構築しなくても、オンデマンドでインスタンスを起動および停止できます。
* GPU 対応のディープラーニングフレームワークを使用する前に CUDA をインストールする必要があります。
* ポート転送を使用して、Jupyter Notebook をリモートサーバーで実行できます。

## 演習

1. クラウドは便利ですが、安くはありません。[spot instances](https://aws.amazon.com/ec2/spot/)の起動方法を確認して、コストを削減する方法をご覧ください。
1. さまざまな GPU サーバーを試してみてください。彼らはどれくらい速いですか？
1. マルチ GPU サーバーを試してみてください。物事をどれだけうまくスケールアップできますか？

[Discussions](https://discuss.d2l.ai/t/423)

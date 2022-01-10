# 取り付け
:label:`chap_installation`

実践的な学習経験を得るために、Python、Jupyter ノートブック、関連ライブラリ、およびブック自体を実行するために必要なコードを実行するための環境をセットアップする必要があります。 

## Miniconda をインストールする

もっとも簡単な方法は [Miniconda](https://conda.io/en/latest/miniconda.html) をインストールすることです。Python 3.x バージョンが必要です。マシンに既に conda がインストールされている場合は、次の手順を省略できます。 

Miniconda の Web サイトにアクセスして、お使いの Python 3.x のバージョンとマシンアーキテクチャに基づいて、ご使用のシステムに適したバージョンを判断してください。たとえば、macOS と Python 3.x を使用している場合、名前に「Miniconda3」と「macOSX」という文字列が含まれる bash スクリプトをダウンロードし、ダウンロード場所に移動して、次のようにインストールを実行します。

```bash
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

Python 3.x を使用している Linux ユーザは、名前に「Miniconda3」と「Linux」という文字列を含むファイルをダウンロードし、ダウンロード先で以下を実行します。

```bash
sh Miniconda3-latest-Linux-x86_64.sh -b
```

次に、`conda` を直接実行できるように、シェルを初期化します。

```bash
~/miniconda3/bin/conda init
```

ここで、現在のシェルを閉じてから再度開きます。新しい環境は次のように作成できるはずです。

```bash
conda create --name d2l python=3.8 -y
```

## D2L ノートブックのダウンロード

次に、この本のコードをダウンロードする必要があります。HTML ページの上部にある [すべてのノートブック] タブをクリックすると、コードをダウンロードして解凍できます。または、`unzip` (それ以外の場合は `sudo apt install unzip` を実行) を使用できる場合は、次のようにします。

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

これで `d2l` 環境をアクティブ化できます。

```bash
conda activate d2l
```

## フレームワークと `d2l` パッケージのインストール

ディープラーニングフレームワークをインストールする前に、まずマシンに適切な GPU が搭載されているかどうかを確認してください (標準的なラップトップのディスプレイに電力を供給する GPU は、この目的に適していません)。GPU サーバーで作業している場合は、:ref:`subsec_gpu` に進み、関連ライブラリの GPU 対応バージョンをインストールする手順を確認してください。 

マシンにGPUが搭載されていない場合でも、まだ心配する必要はありません。CPUは、最初の数章を完了するのに十分な馬力を提供します。大きなモデルを実行する前に GPU にアクセスする必要があることを覚えておいてください。CPU バージョンをインストールするには、以下のコマンドを実行します。

:begin_tab:`mxnet`
```bash
pip install mxnet==1.7.0.post1
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch torchvision
```
:end_tab:

:begin_tab:`tensorflow`
CPU と GPU の両方をサポートする TensorFlow は、次のようにしてインストールできます。

```bash
pip install tensorflow tensorflow-probability
```
:end_tab:

次のステップは、本書でよく使われる関数とクラスをカプセル化するために開発した `d2l` パッケージをインストールすることです。

```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```

これらのインストール手順が完了したら、以下を実行して Jupyter ノートブックサーバーを起動できます。

```bash
jupyter notebook
```

この時点で、お使いの Web ブラウザで http://localhost:8888 (既に自動的に開かれている場合もあります) を開くことができます。その後、本の各セクションのコードを実行できます。本のコードを実行したり、ディープラーニングフレームワークや `d2l` パッケージを更新する前に、必ず `conda activate d2l` を実行してランタイム環境をアクティブ化してください。環境を終了するには、`conda deactivate` を実行します。 

## GPU サポート
:label:`subsec_gpu`

:begin_tab:`mxnet`
既定では、MXNet は GPU をサポートせずにインストールされ、どのコンピューター (ほとんどのラップトップを含む) でも確実に実行できます。本書の一部では、GPU での実行が必須または推奨されています。コンピュータに NVIDIA グラフィックカードが搭載されていて [CUDA](https://developer.nvidia.com/cuda-downloads) がインストールされている場合は、GPU 対応バージョンをインストールする必要があります。CPU のみのバージョンをインストールしている場合は、まず次のコマンドを実行して削除する必要があります。

```bash
pip uninstall mxnet
```

ここで、インストールした CUDA のバージョンを調べる必要があります。これを確認するには、`nvcc --version` または `cat /usr/local/cuda/version.txt` を実行します。CUDA 10.1 をインストールしたと仮定し、次のコマンドでインストールできます。

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```

最後の数字は、CUDA のバージョンに応じて変更できます。たとえば、CUDA 10.0 の場合は `cu100`、CUDA 9.0 の場合は `cu90` などです。
:end_tab:

:begin_tab:`pytorch,tensorflow`
既定では、ディープラーニングフレームワークは GPU サポート付きでインストールされます。コンピュータに NVIDIA GPU が搭載され、[CUDA](https://developer.nvidia.com/cuda-downloads) がインストールされていれば、準備は完了です。
:end_tab:

## 演習

1. 本のコードをダウンロードし、ランタイム環境をインストールします。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:

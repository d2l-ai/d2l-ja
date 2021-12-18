# インストール
:label:`chap_installation`

ハンズオンを始めるために、Python、Jupyter ノートブック、関連するライブラリ、この書籍を実行するためのコードをセットアップする必要があります。


## Miniconda のインストール

最もシンプルに始める方法は[Miniconda](https://conda.io/en/latest/miniconda.html)をインストールすることです。。Python 3.X のバージョンが推奨です。もし conda がすでにインストールされていれば以下のステップをスキップすることができます。

Visit the Miniconda website and determine the appropriate version for your system based on your Python 3.x version and machine architecture. For example, if you are using macOS and Python 3.x you would download the bash script whose name contains the strings "Miniconda3" and "MacOSX", navigate to the download location, and execute the installation as follows:

Miniconda のウェブサイトに移動し、Python 3.x のバージョンや、お使いの計算機のアーキテクチャに応じて、適切なバージョンを決定してください。例えば、macOS で Python 3.x を利用している場合は、Miniconda3やMacOSXといった文字列を含む bash スクリプトをダウンロードします。ダウンロードしたフォルダに移動し、以下の通りインストールを実行します。

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

そして Python 3.X を利用している Linux ユーザは、Miniconda3やLinuxといった文字列を含むファイルをダウンロードし、ダウンロードしたフォルダに移動して以下を実行します。

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

次に、シェルを初期化するために、直接 `conda` を実行します。

```bash
~/miniconda3/bin/conda init
```

現在のシェルを閉じて再度開いてください。以下のように新しい環境を作ることができるはずです。

```bash
conda create --name d2l python=3.8 -y
```

## D2L ノートブックのダウンロード
You can click the "All Notebooks" tab on the top of any HTML page to download and unzip the code. Alternatively, if you have unzip (otherwise run sudo apt install unzip) available:

次にこの書籍のコードをダウンロードしましょう。いずれかの HTML のページの上部から All Notebooks タブをクリックして、コードをダウンロードして、zip を回答します。もしくは、unzip をインストール済み (`run sudo apt install unzip`でインストール可能) であれば、以下で行うことも可能です。

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

ここで `d2l` の環境を activate することができます。

```bash
conda activate d2l
```

## フレームワーク と `d2l` パッケージのインストール

深層学習のフレームワークをインストールする前に、まず、利用する計算機に適切なGPU (標準的なノートPCでグラフィックスのために利用されるGPUは対象外です) が利用可能かどうかを確認してください。GPU サーバにインストールしようとしているなら、:ref:`subsec_gpu` に従って、依存ライブラリについて、GPU に対して親和性の高いバージョンをインストールしてください。

もし GPU がなければ、現時点では心配はありません。CPU でも十分に序盤の章をこなすことができるでしょう。より規模の大きいモデルを動かす際には GPU が必要になるということを覚えておいてください.
CPU バージョンをインストールするためには、以下のコマンドを実行します。



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
以下の方法で、CPUとGPUサポートの Tensorflow をインストールすることができます。

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:

次に、この書籍でよく使う関数やクラスをまとめた `d2l` パッケージもインストールしましょう。

```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```

上記のインストールを完了したら、実行のために Jupyter ノートブックを開きます。

```bash
jupyter notebook
```

この段階で、http://localhost:8888 (通常、自動で開きます) をブラウザで開くことができます。そして、この書籍の各章のコードを実行することができます。この書籍のコードを実行したり、深層学習のフレームワークや `d2l` のパッケージを更新する前には、`conda activate d2l` を必ず実行して実行環境を activate しましょう。環境から出る場合は、`conda deactivate` を実行します。



## GPUのサポート
:label:`subsec_gpu`

:begin_tab:`mxnet`
デフォルトでは、MXNetはあらゆるコンピュータ (ノートパソコンも含む)で実行できるように、GPUをサポートしないバージョンがインストールされます。この書籍の一部は、GPUの利用を必要としたり、推薦したりします。もし読者のコンピュータが、NVIDIAのグラフィックカードを備えていて、[CUDA](https://developer.nvidia.com/cuda-downloads)がインストールされているのであれば、GPUを利用可能なMXNet をインストールしましょう。CPU のみをサポートするMXNet をインストールしていた場合は、以下を実行して、まずそれを削除する必要があります。

```
pip uninstall mxnet
```

次に、インストール済みの CUDA のバージョンを知る必要があります。
 `nvcc --version` や `cat /usr/local/cuda/version.txt` を実行して知ることができるかもしれません。CUDA 10.1 がインストールされているとすれば、以下のコマンドで MXNet をインストールすることができます。

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python
# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```

CUDA のバージョンにあわせて、最後の数字を変えることができます。`cu100` は
CUDA 10.0、`cu90` は CUDA 9.0 を表します。

:end_tab:

:begin_tab:`pytorch,tensorflow`
デフォルトでは、これらの深層学習フレームワークは GPU をサポートします。NVIDIA の GPU の計算機をもっていて、[CUDA](https://developer.nvidia.com/cuda-downloads) をインストールしていれば、準備はすべて完了です。
:end_tab:


## 練習

1. この本のコードをダウンロードして、実行環境をインストールしましょう。

:begin_tab:`mxnet`
[フォーラム](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[フォーラム](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[フォーラム](https://discuss.d2l.ai/t/436)
:end_tab:

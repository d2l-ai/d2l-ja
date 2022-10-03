# インストール
:label:`chap_installation`

起動して実行するには、Python、Jupyter Notebook、関連するライブラリ、および本自体を実行するために必要なコードを実行するための環境が必要です。 

## Miniconda をインストールする

最も単純なオプションは、[Miniconda](https://conda.io/en/latest/miniconda.html) をインストールすることです。Python 3.x バージョンが必要であることに注意してください。マシンに既に conda がインストールされている場合は、次の手順をスキップできます。 

Miniconda の Web サイトにアクセスし、お使いの Python 3.x のバージョンとマシンアーキテクチャに基づいて、システムに適したバージョンを決定してください。お使いの Python のバージョンが 3.9 (テスト版) だとします。macOS を使用している場合は、名前に「macOSX」という文字列が含まれる bash スクリプトをダウンロードし、ダウンロード場所に移動して、次のようにインストールを実行します (インテル Mac を例にとります)。

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```

Linuxユーザーは、名前に「Linux」という文字列を含むファイルをダウンロードし、ダウンロード場所で以下を実行します。

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```

次に、`conda` を直接実行できるようにシェルを初期化します。

```bash
~/miniconda3/bin/conda init
```

次に、現在のシェルを閉じてから再度開きます。次のようにして新しい環境を作成できるはずです。

```bash
conda create --name d2l python=3.9 -y
```

これで、`d2l` 環境をアクティブ化できます。

```bash
conda activate d2l
```

## ディープラーニングフレームワークと `d2l` パッケージのインストール

ディープラーニングフレームワークをインストールする前に、マシンに適切な GPU があるかどうかを確認してください (標準的なラップトップのディスプレイに電力を供給する GPU は、私たちの目的には関係ありません)。たとえば、コンピューターに NVIDIA GPU が搭載され、[CUDA](https://developer.nvidia.com/cuda-downloads) がインストールされていれば、これで準備は完了です。お使いのマシンにGPUが搭載されていなければ、まだ心配する必要はありません。CPUは、最初の数章を読み進めるのに十分な馬力を提供します。大きなモデルを実行する前に GPU にアクセスすることを忘れないでください。

:begin_tab:`mxnet`
GPU 対応バージョンの MXNet をインストールするには、インストールされている CUDA のバージョンを確認する必要があります。これを確認するには、`nvcc --version` または `cat /usr/local/cuda/version.txt` を実行します。CUDA 10.2 がインストールされていると仮定して、次のコマンドを実行します。

```bash
# For macOS and Linux users
pip install mxnet-cu102==1.7.0

# For Windows users
pip install mxnet-cu102==1.7.0 -f https://dist.mxnet.io/python
```

最後の桁は、CUDAのバージョンに応じて変更できます。たとえば、CUDA 10.1の場合は`cu101`、CUDA 9.0の場合は`cu90`です。 

マシンに NVIDIA GPU または CUDA がない場合は、次の手順で CPU バージョンをインストールできます。

```bash
pip install mxnet==1.7.0.post1
```
:end_tab:

:begin_tab:`pytorch`
PyTorch は、以下のように CPU または GPU をサポートしてインストールできます。

```bash
pip install torch torchvision
```
:end_tab:

:begin_tab:`tensorflow`
TensorFlow は、次のように CPU または GPU をサポートしてインストールできます。

```bash
pip install tensorflow tensorflow-probability
```
:end_tab:

次のステップは、この本でよく使われる関数とクラスをカプセル化するために開発した `d2l` パッケージをインストールすることです。

```bash
pip install d2l==1.0.0a1.post0
```

## コードのダウンロードと実行

次に、ノートブックをダウンロードして、ブックの各コードブロックを実行できるようにします。[the D2L.ai website](https://d2l.ai/)のHTMLページの上部にある「ノートブック」タブをクリックするだけで、コードをダウンロードして解凍できます。または、以下のようにコマンドラインからノートブックをフェッチできます。

:begin_tab:`mxnet`
```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```
:end_tab:

:begin_tab:`pytorch`
```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```
:end_tab:

:begin_tab:`tensorflow`
```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```
:end_tab:

`unzip` をまだインストールしていない場合は、まず `sudo apt-get install unzip` を実行します。これで、以下を実行して Jupyter Notebook サーバーを起動できます。

```bash
jupyter notebook
```

この時点で、Web ブラウザで http://localhost:8888 (既に自動的に開かれている場合があります) を開くことができます。その後、本の各セクションのコードを実行できます。新しいコマンドラインウィンドウを開くたびに、D2L ノートブックを実行したり、パッケージ (ディープラーニングフレームワークまたは `d2l` パッケージ) を更新したりする前に、`conda activate d2l` を実行してランタイム環境をアクティブ化する必要があります。環境を終了するには、`conda deactivate` を実行します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:

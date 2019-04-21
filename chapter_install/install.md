# インストール

ハンズオンを始めてもらうために、Pythonの環境、Jupyterの対話的なノートブック、関連するライブラリ、*この書籍を実行するための*コードをセットアップしてもらう必要があります。


## コードの入手と実行環境のインストール

この書籍とコードはフリーでダウンロードすることができます。簡単にするために、すべてのライブラリをインストールするためのcondaという人気のPythonのパッケージ管理ツールをおすすめします。WindowsのユーザとLinux/macOSのユーザは、それぞれ以下の説明に従ってください。


### Windowsユーザ

もし、この書籍のコードを最初に動かすのであれば、以下の5つのステップを実行する必要があります。。一度動かしていれば、Step 4やStep 5にスキップすることができます。

Step 1は、利用するOSに応じて、[Miniconda](https://conda.io/en/master/miniconda.html)をダウンロードしてインストールします。インストールでは、"Add Anaconda to the system PATH environment variable (Anacondaをシステム環境変数PATHに追加する)"のオプションを選ぶ必要があります。

Step 2は、この書籍のコードの圧縮ファイルをダウンロードします。コードは https://www.d2l.ai/d2l-en-1.0.zip からダウンロード可能です。zipファイルをダウンロードしたら、`d2l-en`というフォルダを作成して、zipファイルをフォルダに展開します。カレントフォルダで、コマンドラインのモードに入るため、ファイルのエクスプローラーのアドレスバーに`cmd`と入力します。

Step3は、この書籍で必要になるライブラリをインストールするために、condaを利用した仮想の事項環境を作成します。`environment.yml`というファイルが、ダウンロードしたzipファイルの中に含まれています。ライブラリ (MXNetや`d2lzh`といったパッケージなど)や、この書籍のコードと依存関係にあるライブラリのバージョンを見るために、そのファイルをテキストエディタで開いてみましょう。

```
conda env create -f environment.yml
```
Step 4は、さきほど作成した環境を有効にします。この環境を有効にすることは、この書籍のコードを実行するための前提条件になります。その環境から抜けるためには、`conda deactivate`というコマンドを実行します (もし、condaのバージョンが4.4未満であれば、`deactivate`というコマンドを実行します。)

```
# If the conda version is lower than 4.4, use the command `activate gluon`
conda activate gluon
```

Step 5ではJupter notebookを開きます。

```
jupyter notebook
```

この時点で http://localhost:8888 をブラウザで開くと（通常、自動的に開かれます）、この書籍の各節のコードを見たり実行したりすることができます。

### Linux/macOSユーザ

Step 1は、利用するOSに応じて、[Miniconda](https://conda.io/en/master/miniconda.html)をダウンロードしてインストールします。sh形式のファイルになっています。ターミナルのアプリケーションで開いて、以下のように、shファイルを実行するコマンド入力します

```
# The file name is subject to change, always use the one downloaded from the
# Miniconda website
sh Miniconda3-latest-Linux-x86_64.sh
```

インストール中に、利用条件が表示されるでしょう。"↓"を押して読み進め、"Q"を押して読むのを終了します。その後、以下のような質問に答えます。

```
Do you accept the license terms? [yes|no]　（ライセンス条件に同意しますか? [はい|いいえ])
[no] >>> yes
Do you wish the installer to prepend the Miniconda3 install location
to PATH in your /home/your_name/your_file ? [yes|no]
(/home/your_name/your_file のPATHの最初に、Miniconda3のインストール位置を追加して良いですか? [はい|いいえ])
[no] >>> yes
```

インストールが完了した後にcondaを有効化します。Linuxユーザは`source ~/.bashrc`を実行するか、コマンドラインのアプリケーションを再起動する必要があります。macOSのユーザは、`source ~/.bash_profile`を実行するか、コマンドラインのアプリケーションを再起動します。

Step 2では、この書籍のコードの圧縮ファイルをダウンロードしてフォルダに展開します。そして次のコマンドを実行します。`unzip`をインストールしていないLinuxユーザは、`sudo apt install unzip`のコマンドを実行することでunzipをインストールすることができます。

```
mkdir d2l-en && cd d2l-en
curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```
Step 3からStep 5までは、上述しているWindowsユーザ向けのStepを参照してください。
もし、condaのバージョンが4.4未満であれば、Step 4のコマンドを`source activate gluon`に置きかえて、`source deactivate`のコマンドを利用して仮想環境から抜けます。

## コードと実行環境の更新

ディープラーニングとMXNetは急速に進化するため、このオープンソースの書籍も定期的に更新されて、リリースされていくでしょう。この書籍のオープンソースのコンテンツ（例えば、コード）を対応する実行環境（例えば、最新版のMXNet)で更新するためには、以下の手順に従ってください。

Step 1は、この書籍のコードを含む最新版の圧縮ファイルをダウンロードします。そのファイルは、https://www.d2l.ai/d2l-en.zip からダウンロードできます。zipファイルを展開したら、そのフォルダ `d2l-en` に入ります。

Step 2は、その実行環境を次のコマンドでアップデートします。

```
conda env update -f environment.yml
```

以降の、環境を有効化したり、Jupyterを実行する手順は、すでに説明した手順と同じように行います。

## GPUのサポート

デフォルトでは、MXNetはあらゆるコンピュータ (ノートパソコンも含む)で実行できるように、GPUを利用しないようにインストールされます。この書籍の一部は、GPUの利用を必要としたり、推薦したりします。もし読者のコンピュータが、NVIDIAのグラフィックカードを備えていて、CUDAがインストールされているのであれば、CUDAを利用可能なビルドをダウンロードするように、conda環境を修正すべきです。

Step 1では、GPUをサポートしないMXNetをアンインストールします。もし、この書籍を実行する仮想環境をインストールしていれば、GPUをサポートしないMXNetをアンインストールした環境を有効化する必要があります。

```
pip uninstall mxnet
```

そして仮想環境から抜けます。

Step 2では、`environment.yml`の環境に関する記述を更新します。おそらく、`mxnet`を`mxnet-cu90`で書き換えるでしょう。ハイフンのあとの数字 (上記の90)は、読者がインストールしているCUDAのバージョンに対応している必要があります。例えば、CUDA 8.0を利用していれば、`mxnet-cu90`ではなく`mxnet-cu80`を利用する必要があります。これを、conda環境を作成する*前*に実行する必要があります。もし作成前に実行しなければ、あとで再度ビルドする必要があるでしょう。

Step 3では、以下のコマンドで仮想環境を更新します。

```
conda env update -f environment.yml
```

以上の手順を終えていれば、仮想環境を有効化することで、GPUがサポートされたMXNetを使用して、この書籍の内容を実行できます。もし、更新されたコードを後でダウンロードした場合は、GPUサポートのMXNetを利用するための3ステップを再度実行する必要があります。

## 練習

1. この本のコードをダウンロードして、実行環境をインストールしましょう。


## [議論](https://discuss.mxnet.io/t/2315)のためのQRコード

![](../img/qr_install.svg)

# Gluonで始めよう

始めるにあたって、ノートブックの実行に必要なコードのダウンロードとインストールが必要です。この節を読み飛ばしても、以降の理論的な理解に影響ありませんが、読者がここでハンズオンを経験することを強くおすすめします。コードを修正したり、書いたりして、その結果を確認することで、更に多くの学びを得られるでしょう。手短ではありますが、始めるために以下を行う必要があります。

1. condaをインストール
1. 書籍で動くコードをダウンロード
1. もしGPUをもっていて、まだ使ったことがなければ、GPUドライバをインストール
1. MXNetや書籍のコード例を実行するためのconda環境のビルド

## Conda

簡単に全てのライブラリをインストールする方法として、[conda](https://conda.io)という人気のPythonパッケージの管理ツールをおすすめします。

1. [conda.io/miniconda.html](https://conda.io/miniconda.html)にある[Miniconda](https://conda.io/miniconda.html)をOSにあわせて、ダウンロード、インストールします。
1. `source ~/.bashrc` (Linux) や `source ~/.bash_profile` (macOS)を実行してシェルを更新します。環境変数PATHにAnacondaが追加されていることを確認してください。
1. この書籍からノートブックを含むtarballをダウンロードします。ファイルは、[www.d2l.ai/d2l-en-1.0.zip](https://www.d2l.ai/d2l-en-1.0.zip)においています。代わりに、Githubから最新バージョンをクローンすることもできます。
1. ZIPファイルを解凍して、その中身をチュートリアルのためのフォルダに移動させます。

上記の操作は、Linuxのコマンドラインを利用して、以下のように実行することができます。MacOSの場合は、最初の行をLinuxからMacOSXに変更してください。Windowsの場合は、上のリンクを参考にしてください。

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
mkdir d2l-en
cd d2l-en
curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip
unzip d2l-en-1.0.zip
rm d2l-en-1.0.zip
```

## GPUサポート

デフォルトでは、MXNetはどのコンピュータ（多くの場合ノートパソコン）でも動くために、GPUサポート無しでインストールされます。もし読者がGPUを利用できるコンピュータを持つ幸せな人であれば、CUDAを利用可能なビルドをダウンロードしてconda環境を修正すべきでしょう。もちろん適切なドライバがインストールされている必要があります。具体的には、以下を実行する必要があります。

1. 利用するGPUにあった[NVIDIA Drivers](https://www.nvidia.com/drivers)がインストールされていることを確認
1. GPUのためのプログラミング言語である[CUDA](https://developer.nvidia.com/cuda-downloads)をインストール
1. 深層学習用に最適化された多数のライブラリを含む[CUDNN](https://developer.nvidia.com/cudnn)をインストール
1. もし、さらなる高速化を求めるなら、[TensorRT](https://developer.nvidia.com/tensorrt)をインストール

このインストールのプロセスはいくらか時間がかかり、多数の異なるライセンスへの同意と、インストールスクリプトの実行を求められるでしょう。OSやハードウェアによって細かい部分は異なります。

次に、`environment.yml`に記載された環境を更新します。`mxnet`を
`mxnet-cu92`や読者がインストールしたCUDAのバージョンで書き換えます。もしCUDA 8.0を利用しているのであれば、`mxnet-cu92`を `mxnet-cu80`に書き換えます。これをconda環境を作る*前に*行うほうが良いです。もしそうでなければ、後に再起動が必要になるでしょう。Linuxでは以下のようなコマンドで編集できます（Windowsではメモ帳を使って`environment.yml`を直接編集できます。）

```
cd d2l
emacs environment.yml
```

## Conda環境

手短に言うと、condaは再現可能かつ信頼できる方法で、一連のPythonライブラリをセットアップする機構を提供しており、すべてのソフトウェアの依存関係が満足されるようになっています。ここでは、始めるにあたって必要なことを説明します。

1. condaを利用して環境の作成とアクティベートをします（利用可能にします）。作業を楽にするために、全ての構成を保持した`environment.yml`というファイルを作成しています。
1. 環境をアクティベートします。
1. 実験を始めるためにJupyterノートブックを開きます。

### Windows

これまで通り、ターミナルを開いて以下を実行します。

```
conda env create -f environment.yml
cd d2l-en
activate gluon
jupyter notebook
```

もし、ライブラリをかえて上記と同じ操作を行うときは、最初の1行目(conda createのところ)はスキップしてください。これで構成
をアクティブにすることができるでしょう。Jupyer notebookの代わりにJupyterLabを使うこともできます。その場合は`jupyter noebook`としているところを`jupyter lab`にします。JupyterLabはもっと機能的なJupyter環境を提供してくれます(JupyerLabがインストールされている必要があります)。アクティブなcondaのgluon環境において、`conda install jupyterlab`とすると手動でインストールできます。

もしブラウザが正常に動いて入れば、Jupyterを起動すると、ブラウザ内で新しいウィンドウが開くでしょう。もし何も起こらなければ、http://localhost:8888 にアクセスして手動で開きます。ノートブックの中には、データセットや学習済みモデルを自動でダウンロードするものもあります。`MXNET_GLUON_REPO`の変数を上書きすれば、レポジトリの場所を変更することができます。

### LinuxとMacOSX

Linuxの場合の手順も非常に似ていて、少しだけ異なるanacondaのコマンドラインオプションを利用します。

```
conda env create -f environment.yml
cd d2l-en
source activate gluon
jupyter notebook
```

Windowsと他のインストールとの主な違いは、LinuxとmacOSでは`source activate gluon`を利用するのに対して、Windowsでは`activate gluon`を利用します。こうして、同じことがWindowsにも適用されます。より高性能な環境を求めるならJupyterLabをインストールしましょう。

## Gluonのアップデート

もしレポジトリをアップデートしたい場合や、新しいバージョンのCUDAやMXNetをインストールした場合は、condaのコマンドを利用して以下を実行します。これまでと同様に、あわせてパッケージをアップデートすることも忘れないようにしましょう。

```
cd d2l-en
conda env update -f environment.yml
```

## まとめ

* Conda is a Python package manager that ensures that all software dependencies are met.
* `environment.yml` has the full configuration for the book. All notebooks are available for download or on GitHub.
* Install GPU drivers and update the configuration if you have GPUs. This will shorten the time to train significantly.

## Exercise

1. Download the code for the book and install the runtime environment.
1. Follow the links at the bottom of the section to the forum in case you have questions and need further help.
1. Create an account on the forum and introduce yourself.

## Discuss on our Forum

<div id="discuss" topic_id="2315"></div>

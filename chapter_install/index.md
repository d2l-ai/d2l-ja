# インストール
:label:`chap_installation`

ハンズオンを始めてもらうために、Pythonの環境、Jupyterの対話的なノートブック、関連するライブラリ、この書籍を実行するためのコードをセットアップしてもらう必要があります。


## Miniconda のインストール

最もシンプルに始めるために
[Miniconda](https://conda.io/en/latest/miniconda.html)をインストールしましょう。Python 3系が推奨です。もし conda がすでにインストールされていれば以下のステップをスキップすることができます。ウェブサイトから、対応する Miniconda の sh ファイルをダウンロードして、コマンドラインから `sh <FILENAME> -b` を実行してインストールします。macOS のユーザは以下のように行います。

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

そして Linux ユーザは以下のように行います。
```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

次に、`conda`から直接シェルを初期化します。

```bash
~/miniconda3/bin/conda init
```

いまのシェルを閉じて再度開いてください。以下のように新しい環境を作ることができるはずです。

```bash
conda create --name d2l -y
```

## D2L ノートブックのダウンロード

次にこの書籍のコードをダウンロードしましょう。コードを[link](https://d2l.ai/d2l-en-0.7.1.zip)からダウンロードして解凍します。もし `unzip` がインストール済みであれば (なければ `sudo apt install unzip` でインストールできます)、代わりに以下でも可能です。

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-0.7.1.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

ここで `d2l` の環境を activate して、`pip` をインストールします。このコマンドの最後に `y` を入れておきましょう。

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## MXNet と `d2l` パッケージのインストール

MXNet をインストールする前に、まず、利用する計算機に適切なGPU (標準的なノートPCでグラフィックスのために利用されるGPUは対象外です) が利用可能かどうかを確認してください。GPU サーバにインストールしようとしているなら、GPUサポートのMXNet をインストールするための手順 :ref:`subsec_gpu` に従ってください。

もし GPU がなければ、CPUバージョンをインストールしましょう。最初の数章を行う際には十分な性能でしょうが、より規模の大きいモデルを動かす際には GPU を必要とするかもしれません。


```bash
# For Windows users
pip install mxnet==1.6.0b20190926

# For Linux and macOS users
pip install mxnet==1.6.0b20191122
```

この書籍でよく使う関数やクラスをまとめた `d2l` パッケージもインストールしましょう。

```bash
pip install d2l==0.11.1
```

インストールできたら、実行のために Jupyter ノートブックを開きます。

```bash
jupyter notebook
```

この段階で、http://localhost:8888 (通常、自動で開きます) をブラウザで開くことができます。そして、この書籍の各章のコードを実行することができます。この書籍のコードを実行したり、MXNet や `d2l` のパッケージを更新する前には、`conda activate d2l` を必ず実行して実行環境を activate しましょう。環境から出る場合は、`conda deactivate` を実行します。

## 最新バージョンへ更新

この書籍と MXNet は絶えず改善を続けています。次々とリリースされる最新バージョンをチェックしましょう。

1.  https://d2l.ai/d2l-en.zip の URL は常に最新版を保持しています。
2. `d2l` パッケージを `pip install d2l --upgrade` を実行して更新しましょう。
3. CPU バージョンの場合は, `pip install -U --pre mxnet` MXNet を更新できます。


## GPUのサポート

:label:`subsec_gpu`

デフォルトでは、MXNetはあらゆるコンピュータ (ノートパソコンも含む)で実行できるように、GPUを利用しないようにインストールされます。この書籍の一部は、GPUの利用を必要としたり、推薦したりします。もし読者のコンピュータが、NVIDIAのグラフィックカードを備えていて、[CUDA](https://developer.nvidia.com/cuda-downloads)がインストールされているのであれば、GPUを利用可能なMXNet をインストールしましょう。CPU のみをサポートするMXNet をインストールしていた場合は、以下を実行して、まずそれを削除する必要があります。

```
pip uninstall mxnet
```

次に、インストール済みの CUDA のバージョンを知る必要があります。
 `nvcc --version` や `cat /usr/local/cuda/version.txt` を実行して知ることができるかもしれません。CUDA 10.1 がインストールされているとすれば、以下のコマンドで MXNet をインストールすることができます。

```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0b20191122
```

CPU バージョンと同様に、GPUを利用可能な MXNet は `pip install -U --pre mxnet-cu101` で更新できます。CUDAのバージョンに合わせて、最後の数字を変えることができます。例えば、CUDA 10.0であれば `cu100`、CUDA 9.0であれば `cu90` です。利用可能なMXNetのバージョンをすべて調べるためには、`pip search mxnet` を実行します。

## 練習

1. この本のコードをダウンロードして、実行環境をインストールしましょう。


## [議論](https://discuss.mxnet.io/t/2315)のためのQRコード

![](../img/qr_install.svg)

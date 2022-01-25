# Jupyter を使う
:label:`sec_jupyter`

このセクションでは、本書の章にあるコードを Jupyter Notebooks を使用して編集および実行する方法について説明します。:ref:`chap_installation` の説明に従って、Jupyter がインストールされ、コードをダウンロードしたことを確認します。Jupyterについて詳しく知りたい場合は、[Documentation](https://jupyter.readthedocs.io/en/latest/)の優れたチュートリアルをご覧ください。 

## コードをローカルで編集して実行する

本のコードのローカルパスが「xx/yy/d2l-en/」であるとします。シェルを使用してディレクトリをこのパス (`cd xx/yy/d2l-en`) に変更し、`jupyter notebook` コマンドを実行します。ブラウザがこれを自動的に行わない場合、http://localhost:8888 and you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00` を開いてください。 

![The folders containing the code in this book.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

Webページに表示されているフォルダをクリックすると、ノートブックファイルにアクセスできます。通常、接尾辞は「.ipynb」です。簡潔にするために、一時的な「test.ipynb」ファイルを作成します。クリックすると表示される内容は :numref:`fig_jupyter01` のようになります。このノートブックには、マークダウンセルとコードセルが含まれています。マークダウンセルの内容には、「これはタイトルです」と「これはテキストです」が含まれます。code セルには 2 行の Python コードが含まれています。 

![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

マークダウンセルをダブルクリックして編集モードに入ります。:numref:`fig_jupyter02` に示すように、セルの最後に新しいテキスト文字列「Hello world.」を追加します。 

![Edit the markdown cell.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

:numref:`fig_jupyter03` のように、メニューバーの「Cell」$\rightarrow$「Run Cells」をクリックして、編集したセルを実行します。 

![Run the cell.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

実行後、マークダウンセルは :numref:`fig_jupyter04` のようになります。 

![The markdown cell after editing.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

次に、コードセルをクリックします。:numref:`fig_jupyter05` に示すように、コードの最後の行の後に要素に 2 を掛けます。 

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

ショートカット (デフォルトでは「Ctrl+Enter」) を使用してセルを実行し、:numref:`fig_jupyter06` から出力結果を取得することもできます。 

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

ノートブックにさらに多くのセルが含まれている場合は、メニューバーの「Kernel」$\rightarrow$「Restart & Run All」をクリックして、ノートブック全体のすべてのセルを実行できます。メニューバーの「ヘルプ」$\rightarrow$「キーボードショートカットの編集」をクリックすると、好みに合わせてショートカットを編集できます。 

## [詳細オプション]

ローカルでの編集以外にも、ノートブックのマークダウン形式での編集と、Jupyter のリモートでの実行という 2 つの重要なことがあります。後者は、より高速なサーバーでコードを実行したい場合に重要です。Jupyter のネイティブな.ipynb 形式には、ノートブックの内容に特有のものではなく、主にコードの実行方法と実行場所に関連する多くの補助データが格納されているため、前者は重要です。これは Git にとって混乱を招き、コントリビューションのマージが非常に困難になります。幸いなことに、Markdownにはネイティブ編集という代替手段があります。 

### Jupyter のマークダウンファイル

この本の内容に貢献するには、GitHub 上のソースファイル (ipynb ファイルではなく md ファイル) を修正する必要があります。notedown プラグインを使えば、Jupyter で直接 md 形式のノートブックを修正できます。 

まず、notedown プラグインをインストールし、Jupyter Notebook を実行して、プラグインをロードします。

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Jupyter Notebook を実行するたびにデフォルトで notedown プラグインを有効にするには、以下を実行します:まず、Jupyter Notebook 設定ファイルを生成します (既に生成されている場合は、この手順をスキップできます)。

```
jupyter notebook --generate-config
```

次に、Jupyter ノートブック設定ファイルの最後に次の行を追加します (Linux/macOS の場合、通常は `~/.jupyter/jupyter_notebook_config.py` というパスにあります)。

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

その後、`jupyter notebook` コマンドを実行して notedown プラグインをデフォルトで有効にするだけで済みます。 

### リモートサーバーでの Jupyter Notebook の実行

Jupyter Notebook をリモートサーバーで実行し、ローカルコンピューターのブラウザーからアクセスしたい場合があります。Linux または macOS がローカルマシンにインストールされている場合 (Windows は PuTTY などのサードパーティ製ソフトウェアを通じてこの機能をサポートすることもできます)、ポートフォワーディングを使用できます。

```
ssh myserver -L 8888:localhost:8888
```

上記はリモートサーバ `myserver` のアドレスです。その後、http://localhost:8888 を使用して Jupyter ノートブックを実行しているリモートサーバー `myserver` にアクセスできます。次のセクションでは、AWS インスタンスで Jupyter Notebook を実行する方法について詳しく説明します。 

### タイミング

`ExecuteTime` プラグインを使用して、Jupyter ノートブックの各コードセルの実行時間を計ることができます。プラグインをインストールするには、以下のコマンドを使用します。

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## [概要

* 本の章を編集するには、Jupyter でマークダウン形式を有効にする必要があります。
* ポート転送を使用すると、サーバーをリモートで実行できます。

## 演習

1. このブックのコードをローカルで編集して実行してみます。
1. この本のコードをポート転送で*リモート*で編集して実行してみてください。
1. $\mathbb{R}^{1024 \times 1024}$ の 2 つの正方行列について $\mathbf{A}^\top \mathbf{B}$ 対 $\mathbf{A} \mathbf{B}$ を測定します。どちらが速いですか？

[Discussions](https://discuss.d2l.ai/t/421)

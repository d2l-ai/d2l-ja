# Jupyter ノートブックの使用
:label:`sec_jupyter`

このセクションでは、Jupyter Notebook を使用してこの本の各セクションのコードを編集および実行する方法について説明します。:ref:`chap_installation` の説明に従って、Jupyter をインストールし、コードをダウンロードしたことを確認します。Jupyterについて詳しく知りたい場合は、[documentation](https://jupyter.readthedocs.io/en/latest/)の優れたチュートリアルをご覧ください。 

## コードをローカルで編集して実行する

本のコードのローカルパスが `xx/yy/d2l-en/` であるとします。シェルを使用してディレクトリをこのパス (`cd xx/yy/d2l-en`) に変更し、コマンド`jupyter notebook`を実行します。ブラウザが自動的にこれを行わない場合は、http://localhost:8888 and you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00` を開きます。 

![The folders containing the code of this book.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

Web ページに表示されているフォルダをクリックすると、ノートブックファイルにアクセスできます。通常、接尾辞「.ipynb」が付いています。簡潔にするために、一時的な「test.ipynb」ファイルを作成します。クリックした後に表示されるコンテンツは、:numref:`fig_jupyter01`に表示されます。このノートブックには、マークダウンセルとコードセルが含まれています。マークダウンセルの内容には、「これはタイトルです」と「これはテキストです」が含まれます。コードセルには 2 行の Python コードが含まれています。 

![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

マークダウンセルをダブルクリックして編集モードに入ります。:numref:`fig_jupyter02` に示すように、セルの最後に新しいテキスト文字列「Hello world.」を追加します。 

![Edit the markdown cell.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

:numref:`fig_jupyter03`に示すように、メニューバーの「セル」$\rightarrow$「セルの実行」をクリックして、編集したセルを実行します。 

![Run the cell.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

実行後、:numref:`fig_jupyter04`にマークダウンセルが表示されます。 

![The markdown cell after running.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

次に、コードセルをクリックします。:numref:`fig_jupyter05`に示すように、コードの最後の行の後に要素に2を掛けます。 

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

ショートカット (デフォルトでは「Ctrl+Enter」) でセルを実行し、:numref:`fig_jupyter06`からの出力結果を取得することもできます。 

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

ノートブックにさらに多くのセルが含まれている場合は、メニューバーの「Kernel」$\rightarrow$「Restart & Run All」をクリックして、ノートブック全体のすべてのセルを実行できます。メニューバーの「ヘルプ」$\rightarrow$「キーボードショートカットの編集」をクリックすると、好みに応じてショートカットを編集できます。 

## アドバンスオプション

ローカル編集以外にも、2つのことが非常に重要です。マークダウン形式でノートブックを編集することと、Jupyterをリモートで実行することです。後者は、コードを高速なサーバーで実行したい場合に重要です。Jupyter のネイティブ ipynb フォーマットには、コンテンツとは無関係な補助データが多く格納されており、主にコードが実行される方法と場所に関連しているため、前者は重要です。これは Git にとって混乱を招き、コントリビューションのレビューが非常に困難になります。幸いなことに、マークダウン形式のネイティブ編集という代替手段があります。 

### Jupyter のマークダウンファイル

この本のコンテンツに貢献したいのであれば、GitHub のソースファイル (ipynb ファイルではなく md ファイル) を変更する必要があります。notedownプラグインを使用すると、Jupyterでmd形式のノートブックを直接変更できます。 

まず、notedown プラグインをインストールし、Jupyter Notebook を実行して、プラグインをロードします。

```
pip install d2l-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Jupyter Notebook を実行するたびに、デフォルトで notedown プラグインをオンにすることもできます。まず、Jupyter Notebook 設定ファイルを生成します (既に生成されている場合は、このステップをスキップできます)。

```
jupyter notebook --generate-config
```

次に、Jupyter Notebook 設定ファイルの最後に次の行を追加します (Linux/macOS の場合、通常は `~/.jupyter/jupyter_notebook_config.py` のパス)。

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

その後、`jupyter notebook`コマンドを実行して、デフォルトでnotedownプラグインをオンにするだけです。 

### Jupyter Notebooks をリモートサーバーで実行する

Jupyter ノートブックをリモートサーバーで実行し、ローカルコンピューターのブラウザーからアクセスしたい場合があります。Linux または macOS がローカルマシンにインストールされている場合 (Windows は PuTTY などのサードパーティソフトウェアを介してこの機能をサポートすることもできます)、ポート転送を使用できます。

```
ssh myserver -L 8888:localhost:8888
```

上記の文字列 `myserver` は、リモートサーバーのアドレスです。次に http://localhost:8888 を使用して、Jupyter ノートブックを実行するリモートサーバー `myserver` にアクセスできます。AWS インスタンスで Jupyter ノートブックを実行する方法については、この付録の後半で詳しく説明します。 

### タイミング

`ExecuteTime`プラグインを使用して、Jupyterノートブックの各コードセルの実行時間を計ることができます。以下のコマンドを使用してプラグインをインストールします。

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## まとめ

* Jupyter Notebook ツールを使用して、本の各セクションを編集、実行、投稿できます。
* Jupyter ノートブックは、ポート転送を使用してリモートサーバーで実行できます。

## 演習

1. この本のコードをローカルマシンの Jupyter Notebook で編集して実行します。
1. Jupyter Notebookで、この本のコードをポート転送経由で*リモート*で編集して実行します。
1. $\mathbb{R}^{1024 \times 1024}$ の 2 つの正方行列の演算 $\mathbf{A}^\top \mathbf{B}$ と $\mathbf{A} \mathbf{B}$ の実行時間を測定します。どっちが速い？

[Discussions](https://discuss.d2l.ai/t/421)

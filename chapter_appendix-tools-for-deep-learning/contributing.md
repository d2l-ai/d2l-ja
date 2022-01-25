# この本に寄稿する
:label:`sec_how_to_contribute`

[readers](https://github.com/d2l-ai/d2l-en/graphs/contributors) による貢献は、この本の向上に役立っています。タイプミス、古いリンク、引用を見逃したと思われるもの、コードがエレガントに見えない、説明が不明なものを見つけた場合は、貢献して読者を助けてください。通常の本では、印刷間隔 (および誤字訂正間) の遅延は年単位で測定できますが、この本に改善点を組み込むには通常数時間から数日かかります。これはすべて、バージョン管理と継続的インテグレーションテストにより可能です。そのためには、[pull request](https://github.com/d2l-ai/d2l-en/pulls) を GitHub リポジトリにサブミットする必要があります。作成者がプルリクエストをコードリポジトリにマージすると、コントリビューターになります。 

## テキストの軽微な変更

最も一般的な貢献は、一文の編集やタイプミスの修正です。[github repo](https732293614) でソースファイルを探して、マークダウンファイルであるソースファイルを見つけることをお勧めします。次に、右上隅の「このファイルを編集」ボタンをクリックして、マークダウンファイルに変更を加えます。 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

完了したら、ページ下部の [ファイル変更の提案] パネルに変更内容を入力し、[ファイル変更の提案] ボタンをクリックします。変更を確認するための新しいページにリダイレクトされます (:numref:`fig_git_createpr`)。すべて問題なければ、「Create pull request」ボタンをクリックしてプルリクエストを送信できます。 

## 大きな変革を提案する

テキストやコードの大部分を更新する予定がある場合は、この本で使用されている形式についてもう少し詳しく知る必要があります。ソースファイルは [markdown format](https://daringfireball.net/projects/markdown/syntax) をベースにしており、数式、画像、章、引用を参照するなど、[d2lbook](http://book.d2l.ai/user/markdown.html) パッケージを通じて一連の拡張子が付けられています。任意の Markdown エディタを使用してこれらのファイルを開き、変更を加えることができます。 

コードを変更したい場合は、:numref:`sec_jupyter` で説明されているように Jupyter を使用してこれらの Markdown ファイルを開くことをお勧めします。これにより、変更を実行してテストできます。変更を送信する前に、必ずすべての出力をクリアしてください。更新したセクションが CI システムによって実行され、出力が生成されます。 

セクションによっては複数のフレームワーク実装をサポートしている場合があり、`d2lbook` を使用して特定のフレームワークをアクティブ化できます。そのため、他のフレームワーク実装は Markdown コードブロックになり、Jupyter で「すべて実行」を実行しても実行されません。つまり、まず `d2lbook` を次のコマンドでインストールします。

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```

`d2l-en` のルートディレクトリで、次のいずれかのコマンドを実行して特定の実装をアクティブ化できます。

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```

変更を送信する前に、すべてのコードブロック出力をクリアし、次の方法ですべてをアクティブ化してください。

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

デフォルトの実装ではない MXNet という新しいコードブロックを追加する場合は `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab tensorflow` for a TensorFlow code block, or `# @tab all` すべての実装で共有されるコードブロック。詳細については [d2lbook](http://book.d2l.ai/user/code_tabs.html) を参照してください。 

## 新しいセクションまたは新しいフレームワーク実装の追加

強化学習などの新しい章を作成したり、TensorFlow などの新しいフレームワークの実装を追加したりする場合は、電子メールまたは [github issues](https://github.com/d2l-ai/d2l-en/issues) を使用して、最初に作成者に連絡してください。 

## メジャーチェンジの提出

大きな変更を送信するには、標準の `git` プロセスを使用することをお勧めします。簡単に言うと、このプロセスは :numref:`fig_contribute` で説明されているとおりに機能します。 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

手順を詳しく説明します。すでに Git に慣れている場合は、このセクションをスキップしてもかまいません。具体的に言うと、コントリビューターのユーザー名は「astonzhang」と仮定します。 

### Git をインストールする

Git オープンソースブックには [how to install Git](https://git-scm.com/book/en/v2) が記載されています。これは通常、Ubuntu Linux では `apt install git` 経由で、macOS に Xcode 開発者ツールをインストールするか、GitHub の [desktop client](https://desktop.github.com) を使用して動作します。GitHub アカウントを持っていない場合は、サインアップする必要があります。 

### GitHub にログインする

ブックのコードリポジトリの [address](https://github.com/d2l-ai/d2l-en/) をブラウザに入力します。:numref:`fig_git_fork` の右上にある赤いボックスの `Fork` ボタンをクリックして、この本のリポジトリのコピーを作成します。これが*あなたのコピー*になり、好きなように変更できます。 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

これで、この本のコードリポジトリが、スクリーンショット :numref:`fig_git_forked` の左上に表示されている `astonzhang/d2l-en` のように、ユーザー名にフォーク (コピー) されます。 

![Fork the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### リポジトリのクローンを作成する

リポジトリをクローンする (ローカルコピーを作成する) には、リポジトリのアドレスを取得する必要があります。:numref:`fig_git_clone` の緑色のボタンはこれを表示します。このフォークを長期間保持する場合は、ローカルコピーがメインリポジトリで最新であることを確認してください。今のところ、:ref:`chap_installation` の指示に従って作業を開始してください。主な違いは、リポジトリの「自分のフォーク」をダウンロードしていることです。 

![Git clone.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

### ブックとプッシュを編集する

今度は本を編集する時です。:numref:`sec_jupyter` の指示に従って、Jupyter でノートブックを編集することをお勧めします。変更を加え、問題がないことを確認します。`~/d2l-en/chapter_appendix_tools/how-to-conttribute.md` ファイルのタイプミスを修正したと仮定します。その後、どのファイルを変更したかを確認できます。 

この時点で、Git は `chapter_appendix_tools/how-to-contribute.md` ファイルが変更されたことを知らせるメッセージを表示します。

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

これが目的であることを確認したら、以下のコマンドを実行します。

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

変更したコードは、リポジトリの個人用フォークに保存されます。変更の追加をリクエストするには、本の公式リポジトリに対するプルリクエストを作成する必要があります。 

### プルリクエスト

:numref:`fig_git_newpr` に示すように、GitHub 上のリポジトリのフォークに移動し、「新しいプルリクエスト」を選択します。これにより、編集とブックのメインリポジトリの現在の変更点を示す画面が開きます。 

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

### プルリクエストをサブミットする

最後に、:numref:`fig_git_createpr` に示すように、ボタンをクリックしてプルリクエストを送信します。プルリクエストで行った変更内容を必ず説明してください。これにより、著者が本をレビューし、本と統合しやすくなります。変更によっては、すぐに承認されたり、却下されたり、変更に関するフィードバックが得られる可能性が高くなります。それらを組み込んだら、準備は完了です。 

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

プルリクエストは、メインリポジトリのリクエストリストに表示されます。迅速に処理できるよう全力を尽くします。 

## [概要

* GitHub を使ってこの本に貢献できます。
* GitHub でファイルを直接編集して、軽微な変更を加えることができます。
* 大きな変更については、リポジトリをフォークしてローカルで編集し、準備ができたらコントリビューションし直してください。
* プルリクエストは、コントリビューションがまとめられている方法です。巨大なプルリクエストを送信しないようにしてください。これは理解し組み込むのが難しくなるからです。小さいものをいくつか送ってください。

## 演習

1. `d2l-en` リポジトリにスターを付けてフォークします。
1. 改善が必要なコードをいくつか見つけて、プルリクエストを送信してください。
1. 見逃していた参照を見つけてプルリクエストを送信してください。
1. 通常は、新しいブランチを使用してプルリクエストを作成する方が良い方法です。[Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) でそれを行う方法を学んでください。

[Discussions](https://discuss.d2l.ai/t/426)

# この本への貢献
:label:`sec_how_to_contribute`

[readers](https://github.com/d2l-ai/d2l-en/graphs/contributors)による寄稿は、この本の改善に役立ちます。タイプミス、古いリンク、引用を見逃したと思われるもの、コードがエレガントに見えない、または説明が不明なものを見つけた場合は、貢献して読者の助けてください。通常の本では、印刷間隔（およびタイプミスの修正間）の遅延は年単位で測定できますが、この本に改善点を組み込むには通常数時間から数日かかります。これはすべて、バージョン管理と継続的インテグレーション (CI) テストにより可能です。そのためには、[pull request](https://github.com/d2l-ai/d2l-en/pulls) を GitHub リポジトリに送信する必要があります。あなたのプルリクエストが作者によってコードリポジトリにマージされると、あなたはコントリビューターになります。 

## 軽微な変更の提出

最も一般的な貢献は、1つの文を編集するか、タイプミスを修正することです。ソースファイル (マークダウンファイル) を見つけるには、[GitHub repository](https732293614) でソースファイルを見つけることをお勧めします。次に、右上隅の「このファイルを編集」ボタンをクリックして、マークダウンファイルに変更を加えます。 

![Edit the file on Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

完了したら、ページ下部の「ファイル変更の提案」パネルに変更の説明を入力し、「ファイル変更の提案」ボタンをクリックします。変更を確認するための新しいページにリダイレクトされます (:numref:`fig_git_createpr`)。すべてが良ければ、「Create pull request」ボタンをクリックしてプルリクエストを送信できます。 

## 大きな変更を提案する

テキストやコードの大部分を更新する予定がある場合は、この本が使用している形式についてもう少し知っておく必要があります。ソースファイルは [markdown format](https://daringfireball.net/projects/markdown/syntax) に基づいており、方程式、画像、章、引用の参照など、[d2lbook](http://book.d2l.ai/user/markdown.html) パッケージによる一連の拡張子が付いています。任意のマークダウンエディタを使用してこれらのファイルを開き、変更を加えることができます。 

コードを変更したい場合は、:numref:`sec_jupyter`で説明されているように、Jupyter Notebookを使用してこれらのマークダウンファイルを開くことをお勧めします。変更を実行してテストできるようにします。変更を送信する前にすべての出力をクリアすることを忘れないでください。CI システムは、更新したセクションを実行して出力を生成します。 

一部のセクションでは、複数のフレームワーク実装をサポートしている場合があります。デフォルトの実装ではない新しいコードブロック (MXNet) を追加する場合は `# @tab` to mark this block on the beginning line. For example, ` # @tab pytorch` for a PyTorch code block, `# @tab tensorflow` for a TensorFlow code block, or `# @tab all` a shared code block for all implementations. You may refer to the [`d2lbook`](http://book.d2l.ai/user/code_tabs.html) パッケージの詳細については。 

## 主な変更の提出

大きな変更を送信するには、標準の Git プロセスを使用することをお勧めします。簡単に言うと、このプロセスは:numref:`fig_contribute`で説明されているように機能します。 

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

手順を詳しく説明します。既に Git に慣れている場合は、このセクションをスキップできます。具体的に言うと、コントリビューターのユーザー名は「astonzhang」と仮定します。 

### Git をインストールする

Git オープンソースの本には [how to install Git](https://git-scm.com/book/en/v2) が記載されています。これは通常、Ubuntu Linuxの`apt install git`を介して、macOSにXcode開発者ツールをインストールするか、GitHubの[desktop client](https://desktop.github.com)を使用して機能します。GitHub アカウントを持っていない場合は、サインアップする必要があります。 

### GitHub にログインする

ブラウザに本のコードリポジトリの [address](https://github.com/d2l-ai/d2l-en/) を入力します。:numref:`fig_git_fork`の右上にある赤いボックス内の`Fork`ボタンをクリックして、この本のリポジトリのコピーを作成します。これは*あなたのコピー*になり、好きなように変更することができます。 

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`

これで、この本のコードリポジトリは、:numref:`fig_git_forked`の左上に表示されている`astonzhang/d2l-en`のように、あなたのユーザー名にフォーク（つまり、コピー）されます。 

![The forked code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### リポジトリのクローンを作成する

リポジトリをクローンする (つまり、ローカルコピーを作成する) には、リポジトリのアドレスを取得する必要があります。:numref:`fig_git_clone`の緑色のボタンは、これを表示します。このフォークを長く保持する場合は、ローカルコピーがメインリポジトリで最新であることを確認してください。とりあえずは、:ref:`chap_installation`の指示に従って始めてください。主な違いは、リポジトリの「自分のフォーク」をダウンロードしていることです。 

![Cloning the repository.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```

### 編集とプッシュ

今度は本を編集する時です。:numref:`sec_jupyter`の指示に従って、Jupyter ノートブックで編集するのが最善です。変更を加えて、問題ないことを確認します。ファイル `~/d2l-en/chapter_appendix_tools/how-to-contribute.md` のタイプミスを修正したと仮定します。その後、変更したファイルを確認できます。 

この時点で、Git は `chapter_appendix_tools/how-to-contribute.md` ファイルが変更されたことを知らせます。

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```

これが目的であることを確認したら、次のコマンドを実行します。

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

変更したコードは、リポジトリの個人用フォークに保存されます。変更の追加をリクエストするには、本の公式リポジトリのプルリクエストを作成する必要があります。 

### プルリクエストを送信する

:numref:`fig_git_newpr`に示すように、GitHubのリポジトリのフォークに移動し、「新しいプルリクエスト」を選択します。これにより、編集内容と本のメインリポジトリの最新版との間の変更点を示す画面が開きます。 

![New pull request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`

最後に、:numref:`fig_git_createpr`に示すようにボタンをクリックしてプルリクエストを送信します。プルリクエストで行った変更を必ず説明してください。これにより、著者はそれをレビューしたり、本とマージしたりするのが簡単になります。変更によっては、これがすぐに承認されたり、拒否されたり、変更に関するフィードバックが得られる可能性が高くなります。それらを組み込んだら、準備完了です。 

![Create pull request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

## まとめ

* GitHub を使ってこの本に貢献できます。
* GitHub のファイルを直接編集して、小さな変更を加えることができます。
* 大きな変更については、リポジトリをフォークし、ローカルで編集し、準備ができてから貢献してください。
* プルリクエストは、コントリビューションがどのようにまとめられているかです。大量のプルリクエストを送信しないようにしてください。理解し取り込むのが難しくなるからです。小さいものをいくつか送ったほうがいいです。

## 演習

1. `d2l-ai/d2l-en` リポジトリにスターを付けてフォークします。
1. 改善が必要なもの (参照がないなど) を見つけたら、プルリクエストを送信します。 
1. 通常は、新しいブランチを使用してプルリクエストを作成する方が良い方法です。[Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)でそれを行う方法を学んでください。

[Discussions](https://discuss.d2l.ai/t/426)

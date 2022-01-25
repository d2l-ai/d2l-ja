# ドキュメンテーション

:begin_tab:`mxnet`
この本の長さには制約があるため、すべての MXNet 関数とクラスを導入することはできません (また、そうしたくもないでしょう)。API ドキュメントと追加のチュートリアルと例には、本書以外にも多くのドキュメントが用意されています。このセクションでは、MXNet API を探索するためのガイダンスを提供します。
:end_tab:

:begin_tab:`pytorch`
この本の長さに制約があるため、PyTorch の関数やクラスをひとつひとつ紹介することはできないでしょう (そして皆さんもそうしたくありません)。API ドキュメントと追加のチュートリアルと例には、本書以外にも多くのドキュメントが用意されています。このセクションでは PyTorch API を探索するためのガイダンスを提供します。
:end_tab:

:begin_tab:`tensorflow`
この本の長さには制約があるため、TensorFlow のすべての関数とクラスを紹介することはできません (おそらくそうしたくないでしょう)。API ドキュメントと追加のチュートリアルと例には、本書以外にも多くのドキュメントが用意されています。このセクションでは、TensorFlow API を探索するためのガイダンスを提供します。
:end_tab:

## モジュール内のすべての関数とクラスを検索する

モジュール内で呼び出せる関数とクラスを知るために、`dir` 関数を呼び出します。たとえば、次のようにします (**乱数を生成するためにモジュール内のすべてのプロパティを照会する**)。

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

一般に、`__` で開始および終了する関数 (Python では特殊オブジェクト) や、単一の `_` で始まる関数 (通常は内部関数) は無視できます。残りの関数名または属性名からすると、このモジュールは一様分布 (`uniform`)、正規分布 (`normal`)、多項分布 (`multinomial`) からのサンプリングなど、乱数を生成するためのさまざまな方法を提供していると推測されるかもしれません。 

## 特定の関数とクラスの使い方を調べる

特定の関数またはクラスの使用方法に関するより具体的な指示については、`help` 関数を呼び出すことができます。例として、[**テンソルの`ones`関数の使用方法を探る**]。

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

ドキュメントを見ると、関数 `ones` は指定された形状を持つ新しいテンソルを作成し、すべての要素を値 1 に設定していることがわかります。可能な限り、解釈を確認するために (**クイックテストを実行**) してください。

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

Jupyter ノートブックでは、`?`をクリックすると、ドキュメントが別のウィンドウに表示されます。たとえば、`list?`は `help(list)` とほとんど同じ内容を作成し、新しいブラウザウィンドウに表示します。また、`list?? のように 2 つの疑問符を使うと`の場合、その関数を実装している Python コードも表示されます。 

## [概要

* 公式ドキュメントには、本書にはない説明や例が数多く記載されています。
* `dir` 関数と `help` 関数、または `?` and `？？`Jupyter ノートブックで。

## 演習

1. ディープラーニングフレームワーク内の関数またはクラスのドキュメンテーションを調べます。フレームワークの公式ウェブサイトでもドキュメントを見つけることができますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

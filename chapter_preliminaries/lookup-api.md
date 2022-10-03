```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# ドキュメンテーション

:begin_tab:`mxnet`
すべての MXNet 関数とクラスを紹介することはできませんが (情報がすぐに古くなる可能性もあります)、[API documentation](https://mxnet.apache.org/versions/1.8.0/api) と追加の [tutorials](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) と例でこのようなドキュメントが提供されています。このセクションでは、MXNet API の探索方法に関するガイダンスを提供します。
:end_tab:

:begin_tab:`pytorch`
すべてのPyTorch関数とクラスを紹介することはできませんが（情報がすぐに古くなるかもしれません）、[API documentation](https://pytorch.org/docs/stable/index.html)と追加の[tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)と例はそのようなドキュメントを提供します。このセクションでは、PyTorch API を探索する方法についていくつかのガイダンスを提供します。
:end_tab:

:begin_tab:`tensorflow`
すべての TensorFlow 関数とクラスを導入することはできませんが (情報がすぐに古くなる可能性もあります)、[API documentation](https://www.tensorflow.org/api_docs) と追加の [tutorials](https://www.tensorflow.org/tutorials) と例でこのようなドキュメントが提供されています。このセクションでは、TensorFlow API を探索する方法についていくつかのガイダンスを提供します。
:end_tab:

## モジュール内の関数とクラス

モジュール内で呼び出せる関数とクラスを知るために、`dir` 関数を呼び出します。例えば、(**乱数を生成するためにモジュール内のすべてのプロパティを照会する**):

```{.python .input  n=1}
%%tab mxnet
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

一般に、`__` で開始および終了する関数 (Python では特別なオブジェクト) や、単一の `_` で始まる関数 (通常は内部関数) は無視できます。残りの関数名または属性名に基づいて、このモジュールが一様分布 (`uniform`)、正規分布 (`normal`)、および多項分布 (`multinomial`) からのサンプリングを含む、乱数を生成するためのさまざまな方法を提供していると推測するのは危険かもしれません。 

## 特定の関数とクラス

特定の関数またはクラスの使用方法に関するより具体的な手順については、`help` 関数を呼び出すことができます。例として、[**テンソルの`ones`関数の使用方法を調べる**] してみましょう。

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

ドキュメントから、`ones`関数が指定された形状で新しいテンソルを作成し、すべての要素を1の値に設定することがわかります。可能な限り、解釈を確認するために（**クイックテストを実行**）する必要があります。

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

Jupyter ノートブックでは、`?`は、ドキュメントを別のウィンドウに表示します。たとえば、`list?`は `help(list)` とほぼ同じコンテンツを作成し、新しいブラウザウィンドウに表示します。さらに、`list?? のように2つの疑問符を使うと、`、関数を実装するPythonコードも表示されます。 

公式ドキュメントには、この本以外の多くの説明と例が記載されています。私たちの重点は、カバレッジの完全性ではなく、実際的な問題を迅速に開始できるようにする重要なユースケースをカバーすることにあります。また、ライブラリのソースコードを調べて、プロダクションコードの高品質実装の例を確認することをお勧めします。そうすることで、あなたはより優れた科学者になるだけでなく、より優れたエンジニアにもなるでしょう。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

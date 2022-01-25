# 非同期計算
:label:`sec_async`

今日のコンピューターは高度に並列化されたシステムであり、複数の CPU コア (多くの場合、コアごとに複数のスレッド)、GPU ごとに複数の処理要素、およびデバイスごとに複数の GPU で構成されます。つまり、多くの異なるものを同時に、多くの場合異なるデバイスで処理できます。残念なことに、Python は並列で非同期なコードを書くための素晴らしい方法ではありません。少なくとも特別な助けがなければ、そうではありません。結局のところ、Python はシングルスレッドであり、将来変更される可能性は低いです。MXNet や TensorFlow などのディープラーニングフレームワークでは、パフォーマンスを向上させるために*非同期プログラミング* モデルを採用していますが、PyTorch は Python 独自のスケジューラを使用しており、パフォーマンスのトレードオフが異なります。PyTorch では、デフォルトで GPU 操作は非同期です。GPU を使用する関数を呼び出すと、操作は特定のデバイスにキューに入れられますが、後になるまで実行されるとは限りません。これにより、CPU や他の GPU での操作を含め、より多くの計算を並行して実行できるようになります。 

したがって、非同期プログラミングの仕組みを理解することは、計算要件と相互依存関係を積極的に減らすことで、より効率的なプログラムを開発するのに役立ちます。これにより、メモリのオーバーヘッドを削減し、プロセッサの使用率を高めることができます。

```{.python .input}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## バックエンドによる非同期

:begin_tab:`mxnet`
ウォームアップのために、次のおもちゃの問題を考えてみましょう。乱数行列を生成して乗算します。違いを確認するために、NumPy と `mxnet.np` の両方でそれを行いましょう。
:end_tab:

:begin_tab:`pytorch`
ウォームアップのために、次のおもちゃの問題を考えてみましょう。乱数行列を生成して乗算します。違いを確認するために、NumPy と PyTorch テンソルの両方でそれを行いましょう。PyTorch `tensor` は GPU 上で定義されていることに注意してください。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet を介したベンチマーク出力は、桁違いに高速になりました。両方とも同じプロセッサ上で実行されるので、何か他のことが起こっていなければなりません。戻る前に MXNet にバックエンドの計算をすべて終了させると、以前に何が起きたかがわかります。フロントエンドが Python に制御を返す間、計算はバックエンドによって実行されます。
:end_tab:

:begin_tab:`pytorch`
PyTorch によるベンチマーク出力は桁違いに速くなります。NumPy ドット積は CPU プロセッサで実行され、PyTorch 行列乗算は GPU で実行されるため、後者の方がはるかに高速になることが期待されます。しかし、大きな時差は、何か他のことが起こっているに違いないことを示唆しています.PyTorch では、デフォルトで GPU 操作は非同期です。PyTorch を返す前にすべての計算を強制的に終了させると、以前に何が起きたかがわかります。フロントエンドが Python に制御を返す間、計算はバックエンドによって実行されています。
:end_tab:

```{.python .input}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
大まかに言って、MXNet には Python などを介してユーザーと直接対話するためのフロントエンドと、システムが計算を実行するために使用するバックエンドがあります。:numref:`fig_frontends` に示すように、ユーザーは Python、R、Scala、C++ などのさまざまなフロントエンド言語で MXNet プログラムを書くことができます。使用するフロントエンドプログラミング言語に関係なく、MXNet プログラムの実行は主に C++ 実装のバックエンドで行われます。フロントエンド言語によって発行された操作は、バックエンドに渡されて実行されます。バックエンドは、キューに入れられたタスクを継続的に収集して実行する独自のスレッドを管理します。これを機能させるには、バックエンドがコンピュテーショナルグラフのさまざまなステップ間の依存関係を追跡できなければならないことに注意してください。したがって、相互に依存する演算を並列化することはできません。
:end_tab:

:begin_tab:`pytorch`
大まかに言うと、PyTorch はユーザと直接対話するためのフロントエンド (例えば Python 経由) と、システムが計算を実行するために使用するバックエンドを持っています。:numref:`fig_frontends` に示されているように、ユーザーは PyTorch プログラムを Python や C++ などのさまざまなフロントエンド言語で書くことができます。使用するフロントエンドプログラミング言語に関係なく、PyTorch プログラムの実行は主に C++ 実装のバックエンドで行われます。フロントエンド言語によって発行された操作は、バックエンドに渡されて実行されます。バックエンドは、キューに入れられたタスクを継続的に収集して実行する独自のスレッドを管理します。これを機能させるには、バックエンドがコンピュテーショナルグラフのさまざまなステップ間の依存関係を追跡できなければならないことに注意してください。したがって、相互に依存する演算を並列化することはできません。
:end_tab:

![Programming language frontends and deep learning framework backends.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

ディペンデンシーグラフをもう少しよく理解するために、別のおもちゃの例を見てみましょう。

```{.python .input}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![The backend tracks dependencies between various steps in the computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

上記のコードスニペットは :numref:`fig_asyncgraph` にも示されています。Python フロントエンドスレッドが最初の 3 つのステートメントのうちの 1 つを実行するたびに、タスクをバックエンドキューに返すだけです。最後のステートメントの結果を*printed* する必要がある場合、Python フロントエンドスレッドは C++ バックエンドスレッドが変数 `z` の結果の計算を終了するのを待ちます。この設計の利点の 1 つは、Python フロントエンドスレッドが実際の計算を実行する必要がないことです。したがって、Python のパフォーマンスにかかわらず、プログラム全体のパフォーマンスにほとんど影響はありません。:numref:`fig_threading` は、フロントエンドとバックエンドがどのように相互作用するかを示しています。 

![Interactions of the frontend and backend.](../img/threading.svg)
:label:`fig_threading`

## バリアとブロッカー

:begin_tab:`mxnet`
Python に強制的に完了を待たせるような操作がいくつかあります。 

* ほとんどの場合、`npx.waitall()` は、計算命令がいつ発行されたかにかかわらず、すべての計算が完了するまで待機します。実際には、この演算子を使用するとパフォーマンスが低下する可能性があるため、絶対に必要な場合以外は使用しないでください。
* 特定の変数が利用可能になるまで待ちたい場合は `z.wait_to_read()` を呼び出します。この場合、変数 `z` が計算されるまで MXNet ブロックは Python に戻ります。他の計算はその後も続けられるかもしれません。

これが実際にどのように機能するか見てみましょう。
:end_tab:

```{.python .input}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
両方の操作が完了するまでにかかる時間はほぼ同じです。明らかなブロッキング操作の他に、*implicit* ブロッカーにも注意することをお勧めします。変数を出力するには、その変数が使用可能であることを明確に要求するため、ブロッカになります。最後に、`z.asnumpy()` を介した NumPy への変換と `z.item()` を介したスカラーへの変換はブロックされます。NumPy には非同期の概念がないためです。`print` 関数と同様に値にアクセスする必要があります。  

少量のデータを MXNet のスコープから NumPy に頻繁にコピーしたり、その逆にコピーしたりすると、効率の悪いコードのパフォーマンスが損なわれる可能性があります。このような操作を行うたびに、他の処理を行う前に、関連する用語を取得するために必要なすべての中間結果を計算グラフで評価する必要があるためです。
:end_tab:

```{.python .input}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 計算の向上

:begin_tab:`mxnet`
マルチスレッドの多いシステム (通常のラップトップでも 4 スレッド以上あり、マルチソケットサーバーではこの数が 256 を超える場合がある) では、スケジューリング操作のオーバーヘッドが大きくなる可能性があります。このため、計算とスケジューリングを非同期かつ並列に実行することが非常に望ましいのです。そうすることの利点を説明するために、変数を順次または非同期で複数回インクリメントするとどうなるかを見てみましょう。各加算の間に `wait_to_read` バリアを挿入することで、同期実行をシミュレートします。
:end_tab:

```{.python .input}
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python フロントエンドスレッドと C++ バックエンドスレッドの間のやりとりを少し単純化すると、以下のように要約できます。
1. フロントエンドは、計算タスク `y = x + 1` をキューに挿入するようバックエンドに命令します。
1. その後、バックエンドはキューから計算タスクを受け取り、実際の計算を実行します。
1. その後、バックエンドは計算結果をフロントエンドに返します。
これら 3 つのステージの期間がそれぞれ $t_1, t_2$ と $t_3$ であると仮定します。非同期プログラミングを使用しない場合、10000 回の計算の実行にかかる合計時間は約 $10000 (t_1+ t_2 + t_3)$ になります。非同期プログラミングを使用すると、10000 回の計算の実行にかかる合計時間を $t_1 + 10000 t_2 + t_3$ に短縮できます ($10000 t_2 > 9999t_1$ と仮定)。フロントエンドは、バックエンドがループごとに計算結果を返すのを待つ必要がないためです。
:end_tab:

## [概要

* ディープラーニングフレームワークは、Python フロントエンドを実行バックエンドから切り離す場合があります。これにより、バックエンドへのコマンドの高速な非同期挿入とそれに関連する並列処理が可能になります。
* 非同期は、応答性の高いフロントエンドにつながります。ただし、タスクキューがいっぱいになりすぎると、メモリが過剰に消費される可能性があるため、注意が必要です。フロントエンドとバックエンドをほぼ同期させるために、ミニバッチごとに同期することをお勧めします。
* チップベンダーは、ディープラーニングの効率性についてよりきめ細かな洞察を得るために、高度なパフォーマンス分析ツールを提供しています。

:begin_tab:`mxnet`
* MXNet のメモリ管理から Python への変換では、特定の変数の準備が整うまでバックエンドが強制的に待たされることに注意してください。`print`、`asnumpy`、`item` などの関数はすべてこの効果を持ちます。これは望ましいことですが、不注意に同期を使用するとパフォーマンスが低下する可能性があります。
:end_tab:

## 演習

:begin_tab:`mxnet`
1. 前述したように、非同期計算を使用すると、10000 回の計算を実行するのに必要な合計時間を $t_1 + 10000 t_2 + t_3$ に短縮できます。なぜここで$10000 t_2 > 9999 t_1$を想定しなければならないのですか？
1. `waitall` と `wait_to_read` の差を測定します。ヒント:いくつかの命令を実行し、同期して中間結果を得ます。
:end_tab:

:begin_tab:`pytorch`
1. CPU で、このセクションで同じ行列乗算演算をベンチマークします。バックエンド経由で非同期性を監視できますか？
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab:

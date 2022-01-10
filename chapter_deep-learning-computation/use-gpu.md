# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade` では、過去 20 年間にわたる計算の急速な成長について議論しました。一言で言えば、GPU のパフォーマンスは 2000 年以降 10 年ごとに 1000 倍に向上しています。これは大きなチャンスをもたらしますが、そのようなパフォーマンスを提供する必要性が非常に高いことも示唆しています。 

このセクションでは、この計算性能を研究に活用する方法について説明します。まず、単一の GPU を使用し、後で、複数の GPU と (複数の GPU を持つ) 複数のサーバーの使用方法について説明します。 

具体的には、単一の NVIDIA GPU を計算に使用する方法について説明します。まず、NVIDIA GPU が少なくとも 1 つインストールされていることを確認します。次に [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads) をダウンロードし、プロンプトに従って適切なパスを設定します。これらの準備が完了したら、`nvidia-smi` コマンドを使用して (**グラフィックスカード情報を表示**) できます。

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
MXNet テンソルが NumPy `ndarray` とほとんど同じに見えることに気付いたかもしれません。しかし、いくつかの重要な違いがあります。MXNet と NumPy を区別する重要な機能の 1 つは、多様なハードウェアデバイスのサポートです。 

MXNet では、すべての配列にコンテキストがあります。これまでのところ、デフォルトではすべての変数とそれに関連する計算が CPU に割り当てられています。通常、他のコンテキストはさまざまな GPU です。複数のサーバーにジョブを展開すると、事態はさらに困難になります。配列をコンテキストにインテリジェントに割り当てることで、デバイス間でのデータ転送にかかる時間を最小限に抑えることができます。たとえば、GPU を搭載したサーバーでニューラルネットワークを学習させる場合、通常、モデルのパラメーターは GPU 上に存在することを好みます。 

次に、GPU バージョンの MXNet がインストールされていることを確認する必要があります。CPU バージョンの MXNet が既にインストールされている場合は、先にアンインストールする必要があります。たとえば、`pip uninstall mxnet` コマンドを使用して、使用している CUDA のバージョンに応じて、対応する MXNet バージョンをインストールします。CUDA 10.0 がインストールされている場合、CUDA 10.0 をサポートする MXNet バージョンを `pip install mxnet-cu100` 経由でインストールできます。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、すべての配列にデバイスがあり、コンテキストとして参照することがよくあります。これまでのところ、デフォルトではすべての変数とそれに関連する計算が CPU に割り当てられています。通常、他のコンテキストはさまざまな GPU です。複数のサーバーにジョブを展開すると、事態はさらに困難になります。配列をコンテキストにインテリジェントに割り当てることで、デバイス間でのデータ転送にかかる時間を最小限に抑えることができます。たとえば、GPU を搭載したサーバーでニューラルネットワークを学習させる場合、通常、モデルのパラメーターは GPU 上に存在することを好みます。 

次に、GPU バージョンの PyTorch がインストールされていることを確認する必要があります。PyTorch の CPU バージョンが既にインストールされている場合は、まずそれをアンインストールする必要があります。たとえば、`pip uninstall torch` コマンドを使用し、CUDA のバージョンに応じて対応する PyTorch のバージョンをインストールします。CUDA 10.0 がインストールされていると仮定すると、CUDA 10.0 をサポートする PyTorch バージョンを `pip install torch-cu100` 経由でインストールできます。
:end_tab:

このセクションのプログラムを実行するには、少なくとも 2 つの GPU が必要です。これはほとんどのデスクトップコンピューターでは贅沢ですが、AWS EC2 マルチ GPU インスタンスを使用するなどして、クラウドで簡単に利用できます。他のほとんどのセクションでは、複数の GPU を必要としません。これは、異なるデバイス間でデータがどのように流れるかを示すためだけのものです。 

## [**コンピューティングデバイス**]

CPU や GPU などのデバイスを、ストレージと計算用に指定できます。デフォルトでは、テンソルはメインメモリに作成され、CPU を使用してテンソルが計算されます。

:begin_tab:`mxnet`
MXnet では、CPU と GPU は `cpu()` と `gpu()` で表すことができます。`cpu()` (または括弧内の任意の整数) は、すべての物理 CPU とメモリを意味することに注意してください。つまり、MXNet の計算ではすべての CPU コアを使用しようとします。ただし、`gpu()` は 1 つのカードとそれに対応するメモリだけを表します。複数の GPU がある場合は、`gpu(i)` を使用して $i^\mathrm{th}$ GPU を表します ($i$ は 0 から始まります)。また、`gpu(0)` と `gpu()` は同等です。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、CPU と GPU は `torch.device('cpu')` と `torch.device('cuda')` で示すことができます。`cpu` デバイスとは、すべての物理 CPU とメモリを意味することに注意してください。つまり、PyTorch の計算ではすべての CPU コアを使おうとします。ただし、`gpu` デバイスは 1 つのカードとそれに対応するメモリのみを表します。複数の GPU がある場合は、`torch.device(f'cuda:{i}')` を使用して $i^\mathrm{th}$ GPU を表します ($i$ は 0 から始まります)。また、`gpu:0` と `gpu` は同等です。
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

私たちはできる (**利用可能な GPU の数を問い合わせる**)

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

ここで [**要求された GPU が存在しなくてもコードを実行できる、便利な関数を 2 つ定義します**]

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## テンソルと GPU

デフォルトでは、テンソルは CPU 上に作成されます。[**テンソルが位置するデバイスを問い合わせる**]

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

複数の用語で操作する場合は、それらを同じデバイス上に配置する必要があることに注意することが重要です。たとえば、2 つのテンソルを合計する場合、両方の引数が同じデバイス上に存在することを確認する必要があります。そうしないと、フレームワークは結果を格納する場所や、計算を実行する場所の決定方法さえも認識しません。 

### GPU 上のストレージ

[**テンソルをGPUに格納する**] にはいくつかの方法があります。たとえば、テンソル作成時にストレージデバイスを指定できます。次に、最初の `gpu` にテンソル変数 `X` を作成します。GPU で作成されたテンソルは、この GPU のメモリのみを消費します。`nvidia-smi` コマンドを使用して GPU メモリ使用量を表示できます。一般に、GPU メモリ制限を超えるデータを作成しないようにする必要があります。

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

GPU が 2 つ以上あると仮定すると、次のコードは (**2 つ目の GPU にランダムなテンソルを作成する**)

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### コピー中

[**`X + Y` を計算するには、この操作を実行する場所を決める必要があります。**] たとえば :numref:`fig_copyto` に示すように、`X` を 2 番目の GPU に転送し、そこで演算を実行できます。
**単純に`X`と`Y`を加えないでください。
これは例外になるからです。ランタイムエンジンは何をすべきか分からず、同じデバイス上でデータを見つけることができず、失敗します。`Y` は 2 番目の GPU 上に存在するため、2 つ目の GPU を追加する前に `X` をそこに移動する必要があります。 

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

[**データは同じ GPU (`Z` と `Y`) 上にあるので、合計できます**]

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
変数 `Z` がすでに 2 つ目の GPU に存在するとします。それでも`Z.copyto(gpu(1))`に電話したらどうなるの？その変数が目的のデバイスにすでに存在していても、コピーを作成して新しいメモリを割り当てます。コードが実行されている環境によっては、2 つの変数が既に同じデバイス上に存在している場合があります。そのため、変数が現在異なるデバイスにある場合にのみコピーを作成します。このような場合、`as_in_ctx` を呼び出すことができます。変数が指定されたデバイスにすでに存在する場合、これは何もしません。特にコピーを作成する場合を除き、`as_in_ctx` が最適な方法です。
:end_tab:

:begin_tab:`pytorch`
変数 `Z` が 2 番目の GPU にすでに存在していると想像してください。それでも`Z.cuda(1)`に電話したらどうなるの？この関数は、コピーを作成して新しいメモリを割り当てる代わりに `Z` を返す。
:end_tab:

:begin_tab:`tensorflow`
変数 `Z` が 2 つ目の GPU にすでに存在しているとします。同じデバイススコープで `Z2 = Z` を呼び出した場合はどうなりますか？この関数は、コピーを作成して新しいメモリを割り当てる代わりに `Z` を返す。
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### サイドノート

GPUは高速であることを期待しているため、人々はGPUを機械学習に使用しています。しかし、デバイス間で変数を転送するのは遅いです。そのため、私たちがあなたにそれをさせる前に、ゆっくりとしたことをしたいということを100％確信してほしいのです。ディープラーニングフレームワークがクラッシュすることなく自動的にコピーを実行した場合、低速なコードを書いたことに気付かないかもしれません。 

また、デバイス (CPU、GPU、その他のマシン) 間でのデータ転送は、計算よりもはるかに遅くなります。また、処理を進める前にデータが送信される (または受信される) のを待たなければならないため、並列化が非常に困難になります。そのため、コピー操作は細心の注意を払って行う必要があります。経験則として、小規模な操作の多くは、1 つの大きな操作よりもはるかに悪いです。さらに、何をしているのか分からない限り、コードに散在する多くの単一操作よりも、一度に複数の操作を行う方がはるかに優れています。これは、あるデバイスが別の処理を実行する前に他のデバイスを待機しなければならない場合、そのような操作がブロックされる可能性があるためです。これは、電話で事前に注文して、準備ができたことを確認するのではなく、キューでコーヒーを注文するのと少し似ています。 

最後に、テンソルを出力したり、テンソルを NumPy 形式に変換したりするときに、データがメインメモリにない場合、フレームワークは最初にデータをメインメモリにコピーするため、転送オーバーヘッドが増えます。さらに悪いことに、今では恐ろしいグローバルインタプリタロックの影響を受けて、Python が完了するのをすべて待たせています。 

## [**ニューラルネットワークと GPU **]

同様に、ニューラルネットワークモデルでもデバイスを指定できます。次のコードは、モデルパラメーターを GPU に配置します。

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

次の章では、GPUでモデルを実行する方法の例をさらに多く見ていきます。これは、計算量がいくらか増えるためです。 

入力が GPU 上のテンソルの場合、モデルは同じ GPU で結果を計算します。

```{.python .input}
#@tab all
net(X)
```

(**モデルパラメータが同じ GPU に保存されていることを確認する**)

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

つまり、すべてのデータとパラメータが同じデバイス上にあれば、モデルを効率的に学習できます。次の章では、そのような例をいくつか見ていきます。 

## [概要

* CPU や GPU など、ストレージと計算のためのデバイスを指定できます。デフォルトでは、データはメインメモリに作成され、CPU を使用して計算します。
* ディープラーニングフレームワークでは、CPU でも同じ GPU でも、計算用のすべての入力データが同じデバイス上にある必要があります。
* 注意せずにデータを移動すると、パフォーマンスが大幅に低下する可能性があります。典型的な間違いは次のとおりです。GPU 上のすべてのミニバッチの損失を計算し、コマンドラインでユーザーに報告する (または NumPy `ndarray` に記録する) と、グローバルインタープリターロックがトリガーされ、すべての GPU が停止します。GPU 内のロギング用にメモリを割り当て、大きなログのみを移動する方がはるかに優れています。

## 演習

1. 大きな行列の乗算など、より大きな計算タスクを試して、CPU と GPU の速度の違いを確認します。計算量が少ないタスクについてはどうでしょうか。
1. GPU でモデルパラメーターをどのように読み書きすればよいのですか？
1. $100 \times 100$ 行列の 1000 個の行列と行列の乗算を計算するのにかかる時間を測定し、出力行列の Frobenius ノルムを一度に 1 つの結果ずつ記録するのに対し、GPU でログを保持して最終結果のみを転送するのに対し、ログを記録します。
1. 2 つの GPU で 2 つの行列行列乗算を同時に実行するのにかかる時間と、1 つの GPU で順番に掛け合わせるのにかかる時間を測定します。ヒント:ほぼ直線的なスケーリングが見えるはずです。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:

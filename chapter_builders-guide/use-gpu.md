```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade`では、過去20年間にわたる計算の急速な成長について議論しました。一言で言えば、GPUのパフォーマンスは2000年以降、10年ごとに1000倍に向上しています。これは素晴らしい機会を提供しますが、そのようなパフォーマンスを提供する必要性が非常に高いことも示唆しています。 

このセクションでは、この計算性能を研究に活用する方法について説明します。まず、単一のGPUを使用し、後で複数のGPUと複数のサーバー（複数のGPUを使用）を使用する方法について説明します。 

具体的には、計算に単一の NVIDIA GPU を使用する方法について説明します。まず、少なくとも 1 つの NVIDIA GPU がインストールされていることを確認します。次に、[NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)をダウンロードし、プロンプトに従って適切なパスを設定します。これらの準備が完了したら、`nvidia-smi` コマンドを使用して (**グラフィックスカード情報を表示**) できます。

```{.python .input}
%%tab all
!nvidia-smi
```

:begin_tab:`mxnet`
お気づきかもしれませんが、MXNet テンソルは NumPy `ndarray` とほとんど同じに見えます。しかし、いくつかの重要な違いがあります。MXNet と NumPy を区別する重要な機能の 1 つは、多様なハードウェアデバイスのサポートです。 

MXNet では、すべての配列にコンテキストがあります。これまでは、デフォルトで、すべての変数と関連する計算が CPU に割り当てられています。通常、他のコンテキストはさまざまな GPU です。複数のサーバーにジョブを展開すると、事態はさらに困難になる可能性があります。アレイをコンテキストにインテリジェントに割り当てることで、デバイス間のデータ転送にかかる時間を最小限に抑えることができます。たとえば、GPU を備えたサーバーでニューラルネットワークをトレーニングする場合、通常、モデルのパラメーターは GPU 上に存在することを好みます。 

次に、MXNet の GPU バージョンがインストールされていることを確認する必要があります。CPU バージョンの MXNet が既にインストールされている場合は、まずそれをアンインストールする必要があります。たとえば、`pip uninstall mxnet` コマンドを使用して、使用している CUDA のバージョンに応じて、対応する MXNet バージョンをインストールします。CUDA 10.0 がインストールされていると仮定すると、`pip install mxnet-cu100` を介して CUDA 10.0 をサポートする MXNet バージョンをインストールできます。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、すべての配列にデバイスがあり、私たちはしばしばそれをコンテキストと呼びます。これまでは、デフォルトで、すべての変数と関連する計算が CPU に割り当てられています。通常、他のコンテキストはさまざまな GPU です。複数のサーバーにジョブを展開すると、事態はさらに困難になる可能性があります。アレイをコンテキストにインテリジェントに割り当てることで、デバイス間のデータ転送にかかる時間を最小限に抑えることができます。たとえば、GPU を備えたサーバーでニューラルネットワークをトレーニングする場合、通常、モデルのパラメーターは GPU 上に存在することを好みます。
:end_tab:

このセクションのプログラムを実行するには、少なくとも 2 つの GPU が必要です。これはほとんどのデスクトップコンピューターでは贅沢かもしれませんが、AWS EC2 マルチ GPU インスタンスを使用するなどして、クラウドで簡単に利用できます。他のほとんどのセクションは、複数のGPUを必要としません。代わりに、これは単に異なるデバイス間でデータがどのように流れるかを説明するためです。 

## [**コンピューティングデバイス**]

ストレージや計算用に CPU や GPU などのデバイスを指定できます。デフォルトでは、テンソルはメインメモリに作成され、CPUを使用してそれを計算します。

:begin_tab:`mxnet`
MXNet では、CPU と GPU は `cpu()` と `gpu()` で示されます。`cpu()`（または括弧内の任意の整数）は、すべての物理CPUとメモリを意味することに注意してください。これは、MXNet の計算がすべての CPU コアを使用しようとすることを意味します。ただし、`gpu()`は、1つのカードと対応するメモリのみを表します。複数の GPU がある場合、`gpu(i)` を使用して $i^\mathrm{th}$ GPU を表します ($i$ は 0 から始まります)。また、`gpu(0)`と`gpu()`は同等です。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、CPU と GPU は `torch.device('cpu')` と `torch.device('cuda')` で示されます。`cpu`デバイスは、すべての物理CPUとメモリを意味することに注意してください。これは、PyTorch の計算がすべての CPU コアを使用しようとすることを意味します。ただし、`gpu`デバイスは、1つのカードと対応するメモリのみを表します。複数の GPU がある場合、`torch.device(f'cuda:{i}')` を使用して $i^\mathrm{th}$ GPU を表します ($i$ は 0 から始まります)。また、`gpu:0`と`gpu`は同等です。
:end_tab:

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab all
def cpu():  #@save
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('pytorch'):
        return torch.device('cpu')
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')

def gpu(i=0):  #@save
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('pytorch'):
        return torch.device(f'cuda:{i}')
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')

cpu(), gpu(), gpu(1)
```

できる (**利用可能な GPU の数を照会する**)

```{.python .input}
%%tab all
def num_gpus():  #@save
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('pytorch'):
        return torch.cuda.device_count()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))

num_gpus()
```

ここで、[**要求されたGPUが存在しなくてもコードを実行できる便利な関数を2つ定義します**]。

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## テンソルと GPU

デフォルトでは、テンソルは CPU 上に作成されます。[**テンソルが配置されているデバイスを照会できます。**]

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

複数の用語で操作したい場合は常に、同じデバイス上にある必要があることに注意することが重要です。たとえば、2つのテンソルを合計する場合、両方の引数が同じデバイス上に存在することを確認する必要があります。そうしないと、フレームワークは結果をどこに保存するか、または計算を実行する場所を決める方法さえも知りません。 

### GPU 上のストレージ

[**テンソルをGPUに保存する**] にはいくつかの方法があります。たとえば、テンソルを作成するときにストレージデバイスを指定できます。次に、最初の `gpu` にテンソル変数 `X` を作成します。GPU で作成されたテンソルは、この GPU のメモリのみを消費します。`nvidia-smi` コマンドを使用して GPU のメモリ使用量を表示できます。一般に、GPU メモリ制限を超えるデータを作成しないようにする必要があります。

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

少なくとも 2 つの GPU があると仮定すると、次のコードは次のようになります (**2 番目の GPU でランダムなテンソルを作成します。**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### コピー中

[**`X + Y`を計算する場合、この操作を実行する場所を決める必要があります。**] たとえば、:numref:`fig_copyto`に示すように、`X`を2番目のGPUに転送し、そこで操作を実行できます。
**単純に`X`と`Y`を追加しないでください。
これは例外になるからです。ランタイムエンジンは何をすべきか分からず、同じデバイス上でデータを見つけることができず、失敗します。`Y` は 2 つ目の GPU 上に存在するため、2 つを追加する前に `X` をそこに移動する必要があります。 

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

[**データは同じ GPU (`Z` と `Y` はどちらも) 上にあるので、これらを加算できます。**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
変数 `Z` がすでに 2 つ目の GPU に存在していると想像してください。まだ`Z.copyto(gpu(1))`を呼んだらどうなるの？その変数が目的のデバイスにすでに存在している場合でも、コピーを作成して新しいメモリを割り当てます。コードが実行されている環境によっては、2 つの変数がすでに同じデバイス上に存在している場合があります。そのため、変数が現在別のデバイスにある場合にのみコピーを作成します。このような場合は、`as_in_ctx`に電話することができます。変数が指定したデバイスにすでに存在する場合、これは何もしません。特にコピーを作成する場合を除き、`as_in_ctx`が最適な方法です。
:end_tab:

:begin_tab:`pytorch`
変数 `Z` がすでに 2 つ目の GPU に存在していると想像してください。まだ`Z.cuda(1)`を呼んだらどうなるの？コピーを作成して新しいメモリを割り当てる代わりに、`Z`を返します。
:end_tab:

:begin_tab:`tensorflow`
変数 `Z` がすでに 2 つ目の GPU に存在していると想像してください。同じデバイススコープでまだ `Z2 = Z` を呼び出すとどうなりますか？コピーを作成して新しいメモリを割り当てる代わりに、`Z`を返します。
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### サイドノート

人々は高速であることを期待しているため、機械学習を行うためにGPUを使用しています。しかし、デバイス間で変数を転送するのは遅いです。ですから、私たちがあなたにそれをさせる前に、あなたが何か遅いことをしたいということを100％確信してほしい。ディープラーニングフレームワークがクラッシュせずにコピーを自動的に実行しただけなら、遅いコードを書いたことに気付かないかもしれません。 

また、デバイス (CPU、GPU、その他のマシン) 間でのデータ転送は、計算よりもはるかに低速です。また、より多くの操作を進める前にデータが送信される（または受信される）のを待たなければならないため、並列化がはるかに困難になります。このため、コピー操作は細心の注意を払って行う必要があります。経験則として、多くの小規模な操作は、1つの大きな操作よりもはるかに悪いです。さらに、何をしているのか分からない限り、一度に複数の操作を行うと、コードに散在する多くの単一操作よりもはるかに優れています。これは、あるデバイスが他の何かを実行する前に他のデバイスを待たなければならない場合、そのような操作がブロックされる可能性があるためです。これは、電話で予約注文して、準備ができていることを確認するのではなく、順番待ちでコーヒーを注文するようなものです。 

最後に、テンソルを出力するか、テンソルをNumPy形式に変換するときに、データがメインメモリにない場合、フレームワークはまずそれをメインメモリにコピーし、その結果、追加の送信オーバーヘッドが発生します。さらに悪いことに、Pythonが完了するまですべてを待たせる、恐ろしいグローバルインタプリタロックの影響を受けます。 

## [**ニューラルネットワークとGPU**]

同様に、ニューラルネットワークモデルでもデバイスを指定できます。次のコードは、モデルパラメーターを GPU に配置します。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

次の章では、GPUでモデルを実行する方法の例をさらに多く見ていきます。これは、計算負荷がいくらか高くなるためです。 

入力が GPU 上のテンソルの場合、モデルは同じ GPU で結果を計算します。

```{.python .input}
%%tab all
net(X)
```

それでは (**モデルパラメータが同じ GPU に保存されていることを確認する**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

トレーナーに GPU をサポートさせてください。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

つまり、すべてのデータとパラメータが同じデバイス上にある限り、モデルを効率的に学習できます。次の章では、そのような例をいくつか見ていきます。 

## まとめ

* CPU や GPU など、ストレージや計算用のデバイスを指定できます。既定では、データはメインメモリに作成され、計算に CPU を使用します。
* ディープラーニングフレームワークでは、計算用のすべての入力データが CPU でも同じ GPU でも、同じデバイス上にある必要があります。
* 注意せずにデータを移動すると、パフォーマンスが大幅に低下する可能性があります。典型的な間違いは次のとおりです。GPU上のすべてのミニバッチの損失を計算し、コマンドラインでユーザーに報告する（またはNumPy `ndarray`に記録する）と、グローバルインタープリターロックがトリガーされ、すべてのGPUが停止します。GPU 内でロギング用のメモリを割り当て、より大きなログのみを移動する方がはるかに優れています。

## 演習

1. 大きな行列の乗算など、より大きな計算タスクを試して、CPUとGPUの速度の違いを確認してください。計算量が少ないタスクはどうですか？
1. GPU でモデルパラメーターをどのように読み書きすべきですか?
1. $100 \times 100$ 行列の 1000 行列と行列の乗算を計算するのにかかる時間を測定し、出力行列のフロベニウスノルムを一度に 1 つずつ記録します。対数を GPU に保持して最終結果のみを転送するのとは異なります。
1. 2 つの GPU で 2 つの行列-行列乗算を同時に実行するのにかかる時間と、1 つの GPU で連続して実行するのにかかる時間を測定します。ヒント:ほぼ直線的なスケーリングが見えるはずです。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:

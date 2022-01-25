# 自動並列処理
:label:`sec_auto_para`

ディープラーニングフレームワーク (MXNet や PyTorch など) は、バックエンドで計算グラフを自動的に作成します。計算グラフを使用することで、システムはすべての依存関係を認識し、相互に依存しない複数のタスクを選択的に並列に実行して速度を向上させることができます。たとえば、:numref:`sec_async` の :numref:`fig_asyncgraph` は 2 つの変数を個別に初期化します。したがって、システムはこれらを並行して実行することを選択できます。 

通常、1 人のオペレータが、すべての CPU または 1 つの GPU 上のすべての計算リソースを使用します。たとえば、`dot` オペレータは、1 台のマシンに複数の CPU プロセッサがある場合でも、すべての CPU のすべてのコア (およびスレッド) を使用します。同じことが単一の GPU にも当てはまります。したがって、並列化は単一デバイスのコンピューターではあまり役に立ちません。複数のデバイスでは、事態がさらに重要になります。通常、並列化は複数の GPU 間で最も重要ですが、ローカル CPU を追加するとパフォーマンスがわずかに向上します。たとえば、GPU と CPU を組み合わせたコンピュータービジョンモデルのトレーニングに重点を置いた :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` を参照してください。自動並列化フレームワークの利便性により、数行の Python コードで同じ目標を達成できます。より広い意味では、自動並列計算に関する議論は、CPUとGPUの両方を用いた並列計算と、計算と通信の並列化に重点を置いています。 

このセクションの実験を実行するには、少なくとも 2 つの GPU が必要であることに注意してください。

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU での並列計算

まず、テストする参照ワークロードを定義します。以下の `run` 関数は、`x_gpu1` と `x_gpu2` の 2 つの変数に割り当てられたデータを使用して、選択したデバイスで 10 個の行列-行列乗算を実行します。

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
次に、この関数をデータに適用します。キャッシングが結果に影響を与えないように、測定前にどちらかのデバイスに対してシングルパスを実行して、デバイスをウォームアップします。
:end_tab:

:begin_tab:`pytorch`
次に、この関数をデータに適用します。キャッシングが結果に影響しないように、測定の前にどちらか一方に対してシングルパスを実行してデバイスをウォームアップします。`torch.cuda.synchronize()` は、CUDA デバイス上のすべてのストリームのすべてのカーネルが完了するまで待機します。これは `device` 引数を取ります。この引数は、同期が必要なデバイスです。device 引数が `None` (デフォルト) の場合、`current_device()` で指定された現在のデバイスが使われます。
:end_tab:

```{.python .input}
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
両方のタスク間で `waitall` ステートメントを削除すると、システムは両方のデバイスで計算を自動的に並列化できます。
:end_tab:

:begin_tab:`pytorch`
両方のタスク間で `synchronize` ステートメントを削除すると、システムは両方のデバイスで計算を自動的に並列化できます。
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

上記の場合、ディープラーニングフレームワークはユーザーに代わって高度なコードを必要とせずに両方の GPU デバイスで計算を自動的にスケジュールするため、合計実行時間は各部分の合計よりも短くなります。 

## 並列計算と通信

多くの場合、CPUとGPU間、または異なるGPU間など、異なるデバイス間でデータを移動する必要があります。たとえば、複数のアクセラレータカードにわたって勾配を集約する必要がある分散最適化を実行したい場合に発生します。GPU で計算し、結果を CPU にコピーして戻すことで、これをシミュレートしてみましょう。

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
これはやや非効率的です。リストの残りの部分が計算されている間に、`y` の一部を CPU にコピーし始めることができたことに注意してください。この状況は、たとえばミニバッチで勾配を計算するときに発生します。一部のパラメーターのグラデーションは、他のパラメーターのグラデーションよりも早く利用できます。したがって、GPU がまだ実行されている間に PCI-Express バス帯域幅の使用を開始することが有利になります。両方のパーツ間で `waitall` を削除すると、このシナリオをシミュレートできます。
:end_tab:

:begin_tab:`pytorch`
これはやや非効率的です。リストの残りの部分が計算されている間に、すでに `y` の一部を CPU にコピーし始めることができたことに注意してください。この状況は、例えば、ミニバッチで (backprop) 勾配を計算するときに発生します。一部のパラメーターのグラデーションは、他のパラメーターのグラデーションよりも早く利用できます。したがって、GPU がまだ実行されている間に PCI-Express バス帯域幅の使用を開始することが有利になります。PyTorch では `to()` や `copy_()` のようないくつかの関数が明示的な `non_blocking` 引数を認めているため、呼び出し元は同期が不要なときにバイパスすることができます。`non_blocking=True` を設定すると、このシナリオをシミュレートできます。
:end_tab:

```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

両方の操作に必要な合計時間が (予想どおり)、パーツの合計よりも短くなっています。このタスクは、CPU と GPU 間のバスという異なるリソースを使用するため、並列計算とは異なります。実際、両方のデバイスで計算と通信を同時に行うことができました。前述のとおり、計算と通信には依存関係があります。`y[i]` は CPU にコピーする前に計算する必要があります。幸いなことに、`y[i]` の計算中に `y[i-1]` をコピーして、総実行時間を短縮できます。 

最後に、:numref:`fig_twogpu` に示すように、CPU と 2 つの GPU でトレーニングする場合の、単純な 2 層 MLP の計算グラフとその依存関係の図を示します。この結果生じる並列プログラムを手動でスケジュールするのはかなり面倒です。これは、最適化のためにグラフベースのコンピューティングバックエンドを持つことが有利な点です。 

![The computational graph and its dependencies of a two-layer MLP on a CPU and two GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`

## [概要

* 最新のシステムには、複数の GPU や CPU など、さまざまなデバイスがあります。これらは並行して非同期的に使用できます。 
* また、最新のシステムには、PCI Express、ストレージ (通常はソリッドステートドライブまたはネットワーク経由)、ネットワーク帯域幅など、さまざまな通信リソースがあります。これらを並行して使用することで、効率を最大限に高めることができます。 
* バックエンドは、自動並列計算と通信によってパフォーマンスを向上させることができます。 

## 演習

1. このセクションで定義されている `run` 関数では、8 つの操作が実行されました。両者の間には依存関係はありません。ディープラーニングフレームワークがこれらを並列に自動的に実行するかどうかを確認する実験を設計します。
1. 個々のオペレータのワークロードが十分に小さい場合は、単一の CPU または GPU でも並列化が役立ちます。これを検証する実験を計画します。 
1. CPU、GPU、および両方のデバイス間の通信で並列計算を使用する実験を設計します。
1. NVIDIA の [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) などのデバッガーを使用して、コードが効率的であることを確認します。 
1. より複雑なデータ依存関係を含む計算タスクを設計し、実験を実行して、パフォーマンスを向上させながら正しい結果が得られるかどうかを確認します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:

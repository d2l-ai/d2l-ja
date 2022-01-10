# 学習率スケジューリング
:label:`sec_scheduler`

これまでは、重みベクトルが更新される*速度*ではなく、重みベクトルの更新方法に関する最適化*アルゴリズム*に主に注目しました。とはいえ、学習率の調整は実際のアルゴリズムと同じくらい重要な場合がよくあります。考慮すべき点は多数あります。 

* 最も明らかなのは、学習率の*大きさ*が重要です。大きすぎると最適化が発散し、小さすぎるとトレーニングに時間がかかりすぎたり、最適ではない結果になってしまいます。以前、問題の条件番号が重要であることがわかりました (詳細については :numref:`sec_momentum` を参照)。直感的には、最も感度の低い方向と最も感度の高い方向の変化量の比率です。
* 第二に、減衰率も同様に重要です。学習率が大きいままであれば、単純に最小値付近で跳ね返り、最適性に達しない可能性があります。:numref:`sec_minibatch_sgd` ではこれについて詳しく説明し、:numref:`sec_sgd` で性能保証を分析しました。要するに、レートを減衰させたいが、$\mathcal{O}(t^{-\frac{1}{2}})$ よりも遅くなることが望ましく、凸問題には良い選択です。
* 同様に重要なもう 1 つの側面は、*初期化* です。これは、パラメーターの初期設定 (詳細については :numref:`sec_numerical_stability` を参照) と、パラメーターの初期設定方法の両方に関係します。これは、*ウォームアップ*、つまり最初にソリューションに向かってどれだけ早く動き始めるかというモニカに該当します。最初の大きなステップは、特にパラメーターの初期セットがランダムであるため、有益ではない場合があります。最初の更新の指示も全く意味がないかもしれません。
* 最後に、周期的な学習率調整を実行する最適化バリアントがいくつかあります。これは現在の章の範囲外です。読者には :cite:`Izmailov.Podoprikhin.Garipov.ea.2018` の詳細、例えば、パラメーターの*パス* 全体を平均化してよりよい解を得る方法などを確認することをお勧めします。

学習率を管理するには多くの詳細が必要であることを考えると、ほとんどのディープラーニングフレームワークにはこれを自動的に処理するツールがあります。この章では、異なるスケジュールが精度に及ぼす影響を確認し、*学習率スケジューラ*によってこれを効率的に管理する方法についても説明します。 

## おもちゃ問題

まず、簡単に計算できるほど安価でありながら、重要な側面のいくつかを説明するのに十分自明ではないおもちゃの問題から始めます。そのために、ファッションMNISTに適用されるように、少し近代化されたバージョンのLeNet（`sigmoid`アクティベーションの代わりに`relu`、AveragePoolingではなくmaxPooling）を選択します。さらに、パフォーマンスのためにネットワークをハイブリダイズします。ほとんどのコードは標準であるため、詳細な説明は不要で、基本を紹介します。必要に応じて復習については :numref:`chap_cnn` を参照してください。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    
        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the 
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0, 
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

このアルゴリズムを既定の設定 (学習率 $0.3$、反復数 $30$ の学習など) で呼び出すとどうなるかを見てみましょう。テスト精度の進歩が一点を超えて停滞する一方で、トレーニングの精度が上昇し続けることに注目してください。両方のカーブ間のギャップは過適合を示しています。

```{.python .input}
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## スケジューラ

学習率を調整する 1 つの方法は、各ステップで明示的に設定することです。これは `set_learning_rate` メソッドによって簡単に実現できます。最適化の進行状況に応じて動的な方法で、エポックごとに (またはすべてのミニバッチの後でも)、下方に調整することができます。

```{.python .input}
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

より一般的には、スケジューラを定義したいと考えています。更新回数を指定して起動すると、適切な学習率の値を返します。学習率を $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$ に設定する単純なものを定義しましょう。

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

値の範囲にわたってその動作をプロットしてみましょう。

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

さて、これがFashion-MNISTのトレーニングにどのように役立つかを見てみましょう。スケジューラをトレーニングアルゴリズムの追加引数として指定するだけです。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

これは以前よりもかなり良く機能しました。目立つのは、カーブが以前よりもかなり滑らかだったことです。第二に、過適合が少なかった。残念ながら、特定の戦略が*理論*の過剰適合を少なくする理由については、十分に解決された質問ではありません。ステップサイズが小さくなるとパラメータがゼロに近くなるため単純になるという議論がいくつかあります。しかし、これは現象を完全に説明するものではありません。なぜなら、私たちは本当に早く止まるのではなく、単に学習率を緩やかに減らすだけだからです。 

## 政策

さまざまな学習率スケジューラをすべてカバーすることはできませんが、一般的なポリシーの概要を以下に示します。一般的な選択肢は、多項式減衰と区分的定数スケジュールです。さらに、コサイン学習率スケジュールは、いくつかの問題について経験的にうまく機能することがわかっています。最後に、いくつかの問題では、大きな学習率を使用する前にオプティマイザーをウォームアップすると効果的です。 

### ファクタースケジューラ

多項式減衰の代替法として、$\alpha \in (0, 1)$ の場合は $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ という乗法があります。学習率が妥当な下限を超えて減衰するのを防ぐため、更新式は $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$ に変更されることがよくあります。

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

これは、`lr_scheduler.FactorScheduler` オブジェクトを介して MXNet に組み込まれているスケジューラでも実現できます。ウォームアップ期間、ウォームアップモード (線形または一定)、必要な更新の最大数など、さらにいくつかのパラメータが必要です。今後は、組み込みスケジューラを必要に応じて使用し、ここではその機能についてのみ説明します。図に示すように、必要に応じて独自のスケジューラを作成するのはかなり簡単です。 

### 多要素スケジューラ

ディープネットワークに学習させるための一般的な戦略は、学習率を区分的に一定に保ち、一定量だけ頻繁に減少させることです。つまり、$s = \{5, 10, 20\}$ のようにレートを下げる回数を指定すると、$t \in s$ のたびに $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ が減少します。各ステップで値が半分になると仮定すると、次のように実装できます。

```{.python .input}
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler) 
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr
  
    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

この区分的定数学習率スケジュールの背後にある直感は、重みベクトルの分布で定常点に達するまで最適化を進めることができるというものです。それから（そしてそのときだけ）レートを下げて、より高品質のプロキシを良いローカル最小値まで下げます。以下の例は、これによってどのように若干優れた解が得られるかを示しています。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### 余弦スケジューラ

:cite:`Loshchilov.Hutter.2016` によって、やや複雑なヒューリスティックが提案されました。最初は、学習率を大幅に下げたくない場合があり、さらに、最終的には非常に小さい学習率を使用して解を「絞り込み」たい場合があるという観察に依存しています。この結果、$t \in [0, T]$ の範囲の学習率に対して次の関数形式をもつ余弦のようなスケジュールが得られます。 

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$

ここで、$\eta_0$ は初期学習レート、$\eta_T$ は時間 $T$ でのターゲットレートです。さらに、$t > T$ では、値を $\eta_T$ に固定するだけで、値を再び増やすことはありません。次の例では、最大更新ステップ $T = 20$ を設定します。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
  
    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

コンピュータビジョンの文脈では、このスケジュールは結果を改善する可能性があります*。ただし、このような改善は保証されないことに注意してください (下記参照)。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### ウォームアップ

場合によっては、パラメーターの初期化だけでは適切な解が得られないことがあります。これは特に、不安定な最適化問題を引き起こす可能性がある一部の高度なネットワーク設計では問題になります。最初の発散を防ぐために十分に小さい学習率を選択することで、これに対処できます。残念ながら、これは進行が遅いことを意味します。逆に、学習率が大きいと、最初は発散につながります。 

このジレンマに対する簡単な解決策は、学習率が初期最大値まで「増加」するウォームアップ期間を使用し、最適化プロセスが終了するまで速度を冷却することです。簡単にするために、通常、この目的のために線形増加を使用します。これにより、以下に示す形式のスケジュールが作成されます。

```{.python .input}
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

ネットワークの収束は最初は良好であることに注意してください (特に、最初の 5 つのエポックではパフォーマンスを観察します)。

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device, 
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

ウォームアップは (余弦だけでなく) どのスケジューラにも適用できます。学習率スケジュールの詳細およびその他多くの実験については、:cite:`Gotmare.Keskar.Xiong.ea.2018` も参照してください。特に、ウォームアップフェーズにより、非常に深いネットワークにおけるパラメーターの発散量が制限されることがわかりました。これは直感的に理解できます。なぜなら、最初に進行するのに最も時間がかかるネットワークの部分では、ランダムな初期化が原因で大幅な相違が予想されるからです。 

## [概要

* 学習中に学習率を下げると、精度が向上し、(最も厄介なことに) モデルの過適合が減少します。
* 実際には、進歩が頭打ちになったときに学習率を区分的に減少させることが効果的です。基本的に、これにより、適切な解に効率的に収束し、学習率を下げることによってパラメーターの固有の分散を減らすことが保証されます。
* コサインスケジューラは、一部のコンピュータビジョンの問題でよく使われます。このようなスケジューラの詳細は [GluonCV](http://gluon-cv.mxnet.io) などを参照のこと。
* 最適化の前にウォームアップ期間を設けると、相違を防ぐことができます。
* 最適化は、ディープラーニングにおいて複数の目的に役立ちます。学習目標を最小化する以外に、最適化アルゴリズムと学習率スケジューリングの選択が異なると、（同じ量の学習誤差に対して）テストセットの一般化と過適合の量がかなり異なる可能性があります。

## 演習

1. 所定の固定学習率に対する最適化動作を試します。この方法で入手できる最高のモデルは何ですか？
1. 学習率の低下の指数を変更すると、収束はどのように変化しますか？実験の便宜上、`PolyScheduler` を使用してください。
1. コサインスケジューラーは、ImageNet のトレーニングなど、大規模なコンピュータービジョンの問題に適用します。他のスケジューラと比較して、パフォーマンスにどのような影響がありますか？
1. ウォームアップはどれくらい続くべきですか？
1. 最適化とサンプリングを結びつけることはできますか？まず、確率的勾配ランジュバンダイナミクスの :cite:`Welling.Teh.2011` の結果を使用します。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:

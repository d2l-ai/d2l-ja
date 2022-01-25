# エンコーダ/デコーダアーキテクチャ
:label:`sec_encoder-decoder`

:numref:`sec_machine_translation` で説明したように、入力と出力の両方が可変長配列である配列変換モデルでは、機械翻訳は主要な問題領域です。このような入力と出力を処理するために、2 つの主要コンポーネントを持つアーキテクチャを設計できます。最初のコンポーネントは*encoder* です。これは可変長のシーケンスを入力として受け取り、固定された形状の状態に変換します。2 番目のコンポーネントは*decoder* で、固定形状のエンコードされた状態を可変長のシーケンスにマッピングします。これは*エンコーダ-デコーダ* アーキテクチャと呼ばれ、:numref:`fig_encoder_decoder` に示されています。 

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

英語からフランス語への機械翻訳を例に挙げてみましょう。英語の入力シーケンス:「彼ら」,「are」,「watching」,」。このエンコーダ/デコーダアーキテクチャは、まず可変長入力を状態にエンコードし、次に状態をデコードして、トークンごとに変換されたシーケンストークンを出力として生成します。「Ils」,「regardent」,「.」エンコーダ/デコーダアーキテクチャは以降のセクションで異なる配列変換モデルの基礎を形成するため、この節ではこのアーキテクチャをインタフェースに変換し、後で実装します。 

## (**エンコーダ**)

エンコーダインターフェイスでは、エンコーダが可変長シーケンスを入力 `X` として取るように指定するだけです。この実装は、この基本 `Encoder` クラスを継承するすべてのモデルによって提供されます。

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, X, *args, **kwargs):
        raise NotImplementedError
```

## [**デコーダ**]

次のデコーダーインターフェイスでは、エンコーダー出力 (`enc_outputs`) をエンコードされた状態に変換する `init_state` 関数を追加します。このステップでは、:numref:`subsec_mt_data_loading` で説明した入力の有効長など、追加の入力が必要になる場合があることに注意してください。可変長シーケンストークンをトークンごとに生成するには、デコーダが入力 (たとえば、前のタイムステップで生成されたトークン) とエンコードされた状態を現在のタイムステップで出力トークンにマッピングするたびに。

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state, **kwargs):
        raise NotImplementedError
```

## [**エンコーダとデコーダを組み合わせる**]

最終的に、エンコーダ/デコーダアーキテクチャには、エンコーダとデコーダの両方が含まれており、オプションで追加の引数があります。順伝播では、符号化器の出力を使用して符号化状態が生成され、この状態がさらにデコーダによって入力の 1 つとして使用されます。

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab tensorflow
#@save
class EncoderDecoder(tf.keras.Model):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)
```

エンコーダー/デコーダーアーキテクチャーにおける「状態」という用語は、状態をもつニューラルネットワークを使ってこのアーキテクチャーを実装するきっかけになったと思います。次のセクションでは、このエンコーダ/デコーダアーキテクチャに基づく配列変換モデルの設計にRNNを適用する方法を説明します。 

## [概要

* エンコーダ/デコーダアーキテクチャは、可変長シーケンスの入出力を処理できるため、機械翻訳などのシーケンス変換の問題に適しています。
* エンコーダは可変長シーケンスを入力として受け取り、固定形状の状態に変換します。
* デコーダは、固定形状の符号化状態を可変長シーケンスにマッピングする。

## 演習

1. ニューラルネットワークを使用してエンコーダー/デコーダーアーキテクチャを実装するとします。エンコーダとデコーダは同じタイプのニューラルネットワークでなければなりませんか？  
1. 機械翻訳以外に、エンコーダ/デコーダアーキテクチャを適用できる別のアプリケーションを考えられますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

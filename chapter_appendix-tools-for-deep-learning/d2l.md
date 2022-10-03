# `d2l` API ドキュメント
:label:`sec_d2l`

`d2l` パッケージの以下のメンバーの実装と、それらが定義され説明されているセクションは、[source file](https://github.com/d2l-ai/d2l-en/tree/master/d2l) にあります。

:begin_tab:`mxnet`
```eval_rst
.. currentmodule:: d2l.mxnet
```
:end_tab:

:begin_tab:`pytorch`
```eval_rst
.. currentmodule:: d2l.torch
```
:end_tab:

:begin_tab:`tensorflow`
```eval_rst
.. currentmodule:: d2l.torch
```
:end_tab:

## モデル

```eval_rst 
.. autoclass:: Module
   :members: 

.. autoclass:: LinearRegressionScratch
   :members:

.. autoclass:: LinearRegression
   :members:    

.. autoclass:: Classifier
   :members:
```

## データ

```eval_rst 
.. autoclass:: DataModule
   :members: 

.. autoclass:: SyntheticRegressionData
   :members: 

.. autoclass:: FashionMNIST
   :members:
```

## トレーナー

```eval_rst 
.. autoclass:: Trainer
   :members: 

.. autoclass:: SGD
   :members:
```

## ユーティリティ

```eval_rst 
.. autofunction:: add_to_class

.. autofunction:: cpu

.. autofunction:: gpu

.. autofunction:: num_gpus

.. autoclass:: ProgressBoard
   :members: 

.. autoclass:: HyperParameters
   :members:
```

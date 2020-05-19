#  線形なニューラルネットワーク
:label:`chap_linear`

ディープニューラルネットワークの詳細に入る前に、ニューラルネットワークの学習の基本について説明しておく必要があります。この章では、シンプルなニューラルネットワークアーキテクチャの定義、データの処理、損失関数の指定、モデルのトレーニングなど、トレーニングプロセス全体を取り上げます。より理解しやすくするために、最もシンプルな考えかたから始めます。幸い、線形回帰やロジスティック回帰などの古典的な統計学習手法は、*浅い*ニューラルネットワークとして考えることができます。そこで、これらの古典的なアルゴリズムから始めて、まずは基本を紹介し、softmax回帰（この章の最後で紹介）や多層パーセプトロン（次の章で紹介）など、より複雑な手法に関して基礎を説明します。

```toc
:maxdepth: 2

linear-regression
linear-regression-scratch
linear-regression-gluon
softmax-regression
image-classification-dataset
softmax-regression-scratch
softmax-regression-gluon
```

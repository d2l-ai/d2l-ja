# 経時的なバックプロパゲーション
:label:`sec_bptt`

これまでのところ、私たちは次のようなことを繰り返し言及してきました。
*グラデーションの爆発*、
*消失するグラデーション*、
そして、その必要性は
*RNN の勾配*を切り離します。
たとえば :numref:`sec_rnn_scratch` では、シーケンスに対して `detach` 関数を呼び出しました。モデルをすばやく構築し、それがどのように機能するかを確認するために、これについては完全には説明されていません。このセクションでは、シーケンスモデルのバックプロパゲーションの詳細と、数学がなぜ (そしてどのように機能するのか)、もう少し深く掘り下げます。 

RNN (:numref:`sec_rnn_scratch`) を最初に実装したとき、勾配爆発の影響のいくつかに遭遇しました。特に、この演習を解くと、適切な収束を確保するためにグラデーションクリッピングが不可欠であることがわかりました。この問題をより深く理解するために、この節ではシーケンスモデルの勾配がどのように計算されるかを復習します。この仕組みには概念的に新しいものは何もないことに注意してください。結局のところ、勾配を計算するためにチェーンルールを適用しているだけです。それでも、バックプロパゲーション (:numref:`sec_backprop`) をもう一度見直す価値はあります。 

:numref:`sec_backprop` では、MLPの順方向および逆方向の伝播と計算グラフについて説明しました。RNN での順方向伝搬は比較的簡単です。
*時間によるバックプロパゲーション*は実際には特有のものです
RNN :cite:`Werbos.1990` におけるバックプロパゲーションの応用。モデル変数とパラメーター間の依存関係を得るには、RNN の計算グラフをタイムステップごとに 1 つずつ展開する必要があります。次に、チェーンルールに基づいて、バックプロパゲーションを適用して勾配の計算と保存を行います。シーケンスはかなり長くなることがあるので、依存関係はかなり長くなる可能性があります。たとえば、1000 文字のシーケンスの場合、最初のトークンが最終位置のトークンに大きな影響を与える可能性があります。これは実際には計算上実現可能ではなく（時間がかかりすぎてメモリが多すぎる）、非常にわかりにくい勾配に到達するまでには1000を超える行列積が必要です。これは、計算上および統計上の不確実性に満ちたプロセスです。以下では、何が起きているのか、実際にどのように対処するのかを解明します。 

## RNNの勾配解析
:label:`subsec_bptt_analysis`

まず、RNN がどのように機能するかを単純化したモデルから始めます。このモデルでは、非表示ステートの詳細と更新方法の詳細は無視されます。ここでの数学的表記法では、スカラー、ベクトル、行列が以前ほど明確に区別されることはありません。これらの詳細は分析にとって重要ではなく、このサブセクションの表記法を乱雑にするだけの役目を果たします。 

この簡略化されたモデルでは、タイムステップ $t$ で $h_t$ を隠れ状態、$x_t$ を入力として、$o_t$ を出力として表しています。:numref:`subsec_rnn_w_hidden_states` での説明を思い出してください。入力と隠れ状態を連結して、隠れ層の 1 つの重み変数で乗算できます。したがって、$w_h$ と $w_o$ を使用して、それぞれ隠れ層と出力層の重みを示します。その結果、各タイムステップでの隠れ状態と出力は次のように説明できます。 

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

$f$ と $g$ はそれぞれ隠れ層と出力層の変換です。したがって、反復計算によって相互に依存する値の連鎖 $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ があります。順方向伝播はかなり単純です。必要なのは、$(x_t, h_t, o_t)$ トリプルを一度に 1 タイムステップずつループすることだけです。出力 $o_t$ と目的のラベル $y_t$ との間の不一致は、すべての $T$ タイムステップにわたって目的関数によって次のように評価されます。 

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

バックプロパゲーションでは、特に目的関数 $L$ のパラメーター $w_h$ に関して勾配を計算する場合は少し注意が必要です。具体的に言って、連鎖ルールでは、 

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh` の積の 1 番目と 2 番目の因子は簡単に計算できます。第三の因子 $\partial h_t/\partial w_h$ は、$h_t$ に対するパラメーター $w_h$ の効果を繰り返し計算する必要があるため、注意が必要です。:eqref:`eq_bptt_ht_ot` の反復計算によると、$h_t$ は $h_{t-1}$ と $w_h$ の両方に依存しています。$h_{t-1}$ の計算は $w_h$ にも依存します。したがって、連鎖規則を使用すると次のようになります。 

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

上記の勾配を導出するために、$t=1, 2,\ldots$ に対して $a_{0}=0$ と $a_{t}=b_{t}+c_{t}a_{t-1}$ を満たす 3 つのシーケンス $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ があると仮定します。$t\geq 1$の場合、簡単に見せることができます。 

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

$a_t$、$b_t$、および$c_t$を以下のように置き換えることで 

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

:eqref:`eq_bptt_partial_ht_wh_recur` の勾配計算は $a_{t}=b_{t}+c_{t}a_{t-1}$ を満たします。したがって、:eqref:`eq_bptt_at` に従って、:eqref:`eq_bptt_partial_ht_wh_recur` の反復計算を以下のように削除できます。 

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

チェーンルールを使用して $\partial h_t/\partial w_h$ を再帰的に計算できますが、$t$ が大きい場合は常にこのチェーンが非常に長くなる可能性があります。この問題に対処するためのいくつかの戦略について議論しましょう。 

### 完全計算 ### 

明らかに、:eqref:`eq_bptt_partial_ht_wh_gen`で全和を計算することができます。ただし、初期条件の微妙な変化が結果に多大な影響を与える可能性があるため、これは非常に遅く、勾配が爆発する可能性があります。つまり、バタフライ効果に似た現象が見られ、初期条件の変化が最小限に抑えられれば、結果に不均衡な変化が生じます。これは、推定したいモデルという点では、実際にはまったく望ましくありません。結局のところ、私たちはよく一般化するロバストな推定量を探しています。したがって、この戦略は実際にはほとんど使用されません。 

### タイムステップを切り捨てる###

あるいは、$\tau$ ステップの後に :eqref:`eq_bptt_partial_ht_wh_gen` の合計を切り捨てることもできます。これは、:numref:`sec_rnn_scratch` でグラデーションを切り離したときなど、これまで議論してきたことです。これにより、和を $\partial h_{t-\tau}/\partial w_h$ で終了するだけで、真の勾配の「近似」が得られます。実際には、これはかなりうまくいきます。これは一般に、時間 :cite:`Jaeger.2002` による切り捨てられた逆伝播と呼ばれるものです。この結果の 1 つは、モデルが長期的な結果ではなく短期的な影響に主に焦点を当てていることです。これは、推定がより単純で安定したモデルにバイアスをかけるため、実際には*望ましい*です。 

### ランダム化切り捨て ### 

最後に、$\partial h_t/\partial w_h$ を確率変数で置き換えることができます。この確率変数は、期待値では正しいが、シーケンスを切り捨てます。そのためには、$0 \leq \pi_t \leq 1$ があらかじめ定義された $\xi_t$ のシーケンスを使用します。$P(\xi_t = 0) = 1-\pi_t$ と $P(\xi_t = \pi_t^{-1}) = \pi_t$、つまり $E[\xi_t] = 1$ です。これを使って、:eqref:`eq_bptt_partial_ht_wh_recur` のグラデーション $\partial h_t/\partial w_h$ を 

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

$\xi_t$の定義から、$E[z_t] = \partial h_t/\partial w_h$ということになります。$\xi_t = 0$ の場合は常に、反復計算はそのタイムステップ $t$ で終了します。これにより、さまざまな長さのシーケンスの加重合計が生成されますが、長いシーケンスはまれですが、適切にオーバーウェイトされます。このアイデアは、タレックとオリヴィエ :cite:`Tallec.Ollivier.2017` によって提案されました。 

### 戦略を比較する

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt` は、RNN のバックプロパゲーションを使用して*The Time Machine* 本の最初の数文字を分析するときの 3 つの戦略を示しています。 

* 最初の行は、テキストをさまざまな長さのセグメントに分割するランダム化された切り捨てです。
* 2 行目は、テキストを同じ長さのサブシーケンスに分割する通常の切り捨てです。これは、RNN実験で私たちが行ってきたことです。
* 3 行目は、時間の経過に伴う完全な逆伝播であり、計算上実行不可能な式になります。

残念ながら、理論的には魅力的ですが、ランダム化切り捨ては通常の切り捨てよりはるかにうまく機能しません。これはおそらく多くの要因によるものです。第1に、過去に何度もバックプロパゲーションを行った後のオブザベーションの効果は、実際には依存関係を把握するのに十分である。第2に、分散が大きくなると、ステップ数が増えるほどグラデーションの精度が向上するという事実が打ち消されます。3つ目は、実際には相互作用の範囲が短いモデルを「求めている」ことです。したがって、経時的に定期的に切り捨てられたバックプロパゲーションには、多少の正則化効果があり、望ましい場合もあります。 

## 経時的なバックプロパゲーションの詳細

一般的な原則について議論した後、経時的な逆伝播について詳しく説明しましょう。:numref:`subsec_bptt_analysis` の解析とは異なり、以下では、分解されたすべてのモデルパラメーターに対する目的関数の勾配を計算する方法を示します。単純化するために、隠れ層の活性化関数が恒等マッピング ($\phi(x)=x$) を使用する、バイアスパラメーターのない RNN を考えます。タイムステップ $t$ では、1 つの入力例とラベルをそれぞれ $\mathbf{x}_t \in \mathbb{R}^d$ と $y_t$ とします。隠れ状態 $\mathbf{h}_t \in \mathbb{R}^h$ と出力 $\mathbf{o}_t \in \mathbb{R}^q$ は次のように計算されます。 

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

$\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$、$\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$、$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ はウェイトパラメータです。タイムステップ $t$ での損失を $l(\mathbf{o}_t, y_t)$ で表します。したがって、目的関数、シーケンスの開始から $T$ タイムステップを超える損失は次のようになります。 

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

RNN の計算中にモデル変数とパラメーターの間の依存関係を可視化するために、:numref:`fig_rnn_bptt` に示すように、モデルの計算グラフを描くことができます。たとえば、タイムステップ 3、$\mathbf{h}_3$ の隠れ状態の計算は、モデルパラメーター $\mathbf{W}_{hx}$ と $\mathbf{W}_{hh}$、最後のタイムステップ $\mathbf{h}_2$ の隠れ状態、および現在のタイムステップ $\mathbf{x}_3$ の入力によって決まります。 

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

前述したように、:numref:`fig_rnn_bptt` のモデルパラメータは $\mathbf{W}_{hx}$、$\mathbf{W}_{hh}$、および $\mathbf{W}_{qh}$ です。一般に、このモデルに学習させるには、$\partial L/\partial \mathbf{W}_{hx}$、$\partial L/\partial \mathbf{W}_{hh}$、$\partial L/\partial \mathbf{W}_{qh}$ の各パラメーターに対する勾配計算が必要です。:numref:`fig_rnn_bptt` の依存関係によると、矢印の反対方向にトラバースして、勾配を順番に計算して保存することができます。さまざまな形状の行列、ベクトル、スカラーの乗算を連鎖則で柔軟に表現するために、:numref:`sec_backprop` で説明されているように $\text{prod}$ 演算子を引き続き使用します。 

まず、任意のタイムステップ $t$ におけるモデル出力に対する目的関数の微分は、非常に簡単です。 

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

これで、出力層 $\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ のパラメーター $\mathbf{W}_{qh}$ に対する目的関数の勾配を計算できます。:numref:`fig_rnn_bptt` に基づくと、目的関数 $L$ は $\mathbf{o}_1, \ldots, \mathbf{o}_T$ 経由で $\mathbf{W}_{qh}$ に依存しています。連鎖規則を使用すると、 

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

$\partial L/\partial \mathbf{o}_t$ は:eqref:`eq_bptt_partial_L_ot`によって与えられます。 

次に、:numref:`fig_rnn_bptt` に示すように、最後のタイムステップ $T$ において、目的関数 $L$ は $\mathbf{o}_T$ を介してのみ隠れ状態 $\mathbf{h}_T$ に依存しています。したがって、連鎖規則を使用して勾配 $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$ を簡単に見つけることができます。 

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

目的関数 $L$ が $\mathbf{h}_{t+1}$ および $\mathbf{o}_t$ を経由して $\mathbf{h}_t$ に依存するタイムステップ $t < T$ では、より複雑になります。連鎖規則によれば、任意のタイムステップ $t < T$ における隠れ状態 $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ の勾配は次のように再帰的に計算できます。 

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

解析では、任意のタイムステップ $1 \leq t \leq T$ の反復計算を拡張すると、次のようになります。 

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

:eqref:`eq_bptt_partial_L_ht` からわかるように、この単純な線形の例はすでに長いシーケンスモデルのいくつかの重要な問題を示しています。$\mathbf{W}_{hh}^\top$ の非常に大きなべき乗が関係している可能性があります。その中で、1 より小さい固有値は消滅し、1 より大きい固有値は発散します。これは数値的に不安定で、消失して爆発する勾配の形で現れます。これに対処する 1 つの方法は、:numref:`subsec_bptt_analysis` で説明したように、計算上便利なサイズでタイムステップを切り捨てることです。実際には、この切り捨ては、所定のタイムステップ数の後にグラデーションをデタッチすることによって行われます。後ほど、長期短期記憶などのより洗練された配列モデルがこれをどのように緩和できるかを見ていきます。  

最後に :numref:`fig_rnn_bptt` は、目的関数 $L$ が、隠れ状態 $\mathbf{h}_1, \ldots, \mathbf{h}_T$ を介して隠れ層のモデルパラメーター $\mathbf{W}_{hx}$ および $\mathbf{W}_{hh}$ に依存することを示しています。このようなパラメーター $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ および $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ に関する勾配を計算するために、以下のような連鎖規則を適用します。 

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

:eqref:`eq_bptt_partial_L_hT_final_step` と :eqref:`eq_bptt_partial_L_ht_recur` によって繰り返し計算される $\partial L/\partial \mathbf{h}_t$ は、数値の安定性に影響する重要な量です。 

:numref:`sec_backprop` で説明したように、時間によるバックプロパゲーションは RNN のバックプロパゲーションの適用であるため、rNN のトレーニングでは、順伝播と時間経過のバックプロパゲーションが交互に行われます。また、時間によるバックプロパゲーションは、上記の勾配を順番に計算して保存します。具体的には、$\partial L / \partial \mathbf{W}_{hx}$ と $\partial L / \partial \mathbf{W}_{hh}$ の両方の計算に使用される $\partial L/\partial \mathbf{h}_t$ を格納するなど、格納された中間値は計算の重複を避けるために再利用されます。 

## [概要

* 経時的なバックプロパゲーションは、隠れ状態をもつシーケンスモデルへのバックプロパゲーションの適用にすぎません。
* 切り捨ては、通常の切り捨てやランダム化切り捨てなど、計算上の利便性と数値の安定性のために必要です。
* 行列のべき乗が高いと、固有値が発散したり消失したりすることがあります。これは、爆発または消失するグラデーションの形で現れます。
* 効率的な計算のために、時間経過の逆伝播中に中間値がキャッシュされます。

## 演習

1. 固有値 $\lambda_i$ をもつ対称行列 $\mathbf{M} \in \mathbb{R}^{n \times n}$ があり、対応する固有ベクトルが $\mathbf{v}_i$ ($i = 1, \ldots, n$) であると仮定します。一般性を失うことなく、$|\lambda_i| \geq |\lambda_{i+1}|$ の順序で順序付けられていると仮定します。 
   1. $\mathbf{M}^k$ に固有値 $\lambda_i^k$ があることを示します。
   1. 確率が高いランダムベクトル$\mathbf{x} \in \mathbb{R}^n$の場合、$\mathbf{M}^k \mathbf{x}$が固有ベクトル$\mathbf{v}_1$と非常に整列することを証明します。 
$\mathbf{M}$のうち。この声明を形式化してください。
   1. 上記の結果は、RNNの勾配に対してどのような意味がありますか？
1. 勾配クリッピング以外に、リカレントニューラルネットワークでの勾配爆発に対処する他の方法を考えられますか？

[Discussions](https://discuss.d2l.ai/t/334)

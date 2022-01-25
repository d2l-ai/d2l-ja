# ビームサーチ
:label:`sec_beam-search`

:numref:`sec_seq2seq` では、特別なシーケンスの終わり "<eos>" トークンが予測されるまで、出力シーケンストークンをトークンごとに予測しました。このセクションでは、この*greedy search* 戦略を形式化し、それに関する問題を調査することから始め、この戦略を他の代替戦略と比較します。
*網羅的検索*と*ビーム検索*。

貪欲探索を正式に紹介する前に、:numref:`sec_seq2seq` と同じ数学的表記法を使用して探索問題を形式化しましょう。任意のタイムステップ $t'$ において、デコーダ出力 $y_{t'}$ の確率は $t'$ より前の出力サブシーケンス $y_1, \ldots, y_{t'-1}$ と、入力シーケンスの情報をエンコードするコンテキスト変数 $\mathbf{c}$ を条件とします。計算コストを定量化するには、<eos>出力ボキャブラリを $\mathcal{Y}$ ("" を含む) で表します。したがって、このボキャブラリーセットのカーディナリティ$\left|\mathcal{Y}\right|$はボキャブラリーサイズです。また、出力シーケンスの最大トークン数を $T'$ と指定してみましょう。その結果、私たちの目標は $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ の可能なすべての出力シーケンスから理想的な出力を探すことです。もちろん、これらすべての出力シーケンスでは、"<eos>" 以降を含む部分は実際の出力では破棄されます。 

## 欲張り検索

まず、*欲張り検索*という単純な戦略を見てみましょう。この手法は :numref:`sec_seq2seq` のシーケンスを予測するために使用されています。欲張り探索では、出力シーケンスの任意のタイムステップ $t'$ で、$\mathcal{Y}$ から条件付き確率が最も高いトークンを検索します。つまり、  

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

を出力として出力します。「<eos>" が出力されるか、出力シーケンスが最大長 $T'$ に達すると、出力シーケンスは完了します。 

では、貪欲な検索では何がうまくいかないのでしょうか？実際、*optimalSequence* は最大$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$の出力シーケンスでなければなりません。これは、入力シーケンスに基づいて出力シーケンスを生成する条件付き確率です。残念ながら、欲張り探索によって最適な配列が得られる保証はありません。 

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

例を挙げて説明しましょう。<eos>出力ディクショナリに 4 つのトークン「A」、「B」、「C」、「」があるとします。:numref:`fig_s2s-prob1` では、各タイムステップの 4 つの数値は、<eos>そのタイムステップでそれぞれ「A」、「B」、「C」、「」を生成する条件付き確率を表します。各タイムステップで、貪欲検索では条件付き確率が最も高いトークンが選択されます。したがって、<eos>:numref:`fig_s2s-prob1` では、出力シーケンス「A」、「B」、「C」、および「」が予測されます。この出力シーケンスの条件付き確率は $0.5\times0.4\times0.4\times0.6 = 0.048$ です。 

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

次に、:numref:`fig_s2s-prob2` の別の例を見てみましょう。:numref:`fig_s2s-prob1` とは異なり、タイムステップ 2 では :numref:`fig_s2s-prob2` のトークン「C」を選択します。これは、条件付き確率が * 2 番目に高い「C」です。タイムステップ 3 の基になるタイムステップ 1 と 2 の出力サブシーケンスが :numref:`fig_s2s-prob1` の「A」と「B」から :numref:`fig_s2s-prob2` の「A」と「C」に変化したため、タイムステップ 3 の各トークンの条件付き確率も :numref:`fig_s2s-prob2` で変化しています。タイムステップ 3 でトークン「B」を選択したとします。これで、時間ステップ 4 は、:numref:`fig_s2s-prob1` の「A」、「B」、「C」、「C」とは異なる、最初の 3 つのタイムステップ「A」、「C」での出力部分シーケンスを条件とします。したがって、:numref:`fig_s2s-prob2` のタイムステップ 4 で各トークンを生成する条件付き確率も :numref:`fig_s2s-prob1` のそれとは異なります。その結果、<eos>:numref:`fig_s2s-prob2` の出力シーケンス「A」、「C」、「B」、および「」の条件付き確率は $0.5\times0.3 \times0.6\times0.6=0.054$ となり、:numref:`fig_s2s-prob1` の貪欲探索の条件付き確率よりも高くなります。この例では、<eos>欲張り探索で得られた出力シーケンス「A」、「B」、「C」、「」は最適なシーケンスではありません。 

## 網羅的検索

最適なシーケンスを求めることが目的であれば、*網羅的探索* の使用を検討できます。すべての出力シーケンスを条件付き確率で網羅的に列挙し、条件付き確率が最も高いものを出力します。 

網羅的探索を使用して最適な配列を得ることはできますが、その計算コスト $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ は過度に高くなる可能性があります。たとえば、$|\mathcal{Y}|=10000$ と $T'=10$ の場合、$10000^{10} = 10^{40}$ のシーケンスを評価する必要があります。これは不可能に近い！一方、欲張り探索の計算コストは $\mathcal{O}(\left|\mathcal{Y}\right|T')$ です。通常、網羅的探索よりも大幅に小さくなります。たとえば、$|\mathcal{Y}|=10000$ と $T'=10$ の場合、$10000\times10=10^5$ のシーケンスを評価するだけで済みます。 

## ビームサーチ

配列探索戦略に関する決定はスペクトル上にあり、どちらの極端でも簡単な質問があります。正確さだけが重要な場合はどうなりますか？明らかに、徹底的な検索。計算コストだけが問題になる場合はどうなりますか？明らかに、貪欲な検索。現実世界のアプリケーションでは、通常、この 2 つの極端の中間のどこかで、複雑な質問をします。 

*ビーム検索* は欲張り検索の改良版です。これには、*beam size*、$k$ という名前のハイパーパラメータがあります。 
タイムステップ 1 では、条件付き確率が最も高い $k$ トークンを選択します。それぞれが $k$ 個の候補出力シーケンスの最初のトークンになります。後続の各タイムステップで、前のタイムステップの $k$ の候補出力シーケンスに基づいて、$k\left|\mathcal{Y}\right|$ の選択肢の中から条件付き確率が最も高い $k$ 個の候補出力シーケンスを引き続き選択します。 

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search` は、ビーム探索のプロセスを例示して示します。出力ボキャブラリに $\mathcal{Y} = \{A, B, C, D, E\}$ の 5 つの要素しか含まれていないとします。そのうちの 1 つは「<eos>」です。ビームサイズを 2、出力シーケンスの最大長を 3 とします。タイムステップ 1 で、条件付き確率 $P(y_1 \mid \mathbf{c})$ が最も高いトークンが $A$ と $C$ であると仮定します。タイムステップ 2 で、すべての $y_2 \in \mathcal{Y},$ について計算します。  

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

これら 10 個の値の中から最も大きい 2 つ、たとえば $P(A, B \mid \mathbf{c})$ と $P(C, E \mid \mathbf{c})$ を選びます。次に、タイムステップ 3 で、すべての $y_3 \in \mathcal{Y}$ について計算します。  

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

これら 10 個の値の中から最も大きい 2 つ、たとえば $P(A, B, D \mid \mathbf{c})$ と $P(C, E, D \mid  \mathbf{c}).$ を選びます。その結果、(i) $A$、(ii) $C$、(iii) $A$、$B$、(iv) $C$、$E$、(v) $A$、$E$、(v) $A$、$A$、$E$ $D$、および (vi) $C$、$E$、$D$。  

最後に、これらの 6 つのシーケンス (例えば、「<eos>」を含む破棄部分) に基づいて、最終的な候補出力シーケンスのセットを取得します。次に、次のスコアのうち最も高いシーケンスを出力シーケンスとして選択します。 

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

$L$ は最終的な候補シーケンスの長さで、$\alpha$ は通常 0.75 に設定されます。:eqref:`eq_beam-search-score` の和では、数列が長いほど対数項が多くなるため、分母の $L^\alpha$ という項は長い系列にペナルティを課します。 

ビームサーチの計算コストは $\mathcal{O}(k\left|\mathcal{Y}\right|T')$ です。この結果は、貪欲検索の結果と網羅的探索の結果の中間にあります。実際、欲張りサーチは、ビームサイズが 1 の特殊なビームサーチとして扱うことができます。ビームサイズを柔軟に選択できるので、ビームサーチは精度と計算コストのトレードオフをもたらします。 

## [概要

* 配列探索戦略には、貪欲探索、網羅的探索、ビーム探索などがあります。
* ビームサーチは、ビームサイズを柔軟に選択できるため、精度と計算コストのトレードオフを実現します。

## 演習

1. 網羅的探索を特殊なビーム探索として扱うことはできますか？なぜ、なぜそうではないのですか？
1. :numref:`sec_seq2seq` の機械翻訳問題にビームサーチを適用します。ビームサイズは並進結果と予測速度にどのような影響を与えますか？
1. :numref:`sec_rnn_scratch` では、ユーザーが指定した接頭辞に従ってテキストを生成するために言語モデリングを使用しました。どのような検索戦略が使われていますか？改善してもらえますか？

[Discussions](https://discuss.d2l.ai/t/338)

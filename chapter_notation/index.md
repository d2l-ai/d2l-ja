# 記法
:label:`chap_notation`

本書では、以下の表記規則を順守しています。これらのシンボルの一部はプレースホルダであり、他のシンボルは特定のオブジェクトを参照します。一般的な経験則として、不定冠詞「a」は、シンボルがプレースホルダであり、同様の形式のシンボルが同じタイプの他のオブジェクトを表すことができることを示していることがよくあります。たとえば、「$x$: a scalar」は、小文字が一般にスカラー値を表すことを意味しますが、「$\mathbb{Z}$: 整数の集合」は特にシンボル $\mathbb{Z}$ を指します。 

## 数値オブジェクト

* $x$: スカラー
* $\mathbf{x}$: ベクトル
* $\mathbf{X}$: マトリックス
* $\mathsf{X}$: 一般的なテンソル
* $\mathbf{I}$:（ある特定の次元の）単位行列、すなわち、すべての対角要素に$1$、すべての非対角要素に$0$をもつ正方行列
* $x_i$、$[\mathbf{x}]_i$:$i^\mathrm{th}$ ベクトルの要素 $\mathbf{x}$
* $x_{ij}$、$x_{i,j}$、$[\mathbf{X}]_{ij}$、$[\mathbf{X}]_{i,j}$: 行$i$と列$j$の行列$\mathbf{X}$の要素。

## 集合理論

* $\mathcal{X}$: セット
* $\mathbb{Z}$: 整数の集合
* $\mathbb{Z}^+$: 正の整数の集合
* $\mathbb{R}$: 実数の集合
* $\mathbb{R}^n$: 実数の$n$次元ベクトルの集合
* $\mathbb{R}^{a\times b}$:$a$ 行と $b$ 列をもつ実数の行列の集合
* $|\mathcal{X}|$: セット $\mathcal{X}$ のカーディナリティ (要素の数)
* $\mathcal{A}\cup\mathcal{B}$: セット $\mathcal{A}$ と $\mathcal{B}$ のユニオン
* $\mathcal{A}\cap\mathcal{B}$: セット $\mathcal{A}$ と $\mathcal{B}$ の交差
* $\mathcal{A}\setminus\mathcal{B}$:$\mathcal{A}$ から $\mathcal{B}$ の減算を設定します ($\mathcal{A}$ の $\mathcal{B}$ に属さない要素のみが含まれます)

## 関数と演算子

* $f(\cdot)$: 関数
* $\log(\cdot)$: 自然対数 (底が$e$)
* $\log_2(\cdot)$: 底が$2$の対数
* $\exp(\cdot)$: 指数関数です
* $\mathbf{1}(\cdot)$: インジケーター関数は、ブール引数が真の場合は$1$、そうでない場合は$0$と評価されます
* $\mathbf{1}_{\mathcal{X}}(z)$: 集合メンバーシップ指標関数は、要素$z$が集合$\mathcal{X}$に属している場合は$1$と評価され、そうでなければ$0$と評価される
* $\mathbf{(\cdot)}^\top$: ベクトルまたは行列の転置
* $\mathbf{X}^{-1}$: 行列の逆行列 $\mathbf{X}$
* $\odot$: アダマール (元素的) 積
* $[\cdot, \cdot]$: 連結
* $\|\cdot\|_p$:$\ell_p$ ノルム
* $\|\cdot\|$:$\ell_2$ ノルム
* $\langle \mathbf{x}, \mathbf{y} \rangle$: ベクトル$\mathbf{x}$と$\mathbf{y}$のドット積
* $\sum$: 要素の集合の合計
* $\prod$: 要素の集合上のプロダクト
* $\stackrel{\mathrm{def}}{=}$: 左側のシンボルの定義として表される等価性

## 微積分

* $\frac{dy}{dx}$:$x$ に対する $y$ の派生物
* $\frac{\partial y}{\partial x}$:$x$ に対する $y$ の偏微分
* $\nabla_{\mathbf{x}} y$:$\mathbf{x}$ に対するグラデーション $y$
* $\int_a^b f(x) \;dx$:$x$に対する$a$から$b$への$f$の定積分
* $\int f(x) \;dx$:$x$ に対する $f$ の不定積分

## 確率と情報理論

* $X$: 確率変数
* $P$: 確率分布
* $X \sim P$: 確率変数 $X$ は分布 $P$ に続きます
* $P(X=x)$: 確率変数$X$が値$x$を取る事象に割り当てられる確率
* $P(X \mid Y)$:$Y$が与えられた場合の$X$の条件付き確率分布
* $p(\cdot)$: 分布Pに関連する確率密度関数 (PDF)
* ${E}[X]$: 確率変数の期待 $X$
* $X \perp Y$: 確率変数 $X$ と $Y$ は独立しています
* $X \perp Y \mid Z$: 確率変数$X$および$Y$は、$Z$が与えられた場合に条件付きで独立しています
* $\sigma_X$: 確率変数の標準偏差 $X$
* $\mathrm{Var}(X)$: 確率変数$X$の分散、$\sigma^2_X$と等しい
* $\mathrm{Cov}(X, Y)$: 確率変数の共分散 $X$ と $Y$
* $\rho(X, Y)$:$X$と$Y$の間のピアソン相関係数は、$\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$と等しくなります
* $H(X)$: ランダム変数のエントロピー $X$
* $D_{\mathrm{KL}}(P\|Q)$: 分布$Q$から分布$P$へのKLダイバージェンス（または相対エントロピー）

[Discussions](https://discuss.d2l.ai/t/25)

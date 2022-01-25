# 表記法
:label:`chap_notation`

本書では、以下の表記規則を順守しています。これらの記号にはプレースホルダであるものもあれば、特定のオブジェクトを参照するものもあります。一般的な経験則として、不定冠詞「a」は、シンボルがプレースホルダであり、同じ形式のシンボルが同じタイプの他のオブジェクトを表すことができることを示します。たとえば、「$x$: スカラー」は、小文字が一般的にスカラー値を表すことを意味します。 

## 数値オブジェクト

* $x$: スカラー
* $\mathbf{x}$: ベクトルです
* $\mathbf{X}$: マトリックスです
* $\mathsf{X}$: 一般的なテンソル
* $\mathbf{I}$: 単位行列-正方形。すべての対角線エントリに $1$、すべての対角線外に $0$ をもつ
* $x_i$、$[\mathbf{x}]_i$:$i^\mathrm{th}$ ベクトルの $i^\mathrm{th}$ エレメントです。
* $x_{ij}$、$x_{i,j}$、$[\mathbf{X}]_{ij}$、$[\mathbf{X}]_{i,j}$: 行$i$ と列 $j$ にあるマトリックス $\mathbf{X}$ のエレメントです。

## 集合論

* $\mathcal{X}$: セット
* $\mathbb{Z}$: 整数の集合
* $\mathbb{Z}^+$: 正の整数の集合
* $\mathbb{R}$: 実数の集合
* $\mathbb{R}^n$:$n$ 次元の実数ベクトルの集合
* $\mathbb{R}^{a\times b}$:$a$ 行と $b$ 列をもつ実数の行列の集合
* $|\mathcal{X}|$: 集合 $\mathcal{X}$ の基数 (エレメントの数)
* $\mathcal{A}\cup\mathcal{B}$:$\mathcal{A}$ と $\mathcal{B}$ のセットのユニオン
* $\mathcal{A}\cap\mathcal{B}$: セット$\mathcal{A}$と$\mathcal{B}$の交差部分
* $\mathcal{A}\setminus\mathcal{B}$:$\mathcal{A}$ から $\mathcal{B}$ の減算を設定する ($\mathcal{A}$ のうち $\mathcal{B}$ に属さない要素のみを含む)

## 関数と演算子

* $f(\cdot)$: 関数です
* $\log(\cdot)$: 自然対数 (基数 $e$)
* $\log_2(\cdot)$: 底を底とする対数 $2$
* $\exp(\cdot)$: 指数関数です
* $\mathbf{1}(\cdot)$: インジケーター関数。ブール型引数が真であれば $1$、そうでなければ $0$ に評価されます。
* $\mathbf{1}_{\mathcal{X}}(z)$: セットメンバシップインジケータ関数。エレメント $z$ がセット $\mathcal{X}$ に属していれば $1$ に評価され、そうでなければ $0$ に評価されます。
* $\mathbf{(\cdot)}^\top$: ベクトルまたは行列の転置
* $\mathbf{X}^{-1}$: 行列の逆行列$\mathbf{X}$
* $\odot$: アダマール (要素単位) 積
* $[\cdot, \cdot]$: コンカチネーション
* $\|\cdot\|_p$:$L_p$ ノルム
* $\|\cdot\|$:$L_2$ ノルム
* $\langle \mathbf{x}, \mathbf{y} \rangle$: ベクトル$\mathbf{x}$と$\mathbf{y}$のドット積
* $\sum$: 要素の集合に対する総和
* $\prod$: 要素のコレクションに対するプロダクト
* $\stackrel{\mathrm{def}}{=}$: 左辺のシンボルの定義として表明された等価性

## 微積分

* $\frac{dy}{dx}$:$x$ を基準にした $y$ の微分
* $\frac{\partial y}{\partial x}$:$x$ を基準にした $y$ の偏微分
* $\nabla_{\mathbf{x}} y$:$\mathbf{x}$ を基準にした $y$ のグラデーション
* $\int_a^b f(x) \;dx$:$x$ を基準にして $a$ から $b$ までの $f$ の定積分
* $\int f(x) \;dx$:$x$ を基準にした $f$ の不定積分

## 確率論と情報理論

* $X$: 確率変数です
* $P$: 確率分布
* $X \sim P$: 確率変数 $X$ の分布は $P$ です
* $P(X=x)$: 確率変数 $X$ が値 $x$ を取る事象に割り当てられる確率
* $P(X \mid Y)$:$Y$ が与えられた場合の$X$の条件付き確率分布
* $p(\cdot)$: 分布 P に関連付けられた確率密度関数 (PDF)
* ${E}[X]$: 確率変数の期待値 $X$
* $X \perp Y$: 確率変数 $X$ と $Y$ は独立しています
* $X \perp Y \mid Z$:$Z$ が与えられた場合、確率変数 $X$ と $Y$ は条件付きで独立しています
* $\sigma_X$: 確率変数の標準偏差 $X$
* $\mathrm{Var}(X)$: 確率変数 $X$ の分散、$\sigma^2_X$ と等しい
* $\mathrm{Cov}(X, Y)$: 確率変数の共分散 $X$ と $Y$
* $\rho(X, Y)$:$X$ と $Y$ の間のピアソン相関係数は $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$ と等しくなります
* $H(X)$: 確率変数のエントロピー $X$
* $D_{\mathrm{KL}}(P\|Q)$: 分布 $Q$ から分布 $P$ への KL ダイバージェンス (または相対エントロピー)

[Discussions](https://discuss.d2l.ai/t/25)

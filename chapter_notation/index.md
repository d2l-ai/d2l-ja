# 表記法
:label:`chap_notation`

書籍内の表記法を以下にまとめます。


## 変数

* $x$: スカラー
* $\mathbf{x}$: ベクトル
* $\mathbf{X}$: 行列
* $\mathsf{X}$: テンソル
* $\mathbf{I}$: 単位行列
* $x_i$, $[\mathbf{x}]_i$: ベクトル $\mathbf{x}$ の $i$ 番目の要素
* $x_{ij}$, $[\mathbf{X}]_{ij}$: 行列 $\mathbf{X}$ の行 $i$、列$j$ の要素

## 集合

* $\mathcal{X}$: 集合
* $\mathbb{Z}$: 整数の集合
* $\mathbb{R}$: 実数の集合
* $\mathbb{R}^n$: 実数の $n$ 次元ベクトル
* $\mathbb{R}^{a\times b}$: $a$行$b$列の実数の行列
* $\mathcal{A}\cup\mathcal{B}$: $\mathcal{A}$ と $\mathcal{B}$の和集合
* $\mathcal{A}\cap\mathcal{B}$: $\mathcal{A}$ と $\mathcal{B}$の積集合
* $\mathcal{A}\setminus\mathcal{B}$: $\mathcal{A}$ から $\mathcal{B}$を引いた差集合

## 関数と演算

* $f(\cdot)$: 関数
* $\log(\cdot)$: 自然対数
* $\exp(\cdot)$: 指数関数
* $\mathbf{1}_\mathcal{X}$: 指示関数
* $\mathbf{(\cdot)}^\top$: ベクトルや行列の転置
* $\mathbf{X}^{-1}$: 行列 $\mathbf{X}$ の逆行列
* $\odot$: アダマール (要素ごとの) 積                
* $\lvert \mathcal{X} \rvert$: 集合 $\mathcal{X}$ の基数 (カーディナリティ)
* $\|\cdot\|_p$: $\ell_p$ ノルム            
* $\|\cdot\|$: $\ell_2$ ノルム       
* $\langle \mathbf{x}, \mathbf{y} \rangle$: ベクトル $\mathbf{x}$ と $\mathbf{y}$ のドット積
* $\sum$: 総和                      
* $\prod$: 総乗                  


## 微積分

* $\frac{dy}{dx}$: $y$ の $x$ についての微分        
* $\frac{\partial y}{\partial x}$: $y$ の $x$ についての偏微分
* $\nabla_{\mathbf{x}} y$: $x$ についての $y$ の勾配
* $\int_a^b f(x) \;dx$: $x$に関して $a$ から $b$ までの定積分
* $\int f(x) \;dx$: $f$ の $x$ についての不定積分

## 確率と情報理論

* $P(\cdot)$: 確率分布
* $z \sim P$: 確率変数 $z$ が確率分布 $P$ に従う
* $P(X \mid Y)$: $X \mid Y$ の条件付き確率
* $p(x)$: 確率密度関数
* ${E}_{x} [f(x)]$: $x$に関する$f$の期待値  
* $X \perp Y$: 確率変数 $X$ と $Y$ は独立である
* $X \perp Y \mid Z$: 確率変数 $X$ と $Y$ は、確率変数$Z$ の条件のもとでの条件付き独立
* $\mathrm{Var}(X)$: 確率変数 $X$ の分散
* $\sigma_X$: 確率変数 $X$ の標準偏差
* $\mathrm{Cov}(X, Y)$: 確率変数 $X$ と $Y$ の共分散
* $\rho(X, Y)$:  $X$ と $Y$ の相関係数
* $H(X)$: 確率変数 $X$ のエントロピー
* $D_{\mathrm{KL}}(P\|Q)$: 分布 $P$ と $Q$ の KL-divergence



## 複雑さ

* $\mathcal{O}$: Big O notation (ビッグ・オー記法)


## [議論](https://discuss.mxnet.io/t/4367)

![](../img/qr_notation.svg)

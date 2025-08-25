---
title: 🧮 Axiom and chain rule of vector and matrix differance
author: Hongseo Jang
date: 2025-08-22
category: NLP
layout: post
---

> 이 정리본은 [작성자]가 인공지능을 공부하는 과정에서 이해하기 어려웠던 수학 내용을 중심으로 작성되었습니다. 작성된 내용을 최종적으로 검증하는 데 GPT를 사용했지만, 그 외에는 GPT를 사용하지 않았습니다. 이모지 없이 마크다운으로 작성하려니 너무 재미없어 보여서 추가했으니 양해 부탁드립니다.


## 🧠 사전 지식

### ↗️ What is $sclar$ and $vector$?
먼저 $sclar$와 $vector$의 차이부터 살펴보겠습니다.

$scalar$: 우리 모두가 알고 있는 그 값 맞습니다. 크기만을 가지고 있죠.

$$1, 2, 3, \cdots$$

$vector$: 이 친구는 크기와 방향을 모두 가지고 있습니다. ($ex: [1, 2]$)

$$[1, 2],\;[1, 2, 3],\;[1,2,3,4]\cdots$$

결과적으로 $[1, 2]$는 2차원이고, $[1, 2, 3]$은 3차원, $[1, 2, 3, \cdots , N]$은 N차원을 의미하겠죠?

여기서 중요한 부분이 $vector$나 $matrix$의 전치를 하는 이유는 단지 연산의 모양을 맞추기 위한 과정이지, **"의미가 달라지는것이 아니다".** 라는 겁니다. (자연어처리 부분에서 매우 중요합니다.)

<br>

#### ↗️ What is $\text{Dot product}$ and $\text{Frobenius inner product}$

$\text{Dot product}$:

$$\langle a,b\rangle=a\cdot b=a^\top b=\sum_{i}a_ib_i$$

$\text{Frobenius inner product}$:

$$\langle A,B\rangle=\mathrm{tr}(A^\top B)=\sum_{i j}A_{ij}B_{ij}$$

$\text{Dot product}$는 $vector$에서 사용되고, $\text{Frobenius inner product}$은 $matrix$에서 사용되는데, $vector$와 $matrix$가 뭘 의미하는지 아시나요?

<br>

### ↗️ Can you explain a difference of $vector$ and $matrix$?
$vector$는 앞에서 크기와 방향을 가지고 있는 값이라고 했죠? $\text{row vector}$와 $\text{colmn vector}$가 이에 해당하는데 $\text{row vector}$의 $\text{Transpose}$가 $\text{colmn vector}$죠. 결국 $\text{row vector}$와 $\text{column vector}$는 좌표공간에서 동일한 의미를 갖지만, 연산을 하기 위해서 $\text{Transpose}$를 한다는것을 기억하셔야합니다!

$matrix$는 **$matrix$를 $vector$로 바꾸는 기계**라고 생각하시면 편합니다. 이게 무슨말이냐면, 제가 아래에 예제를 하나 보여드릴게요.

$$\begin{bmatrix}
2 & 0 \\
0 & 3 
\end{bmatrix}
\cdot
\begin{bmatrix}
1 \\
2 
\end{bmatrix}=
\begin{bmatrix}
2 \\
6 
\end{bmatrix}
$$

  앞에 있는 친구가 $matrix$이고, 뒤에 있는 친구가 $vector$죠? $vector$는 **현재 2차원 좌표평면에서 가로 1, 세로 2**를 가리키고 있습니다. 이 친구의 크기는 $\sqrt{1^2+2^2}=\sqrt{5}$겠네요. **원래 $[1,2]$가 $[2,6]$으로 가로로 2배, 세로로 3배 늘어났네요!** 이제 $matrix$를 $vector$로 바꾸는 기계라는 말을 이해할 수 있겠죠?

<br>


## 📖 1. $\text{Frechet Derivative}$과 $\text{Linear Approximation}$

### (1) Definition of $\text{Frechet Derivative}$
$f: \mathbb{R}^n\to \mathbb{R}^m$이 점 $x$에서 미분가능하다는 것은 $L\in \mathcal{L}(\mathbb{R}^n,\mathbb{R}^m)$가 존재하여

$$
\newline
\lim_{\|h\|\to 0}\frac{\|f(x+h)-f(x)-L(h)\|}{\|h\|}=0
\newline
$$

가 성립하는게 됩니다. 이때 $Df(x)=L$, $Df(x)[h]$는 $h$ 방향의 선형근사입니다. (여기서 $\text{norm}$을 사용하는 이유는, $n$차 이상이기 때문입니다.)

### (2) $Jacobian$·$Gradient$·$Hessian$
- **야코비안**: 함수의 출력 성분을 입력 변수에 대해 편미분해서 모아놓은 행렬로 
 $J_f(x)\in\mathbb{R}^{m\times n}$: $Df(x)[h]=J_f(x)h$.
- **그라디언트**: 기울기 벡터로 함수가 가장 가파르게 증가하는 경향을 나타내고, 
(스칼라값 $f:\mathbb{R}^n\to\mathbb{R}$): 리즈 표현에 의해

$$
\newline
Df(x)[h]=\langle \nabla f(x),h\rangle=\nabla f(x)^\top h,\qquad \nabla f(x)\in\mathbb{R}^n.
\newline
$$

- **헤시안**: 모든 변수에 대해 두 번 편미분한 값을 모은 대칭 행렬로 (스칼라값 $f$): $H_f(x)\in\mathbb{R}^{n\times n}$로

$$
\newline
D^2f(x)[h,k]=h^\top H_f(x)k,\qquad \mathrm{d}^2 f= (\mathrm{d}x)^\top H_f(x)\,\mathrm{d}x.
\newline
$$

### (3) 행렬변수의 미분
$f:\mathbb{R}^{n\times p}\to \mathbb{R}$일 때, 그라디언트 $\nabla_X f(X)\in\mathbb{R}^{n\times p}$는

$$
\newline
Df(X)[H]=\langle \nabla_X f(X),\,H\rangle=\mathrm{tr}\!\big(\nabla_X f(X)^\top H\big)
\newline
$$

을 만족하는 유일한 행렬입니다. 행렬값 함수 $F:\mathbb{R}^{n\times p}\to\mathbb{R}^{m\times q}$의 미분은 일반적으로 4차 텐서지만, 실무에서는 $\mathrm{vec}$와 크로네커를 써서 **야코비안 행렬**로 표현합니다:

$$
\newline
\mathrm{vec}(F(X))=J_{F}(X)\,\mathrm{vec}(X),\qquad J_F(X)\in\mathbb{R}^{mq\times np}.
\newline
$$


<br>

## 2. Chain Rule

### (1) 프레셰 관점의 chain rule
$g:\mathbb{R}^n\to\mathbb{R}^m,\ f:\mathbb{R}^m\to\mathbb{R}^p$이면

$$
\newline
D(f\circ g)(x)=Df(g(x))\circ Dg(x).
\newline
$$

좌표 표현으로는 $J_{f\circ g}(x)=J_f(g(x))\,J_g(x)$.

### (2) 스칼라합성의 그라디언트
$f:\mathbb{R}^m\to\mathbb{R}$, $g:\mathbb{R}^n\to\mathbb{R}^m$일 때

$$
\newline
\nabla (f\circ g)(x)=J_g(x)^\top\,\nabla f(g(x)).
\newline
$$

차원: $J_g\in\mathbb{R}^{m\times n}$, $\nabla f\in\mathbb{R}^m$, 결과는 $\mathbb{R}^n$.

### (3) 2계 chain rule
같은 가정에서

$$
\newline
\nabla^2 (f\circ g)(x)
=J_g(x)^\top\,\nabla^2 f(g(x))\,J_g(x)\;+\;\sum_{i=1}^{m}\frac{\partial f}{\partial u_i}(g(x))\,\nabla^2 g_i(x).
\newline
$$

첫 항은 **바깥 함수의 곡률 전파**, 둘째 항은 **안쪽 함수의 곡률 기여**를 의미합니다.

### (4) Vectorization Identity
선형변환 $F(X)=AXB$에 대해

$$
\newline
\mathrm{d}F= A\,\mathrm{d}X\,B,\qquad 
\mathrm{vec}(AXB)=(B^\top\!\otimes A)\,\mathrm{vec}(X).
\newline
$$

여기서 $\mathrm{d}F= A\,\mathrm{d}X\,B$는 $X$가 $dX$만큼 변할 때, 출력값은 $AdXB$만큼 변한다는 것입니다. 
따라서 $J_{\mathrm{vec}F,\mathrm{vec}X}=B^\top\!\otimes A$ 에서 행렬을 곱하는 선형 변환은, $\mathrm{vec}$과 크로네커 곱으로 쓰면 커다란 행렬 곱셈으로 정리되는거죠.

<br>

## 3. 트레이스 기법과 핵심 identity

트레이스의 순환성과 미분규칙은 행렬미분을 “스칼라화”하여 단순화합니다.

$$
\newline
\mathrm{tr}(AB)=\mathrm{tr}(BA),\quad
\mathrm{d}\,\mathrm{tr}(AB)=\mathrm{tr}(A\,\mathrm{d}B)+\mathrm{tr}(\mathrm{d}A\,B).
\newline
$$

자주 쓰는 등식:
- $x^\top y=\mathrm{tr}(y x^\top)\ \Rightarrow\ \mathrm{d}(x^\top y)=y^\top \mathrm{d}x+x^\top \mathrm{d}y$.
- $\mathrm{d}\,\mathrm{tr}(A^\top X)=\mathrm{tr}(A^\top \mathrm{d}X)\ \Rightarrow\ \nabla_X \mathrm{tr}(A^\top X)=A$.
- $X$가 변수일 때 $\mathrm{d}\,\mathrm{tr}(X^\top A X)=\mathrm{tr}\big((A^\top\!+\!A)X\big)^\top \mathrm{d}X$.

<br>

## 4. 이차형식과 규제항

### (1) 이차형식 $x^\top A x$
트레이스로 스칼라화:

$$
\newline
x^\top A x=\mathrm{tr}(x^\top A x)=\mathrm{tr}(A x x^\top).
\newline
$$

미분하면

$$
\newline
\mathrm{d}(x^\top A x)= (\mathrm{d}x)^\top A x + x^\top A\,\mathrm{d}x
= x^\top(A^\top\!+\!A)\,\mathrm{d}x.
\newline
$$
이다. 
따라서

$$
\newline
\nabla_x (x^\top A x)=(A+A^\top)x,\qquad 
\nabla_x^2 (x^\top A x)=A+A^\top.
\newline
$$

대칭 $A=A^\top$이면 $\nabla_x (x^\top A x)=2Ax$, $\nabla_x^2=2A$. 특히 $f(x)=\tfrac{\lambda}{2}\|x\|_2^2$일 때 $\nabla f=\lambda x,\ H=\lambda I$이라고 할 수 있다.

### (2) 로지스틱 손실의 그라디언트·헤시안
데이터 $X\in\mathbb{R}^{n\times d}$, 파라미터 $w\in\mathbb{R}^d$, 표기 $z=Xw$, 시그모이드 $\sigma(z)=1/(1+e^{-z})$를 성분별로 적용. 이진 레이블 $y\in\{0,1\}^n$에 대한 음의 로그우도(크로스엔트로피를 의미합니다.)

$$
\newline
\ell(w)=-\sum_{i=1}^n\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big],\quad p=\sigma(z).
\newline
$$

체인룰과 선형성으로

$$
\newline
\nabla \ell(w)=X^\top(p-y),\qquad 
\nabla^2 \ell(w)=X^\top S X,\quad S=\mathrm{diag}\big(p_i(1-p_i)\big).
\newline
$$

$y\in\{\pm1\}$ 표기에서는 $p_i=\sigma(y_i x_i^\top w)$로 바꾸면 동일한 구조가 나옵니다.

<br>


## 6. 트레이스 순환성 → 프로젝션 미분 증명
$U\in\mathbb{R}^{n\times r}$, 투영행렬 $P=UU^\top$ (열 직교 가정). 스칼라 함수 $g(U)=\mathrm{tr}(AP)$에 대해

$$
\newline
\mathrm{d}g=\mathrm{d}\,\mathrm{tr}(A UU^\top)=\mathrm{tr}\big(A(\mathrm{d}U\,U^\top+U\,\mathrm{d}U^\top)\big)
=\mathrm{tr}(U^\top A\,\mathrm{d}U)+\mathrm{tr}\big((A U)^\top\mathrm{d}U\big).
\newline
$$

트레이스 순환성과 $\mathrm{tr}(B^\top C)=\langle B,C\rangle$로 묶으면

$$
\newline
\mathrm{d}g=\mathrm{tr}\!\big(\big(U^\top(A+A^\top)\big)\mathrm{d}U\big)
=\langle (A+A^\top)U,\ \mathrm{d}U\rangle.
\newline
$$

따라서(제약을 무시한 **유클리드** 그라디언트) $\nabla_U g=(A+A^\top)U$. $A$가 대칭이면 $\nabla_U g=2AU$. 이는 투영 관련 최적화(PCA에서 중요합니다. 이거..)에서 자주 쓰이는 기본 미분입니다.


[작성자]: https://github.com/pxxguin
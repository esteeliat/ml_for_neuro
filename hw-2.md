---
title: Homework 2
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---
$$ {2\over 3} $$
## Question 1: 
## c.
we want to compute $$L(θ)=(P(X_1=x_1))$$*...*$$(P(X_n=x_n))$$.

first, we notice that n1=|{i | X_i=1}|, n2=|{i | X_i=2}|, n3=|{i | X_i=3\}|

so we do get n=n1+n2+n3, therefore - $$P(X_i=1)= θ$$, $$P(X_i=2)={2(1−θ)​ \over 3}$$, $$P(X_i=3)={(1−θ)​\over 3} $$.

we then get that $$L(θ)=$$ $${θ^{n_1}}$$ * ($${2(1-θ) \over 3}){^{n_2}}$$ * ($${(1-θ) \over 3}){^{n_3}}$$ = $${2^{{n_2}} \over 3^{n_2+n_3}}$$ * $${θ^{n_1}}$$ * $${(1-θ)^{n_2+n_3}}$$
## d.
log likelihood function - $$l(θ)=log(L(θ))$$

so $$l(θ) =$$ log($${2^{{n_2}} \over 3^{n_2+n_3}}$$ * $${θ^{n_1}}$$ * $${(1-θ)^{n_2+n_3}}$$) = $${n_2}log(2)-({n_2+n_3})log(3)+{n_1}log(θ)+({n_2+n_3})log(1-θ)$$

now in order to find the value of $$θ$$ that maximizes it we will Differentiate $$l$$ by $$θ$$: 
$$dl \over dθ$$ = $${n_1} \over θ$$ - $${n_2+n_3} \over 1-θ$$

because the first two termsare not depended on $$θ$$ and the derivative of log(θ) is $$1 \over θ$$. also, there is a minus over $${n_2+n_3} \over 1-θ$$ as we derivative $$log(1-θ)$$ by $$θ$$.

now in order to find the maximum we will compare to zero the derivative:
$${n_1} \over θ$$ = $${n_2+n_3} \over 1-θ$$ so $${n_1}(1-θ)=({n_2+n_3})θ$$

=> $$n_1$$ = $$θ({n_1+n_2+n_3})$$ = $$θn$$ => $$\hatθ_{MLE}$$ = $$n_1 \over n$$
## e.
Compute the bias: $$bias(\hatθ)$$ = $$E[\hatθ]-θ$$ = 

from last section we got $$\hatθ$$ = $$n_1 \over n$$ that's equal to $$1 \over n$$ $$\sum_{i=1}^{n}E[{1_{(X_i=1)}}]$$ 

because $$E[\hatθ]={n_1 \over n}$$ = $${1 \over n}E[n_1]$$  so we can write the indicator $${1_{(X_i=1)}}$$ is 1 when $${X_i=1}$$ or $$0$$ else. 

then from linearity $$E[{n_1}]=$$ $$\sum_{i=1}^{n}E[{1_{(X_i=1)}}]$$, because $${n_1}$$ is actually summing all the times $${X_i}=1$$ 

However, since $$P({X_i}=1)=θ$$ we therefore get $$\sum_{i=1}^{n}E[{1_{(X_i=1)}}]$$ = $$nθ$$

so $$E[\hatθ]={nθ \over n}=θ$$ => $$bias(\hatθ)$$ = $$E[\hatθ]-θ$$ = $$θ-θ$$ = 0

and compute the variance: $$Var(\hatθ)$$

$$Var(\hatθ)=Var({1 \over n}$$ $$\sum_{i=1}^{n}[{1_{(X_i=1)}}]$$) = $$Var({1 \over {n^2}}$$ $$\sum_{i=1}^{n}[{1_{(X_i=1)}}]$$)

we know that the variables $${X_1}$$,..., $${X_n}$$ are independent, so Var($$\sum_{i=1}^{n}[{1_{(X_i=1)}}]$$) = $$\sum_{i=1}^{n}Var[{1_{(X_i=1)}}]$$

they are also bernoulli so $$Var[{1_{(X_i=1)}}]$$= $$θ(1-θ)$$,

=> Var($$\sum_{i=1}^{n}[{1_{(X_i=1)}}]$$) = $$nθ(1−θ)$$ 

therefore $$Var(\hatθ)$$ = $${1 \over {n^2}}*nθ(1-θ)={θ(1-θ) \over n}$$
## final answers: $$E[\hatθ]=0$$ and $$Var(\hatθ)={θ(1-θ) \over n}$$ 
## f.
Compute the Mean Squared Error (MSE):

$$MSE(\hatθ)=Var(\hatθ)+{Bias(\hatθ)}^2={θ(1-θ) \over n}-0^2={θ(1-θ) \over n}$$
## Question 2:

## Question 3:

---
title: Homework 2
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---

## Question 1:  
a.  
we donote $\pi=P(D=yes)$ also we will denote $sn=sensitivity=P(T=+∣D=yes)=0.86$ and $sp=specificity=P(T=−∣D=no)=0.88$  
let's remind that $PPV=P(D=yes∣T=+)$ so in order to Use Baye's rule to derive PPV as a function of $\pi$,sn,sp we will do:
since $P(T=+∣D=no)=1−0.88=0.12$  
and $P(T=+)=P(T=+∣D=yes)P(D=yes)+P(T=+∣D=no)P(D=no)=sn*\pi+(1-sp)(1-\pi)$
from Baye's rule: $PPV(\pi,sn,sp)=P(D=yes∣T=+)=\frac{P(T=+∣D=yes)P(D=yes)}{P(T=+)}$  
so  $PPV(\pi,sn,sp)=\frac{sn*\pi}{sn*\pi+(1-sp)(1-\pi)}$    
this means that even with high sensitivity and specificity, the PPV strongly depends on 𝜋. so when the disease is rare, most positive test results may still be false positives.  

b.  Evaluate the PPV when $\pi=0.01:
$sn=0.86$, $sp=0.88$, $1-sp=1-0.88=0.12$
$PPV(\pi,sn,sp)=\frac{sn*\pi}{sn*\pi+(1-sp)(1-\pi)}=\frac{0.0086}{0.0086+0.12*0.99}=\frac{0.0086}{0.1274}≈0.0675$  
even though the test has high sensitivity (86%) and high specificity (88%), when the disease PPV is only 1%, most positive test results are false positives.

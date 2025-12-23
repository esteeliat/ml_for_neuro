---
title: Homework 2
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---

## Question 1:  
a.  we donote $\pi=P(D=yes)$ also we will denote $sn=sensitivity=P(T=+∣D=yes)=0.86$ and $sp=specificity=P(T=−∣D=no)=0.88$  
let's remind that $PPV=P(D=yes∣T=+)$ so in order to Use Baye's rule to derive PPV as a function of $\pi$,sn,sp we will do:
since $P(T=+∣D=no)=1−0.88=0.12$  
and $P(T=+)=P(T=+∣D=yes)P(D=yes)+P(T=+∣D=no)P(D=no)=0.86\pi+0.12(1-\pi)$
from Baye's rule: $PPV(\pi,sn,sp)=P(D=yes∣T=+)=P(T=+∣D=yes)P(D=yes)​/P(T=+)$  
so $PPV(\pi,sn,sp)=0.860.86\pi+0.12(1-\pi)$

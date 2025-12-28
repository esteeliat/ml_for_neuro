---
title: Homework 4
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
## Question 2:  
## Question 3:  
a.  
```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
N = 100
sigma = np.sqrt(0.05)

rng = np.random.default_rng(42)
x = rng.uniform(0.0, 1.0, size=N)

# Ground-truth parameters
R_max = 1.0
C50 = 0.25
n_exp = 2.0

def naka_rushton(x, R_max=R_max, C50=C50, n=n_exp):
    return R_max * (x**n) / (x**n + C50**n + 1e-12)

y_clean = naka_rushton(x)
noise = rng.normal(0.0, sigma, size=N)
y = y_clean + noise

#shuffling before splitting because training and test sets will be #representative and results would not  depend on sampling order
perm = rng.permutation(N)

train_size = int(0.8 * N)
train_idx = perm[:train_size]
test_idx = perm[train_size:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")
```
b.  
```{code-cell}
#Building the polynomial feature matrix
def design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)

#Creates a random permutation of training indices, Ensures folds are random, #not ordered and Split indices into 5 folds
K = 5
N_train = len(x_train)
rng = np.random.default_rng(123)
perm = rng.permutation(N_train)
folds = np.array_split(perm, K)
cv_mse = {}

#Loop for the 5-fold Cross-validation
#outer loop on the polynomials
for deg in range(4):
    fold_mse = []
#inner loop on the fold group
    for k in range(K):
        val_idx = folds[k]
        train_idx = np.hstack([folds[i] for i in range(K) if i != k])

        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_val, y_val = x_train[val_idx], y_train[val_idx]

        X_tr = design_matrix(x_tr, deg)
        X_val = design_matrix(x_val, deg)

        coeffs, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        y_val_pred = X_val @ coeffs

        mse = np.mean((y_val - y_val_pred)**2)
        fold_mse.append(mse)

    cv_mse[deg] = np.mean(fold_mse)
    print(f"Degree {deg} | 5-fold CV MSE = {cv_mse[deg]:.6f}")
```
c.  
```{code-cell}


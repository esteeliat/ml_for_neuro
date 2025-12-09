---
title: Homework 3
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---
## Question 1:  
(a) 
```{code-cell}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = 300  # number of training samples
sigma = np.sqrt(0.05)  # noise standard deviation

rng = np.random.default_rng(7)
x = rng.uniform(0.0, 1.0, size=N)

# Ground-truth parameters
R_max = 1.0
C50 = 0.25
n_exp = 2.0

def naka_rushton(x, R_max=R_max, C50=C50, n=n_exp):
    return R_max * (x**n) / (x**n + C50**n + 1e-12)

y_clean = naka_rushton(x)
noise = rng.normal(0.0, sigma, size=N)  # observation noise
y = y_clean + noise
```
(b)+(c)
```{code-cell}
def design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)

fits = {}
train_mse = {}

for deg in range(4):
    X = design_matrix(x, deg)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred)**2)

    fits[deg] = coeffs
    train_mse[deg] = mse

# Show summary table
rows = []
for deg in range(4):
    row = {"degree": deg, "train_mse": train_mse[deg]}
    for i, c in enumerate(fits[deg]):
        row[f"c{i}"] = c
    rows.append(row)

df = pd.DataFrame(rows)
print("Training MSE and coefficients:")
print(df)

#(C)
x_plot = np.linspace(0, 1, 400)
y_true_plot = naka_rushton(x_plot)

plt.figure(figsize=(9, 6))
plt.scatter(x, y, s=18, alpha=0.6, label="Noisy training data")
plt.plot(x_plot, y_true_plot, linewidth=2, label="True function")

for deg in range(4):
    coeffs = fits[deg]
    X_plot = design_matrix(x_plot, deg)
    y_fit_plot = X_plot @ coeffs
    plt.plot(x_plot, y_fit_plot, linewidth=2, label=f"Degree {deg} fit")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression Fits (degrees 0–3)")
plt.legend()
plt.grid(True)
plt.show()
```

(d)
```{code-cell}
# --- Generate test set ---
N_test = 300
rng = np.random.default_rng(123)   # different seed for test set
x_test = rng.uniform(0.0, 1.0, size=N_test)

# true noise-free values
y_test_clean = naka_rushton(x_test)

# noisy observations
noise_test = rng.normal(0.0, np.sqrt(0.05), size=N_test)
y_test = y_test_clean + noise_test

# --- Compute MSE for each polynomial model ---
test_mse = {}

for deg in range(4):
    coeffs = fits[deg]              # from part (b)
    X_test = design_matrix(x_test, deg)
    y_pred_test = X_test.dot(coeffs)
    mse = np.mean((y_test - y_pred_test)**2)
    test_mse[deg] = mse
    print(f"Degree {deg} test MSE: {mse:.6f}")
```
after generatint the test set and calculating the MSE for each model we get:  
Degree 0 test MSE: 0.138755  
Degree 1 test MSE: 0.062089  
Degree 2 test MSE: 0.055201  
Degree 3 test MSE: 0.054719  

(e)  We can see by the results of (d) that degree 0 is too simplified, in degree 1 we are getting better but still the MSE are higher that 2,3. We can also see in the plot, that Naka–Rushton model has curve and degrees 0,1 can't capture it. In degree 2 we are getting even a lower MSE but still this model has some struggles to capture the curve. Therefore, degree 3 with the lowest MSE and we can see this is the first polynomial flexible enough to approximate the shape. Therefore, the Degree 3 model preformes the best.

## Question 2:

### part a: Split the data into training (80%) and test sets (20%)
```{code-cell}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, RocCurveDisplay, auc, confusion_matrix, ConfusionMatrixDisplay 
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('resources/hw-3-q2-data.csv')

INPUT_PARAMS = ['score_STAI_state_short', 'score_TAI_short', 'score_GAD', 'score_BFI_N']
TARGET = 'score_AMAS_total'

display(df[INPUT_PARAMS + [TARGET]].info(show_counts=True))

X = df[INPUT_PARAMS]  
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.2)
```

### part b: Transform the score_AMAS_total to binary (high and low) 
```{code-cell}
amas_mean = y_train.mean()
y_train = (y_train > amas_mean).astype(int)
y_test = (y_test > amas_mean).astype(int)
```

### part c: Calculate the accuracy of a baseline model (which always predicts the majority class) 
```{code-cell}
majority_class = y_train.value_counts().idxmax()
y_pred = [majority_class] * len(y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy is: {round(accuracy, 2)}")
```

### part d: Train a Logistic regression model:
```{code-cell}
model = LogisticRegression(random_state=0, penalty=None, solver="newton-cg")
_ = model.fit(X_train, y_train)

#i. Print the parameters of the fitted model.
print(f"Intercept: {model.intercept_}")

coefficients_table = pd.DataFrame({'Column': INPUT_PARAMS, 'Coefficients': model.coef_[0]})
print("Coefficients:")
display(coefficients_table)
```

```{code-cell}
#ii. Calculate the Error Rate (Misclassification Rate) and Accuracy on the test set. Which threshold value have you used?
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Error rate: {round(1 - accuracy, 2)}, Accuracy: {round(accuracy, 2)}")
```

```{code-cell}
#iii. Plot the ROC curve and calculate the AUC.
roc_curve = RocCurveDisplay.from_estimator(model, X_test, y_test)
print(f"AUC: {round(roc_curve.roc_auc, 2)}")
```

```{code-cell}
#iv. For theshold values of 0.3 and 0.7, print the confusion matrix and calculate the Sensitivity and Specificity. 
probability = model.predict_proba(X_test)[:, 1]

THRESHOLDS = [0.3, 0.7]

for threshold in THRESHOLDS:
    print(f"Results for threshold {threshold}")
    predictions = (probability >= threshold).astype(int)
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    false_negative = cm[1, 0]
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    true_positive = cm[1, 1]
    
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    print(f"Sensitivity: {round(sensitivity, 2)}; Specificity: {round(specificity, 2)}\n")



```
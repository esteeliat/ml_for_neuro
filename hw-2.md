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

we then get that 
## final answers: $$L(θ)=$$ $${θ^{n_1}}$$ * ($${2(1-θ) \over 3}){^{n_2}}$$ * ($${(1-θ) \over 3}){^{n_3}}$$ = $${2^{{n_2}} \over 3^{n_2+n_3}}$$ * $${θ^{n_1}}$$ * $${(1-θ)^{n_2+n_3}}$$
## d.
log likelihood function - $$l(θ)=log(L(θ))$$

so $$l(θ) =$$ log($${2^{{n_2}} \over 3^{n_2+n_3}}$$ * $${θ^{n_1}}$$ * $${(1-θ)^{n_2+n_3}}$$) = $${n_2}log(2)-({n_2+n_3})log(3)+{n_1}log(θ)+({n_2+n_3})log(1-θ)$$

now in order to find the value of $$θ$$ that maximizes it we will Differentiate $$l$$ by $$θ$$: 
$$dl \over dθ$$ = $${n_1} \over θ$$ - $${n_2+n_3} \over 1-θ$$

because the first two termsare not depended on $$θ$$ and the derivative of log(θ) is $$1 \over θ$$. also, there is a minus over $${n_2+n_3} \over 1-θ$$ as we derivative $$log(1-θ)$$ by $$θ$$.

now in order to find the maximum we will compare to zero the derivative:
$${n_1} \over θ$$ = $${n_2+n_3} \over 1-θ$$ so $${n_1}(1-θ)=({n_2+n_3})θ$$

=> $$n_1$$ = $$θ({n_1+n_2+n_3})$$ = $$θn$$ => 
## final answers: $$\hatθ_{MLE}$$ = $$n_1 \over n$$
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
## final answers: $$MSE(\hatθ)={θ(1-θ) \over n}$$
## Question 2:
### Overall Takeaway
Missing Data: symptoms_other_text, Survival_data, No-residual (Post-surgery) columns all have > 98% missing data and so should be dropped from the analysis. offset_treatment_date is missing in 90% of rows, spanning 69% of patients - but there will only ever be an offset_treatment_date if the treatment_plan was SRS or Surgery so a null value can be expected as long as treatment_plan is 'Routine surveillance'. When checking for this, there are only 2 rows where this rule is broken so we should remove those two rows. Similarly, previous_treatment is null in 83% of rows - but a null value just indicates the patient didn't have any previous treatment (not that data is necessarily missing) and so null values should be replaced with "No previous treatment". age_at_mri, sex_birth, and ethnicity are missing in 14%-20% of patients depending on the variable. Since these variables should not change from visit to visit - for those with multiple visits and data in at least 1 visit - we can impute the data. For patients without any data available - we will drop all their data (luckily there is lots of overlap between the columns so this still leaves us ~79.5% of the data). Lastly, symptoms and treatment_plan have <1% missing data - we should drop patients with missing data.
Age is the only numberical data point, while offset_treatment_date and offset_imagine_date are dates and the rest are categorical variables. A breakdown of the distributions for all main variables can be seen below. Some interesting takeaways from the distribution: 
1) sex is fairly evenly distributed when looking by patient, but when looking by scan - there are more men than women showing that men have more frequent scans
2) ethnicity is mainly white (of varrying origins) which means the study will lack diversity
3) there is a range for age at first scan - but the bulk are in  the 50-70 range
4) treatment plan skews heavily towards surveillance as mentioned in the missing data section
5) there is a fairly even distribution regarding the number of scans each person took, with most people having more than 1 scan indicating an analysis on changes in scans could be done
6) Hearing loss is the predominant symptom, but there are enough data points for some other symptoms that could also be examined


### Understand Columns
```{code-cell}
:label: markdown-myst
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./resources/hw-2-q-2-data.tsv', sep='\t')
display(df.info(show_counts=True))
print(f"{len(df)} rows covering {df['patient_id'].nunique()} Patients")
```
### Missing Data
```{code-cell}
null_counts = df.isnull().sum()
fraction_missing = null_counts / len(df) * 100

print("% Missing Values per Column")
display(fraction_missing.sort_values(ascending=False))

print("% Missing Values per Patient")
patient_ids = df['patient_id'].dropna().unique()
notnull_patient = df.groupby('patient_id').apply(lambda x: x.drop('patient_id', axis=1).notnull().any()).astype(bool)
missing_per_patient = (~notnull_patient).sum(axis=0) / len(notnull_patient) * 100

print("Percent of patients with no not-null value for each column:")
display(missing_per_patient.sort_values(ascending=False))

# 1 - remove columns >90% missing data
cols_to_drop = missing_per_patient[missing_per_patient > 90].index.tolist()
df = df.drop(columns=cols_to_drop)
print(f"Dropped columns with >90% missing data: {cols_to_drop}")

# 2- deal with unexpected null offset_treatment_date
unexpected_null_offset_date = df[(df['treatment_plan'].isin(['Surgery', 'SRS'])) & (df['offset_treatment_date'].isnull())]
ids_to_drop = unexpected_null_offset_date['patient_id'].unique()
df = df[~df['patient_id'].isin(ids_to_drop)]
print(f"Dropped {len(ids_to_drop)} patients because of unexpected null offset_treatment")

# 3 - replace null previous_treatment
df['previous_treatment'] = df['previous_treatment'].fillna("no previous treatment")

# 4 - remove anyone missing any column with <25% missing data
cols_with_less_25_missing = missing_per_patient[missing_per_patient < 25].index.tolist()
missing_any_demographic = notnull_patient[~notnull_patient[cols_with_less_25_missing].all(axis=1)]
ids_to_drop = missing_any_demographic.index.unique()
df = df[~df['patient_id'].isin(ids_to_drop)]
print(f"Dropped {len(ids_to_drop)} patients because of missing demographic data")
```

### types of variables and disributions
```{code-cell}
def show_cat_distr(data, col):
  print(f"\n=== {col}: value counts ===")
  print(data[col].value_counts(dropna=False))
  plt.figure(figsize=(7,4))
  data[col].value_counts(dropna=False).head(10).plot(kind="bar")
  plt.title(f"{col} distribution")
  plt.tight_layout()
  plt.show()

# First show the demographic variables which should be by patient
## Aggregate 'sex', 'ethnicity', and 'age' (using the minimum age) by patient
agg_demo = df.groupby('patient_id').agg({
    'sex_birth': 'first',
    'ethnicity': 'first',
    'age_at_mri': 'min'
}).reset_index()

# Show value distributions for sex and ethnicity
print("Sex by patient")
show_cat_distr(agg_demo, 'sex_birth')
print("Sex by Scan")
show_cat_distr(df, 'sex_birth')
show_cat_distr(agg_demo, 'ethnicity')
plt.figure(figsize=(7,4))
sns.violinplot(data=agg_demo, x="age_at_mri")
plt.title("Age at first scan distribution (by patient)")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

# Now switch to scan specific data 
show_cat_distr(df, 'treatment_plan')

# Show the distribution of offset_imaging_date (which is a date)
offset_imaging_dates = pd.to_datetime(df['offset_imaging_date'], errors='coerce')
plt.figure(figsize=(10, 4))
plt.hist(offset_imaging_dates, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Offset Imaging Date")
plt.ylabel("Number of scans")
plt.title("Distribution of Offset Imaging Date")
plt.tight_layout()
plt.show()

# Next look at the number of scans per person:
# Efficient bar chart of number of scans per patient
df['patient_id'].value_counts().value_counts().sort_index().plot(
    kind="bar",
    figsize=(7, 4)
)
plt.xlabel("Number of scans")
plt.ylabel("Number of patients")
plt.title("Distribution of Number of Scans")
plt.tight_layout()
plt.show()

# Show the distribution of symptoms
all_symptoms = df['symptoms'].dropna().str.split(',')
symptom_list = [sym.strip() for sublist in all_symptoms for sym in sublist]
symptom_df = pd.DataFrame({'symptom': symptom_list})
show_cat_distr(symptom_df, 'symptom')
```


## Question 3:

### 80:20 split
```{code-cell}
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df = pd.read_csv('./resources/neuron.csv')
X = df[['x']]  # Ensure X is 2D for sklearn
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0,
    test_size=0.2
)
```
### Train Linear Regression
```{code-cell}
# Linear regression to x
linear_model = LinearRegression()
_ = linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Linear regression to polynomial features of x (degree 5)
poly_model = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
_ = poly_model.fit(X_train, y_train)
poly_linear_predictions = poly_model.predict(X_test)
```

### Evaluation Linear Regression
```{code-cell}
from sklearn.metrics import mean_squared_error, r2_score
from time import sleep

def evaluate(predictions, title):
    mse = mean_squared_error(y_test, predictions)
    fig, ax = plt.subplots(figsize=(15, 7))
    _ = plt.scatter(X_test, y_test, color="black", label="Real Values")
    _ = plt.scatter(X_test, predictions, color="red", label="Predicted Values")

    sorted_df = X_test.copy()
    sorted_df["pred"] = predictions
    sorted_df = sorted_df.sort_values(by=X_test.columns[0])
    X_sorted = sorted_df[X_test.columns[0]]
    pred_sorted = sorted_df["pred"]
    _ = plt.plot(X_sorted, pred_sorted, color="blue", label="Regression Line")
    _ = plt.legend()
    plt.title(title)
    plt.show()
    print(f'Mean Squared Error: {mse:.2f}')

print("Evaluate LR Predictions")
evaluate(linear_predictions, "Linear Regression Predictions")
print("Evaluate LR Predictions to Polynomial Features of X")
evaluate(poly_linear_predictions, "Polynomial Regression Predictions (Degree 5)")

```
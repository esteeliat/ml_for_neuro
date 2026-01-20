---
title: Homework 5
authors: Estee Rebibo (949968879) and Eden Moran (209185107) 
kernelspec:
  name: python3
  display_name: 'Python 3'
---

## Question 1:  
a.  
```{code-cell}
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# load dataset and extract text and labels
df = pd.read_csv("/mnt/data/reel_synopses_synthetic_dataset.csv")
documents = df["synopsis"].values
y = df["label"]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)

X_tfidf = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", X_tfidf.shape)
print("Number of non-zero entries:", X_tfidf.nnz)
```
b.
```{code-cell}
# train/test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# random forest + CV (AUC)
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]}

rf_cv = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_

# random forest test AUC
rf_probs = best_rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print("Random Forest Test AUC:", rf_auc)
print("Best RF parameters:", rf_cv.best_params_)

# random forest feature importance
rf_importances = best_rf.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1][:10]
rf_top_features = [(feature_names[i], rf_importances[i]) for i in rf_indices]

print("\nTop Random Forest features:")
for feat, val in rf_top_features:
    print(f"{feat}: {val:.4f}")

# Gradient Boosting + CV (AUC)
gb = GradientBoostingClassifier(random_state=42)
gb_param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}

gb_cv = GridSearchCV(estimator=gb, param_grid=gb_param_grid, cv=5, scoring="roc_auc")
gb_cv.fit(X_train, y_train)
best_gb = gb_cv.best_estimator_

# Gradient Boosting test AUC
gb_probs = best_gb.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_probs)

print("\nGradient Boosting Test AUC:", gb_auc)
print("Best GB parameters:", gb_cv.best_params_)

# Gradient Boosting feature importance
gb_importances = best_gb.feature_importances_
gb_indices = np.argsort(gb_importances)[::-1][:10]

gb_top_features = [(feature_names[i], gb_importances[i]) for i in gb_indices]

print("\nTop Gradient Boosting features:")
for feat, val in gb_top_features:
    print(f"{feat}: {val:.4f}")
```
c.  
we choose the gradient boosting model
```{code-cell}
import shap
# using a small background set for SHAP
background = X_train[:100]
explainer = shap.Explainer(best_gb, background)

# choosing the first test sample as record
idx = 0
X_instance = X_test[idx]
shap_values = explainer(X_instance)

# force plot
shap.initjs()

shap.force_plot(explainer.expected_value, shap_values.values, X_instance.toarray(), feature_names=feature_names)
```
d.  
```{code-cell}
import shap

# initialize JS for plots 
shap.initjs()

# create SHAP explainer for the chosen model (Gradient Boosting)
explainer = shap.Explainer(best_gb, X_train[:100])

# compute SHAP values for multiple test samples
X_sample = X_test[:100]
shap_values = explainer(X_sample)

# Global feature importance (bar plot)
shap.summary_plot( shap_values.values, X_sample.toarray(), feature_names=feature_names, plot_type="bar")

# detailed beeswarm plot
shap.summary_plot(shap_values.values, X_sample.toarray(), feature_names=feature_names)
```

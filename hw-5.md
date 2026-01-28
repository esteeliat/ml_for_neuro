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
df = pd.read_csv("resources/hw-5-q-1-data.csv")
documents = df["synopsis"].values
y = (df["was_that_reel_very_successful"] == "Yes").astype(int)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95, min_df=2)
X_tfidf = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", X_tfidf.shape)
print("Number of non-zero entries:", X_tfidf.nnz)
```
b.
```{code-cell}
# Train / Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# random forest + CV (AUC)
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],}

rf_cv = GridSearchCV(rf, rf_param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_

rf_probs = best_rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print("\nRandom Forest Test AUC:", rf_auc)
print("Best RF parameters:", rf_cv.best_params_)

# Random Forest feature importance
rf_importances = best_rf.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1][:10]

print("\nTop Random Forest features:")
for i in rf_indices:
    print(f"{feature_names[i]}: {rf_importances[i]:.4f}")

# Gradient Boosting + CV (AUC)
gb = GradientBoostingClassifier(random_state=42)
gb_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],}

gb_cv = GridSearchCV(gb, gb_param_grid, cv=5, scoring="roc_auc")
gb_cv.fit(X_train, y_train)
best_gb = gb_cv.best_estimator_

gb_probs = best_gb.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, gb_probs)

print("\nGradient Boosting Test AUC:", gb_auc)
print("Best GB parameters:", gb_cv.best_params_)

# Gradient Boosting feature importance
gb_importances = best_gb.feature_importances_
gb_indices = np.argsort(gb_importances)[::-1][:10]

print("\nTop Gradient Boosting features:")
for i in gb_indices:
    print(f"{feature_names[i]}: {gb_importances[i]:.4f}")
```
based on the Cross Validation results, the Gradient Boosting classifier achieved a higher test AUC (0.798) compared to the Random Forest classifier (0.772), indicating better discriminative performance and generalization.
c.  
We choose the gradient boosting model.  
```{code-cell}
import shap
shap.initjs()

# Convert small subset to dense (to avoid memory issues)
X_train_dense_small = X_train[:500].toarray()
X_test_dense_small = X_test[:500].toarray()

# Create SHAP explainer with background
explainer = shap.Explainer(best_gb, X_train_dense_small)

# Example for one instance
idx = 0
X_instance = X_test_dense_small[idx:idx+1]
shap_values_instance = explainer(X_instance)
shap.force_plot(
    explainer.expected_value,
    shap_values_instance.values,
    X_instance,
    feature_names=feature_names)
```
Based on the results of a., the Gradient Boosting classifier was selected as the model.
d.  
```{code-cell}
import shap
shap.initjs()

shap_values_sample = explainer(X_test_dense_small)

# Bar plot for global importance
shap.summary_plot(
    shap_values_sample.values,
    X_test_dense_small,
    feature_names=feature_names,
    plot_type="bar")

# Beeswarm plot
shap.summary_plot(
    shap_values_sample.values,
    X_test_dense_small,
    feature_names=feature_names)
```
the The SHAP bar plot shows that words such as camera, moment, share, park, smile, and laugh have the highest mean absolute SHAP values, therefore they are the most important features across the dataset. The SHAP beeswarm plot further reveals the direction of each feature’s influence. High TF-IDF values (red points) for features like camera, moment, share, smile, and laugh are predominantly associated with positive SHAP values, meaning they increase the predicted probability that a reel is successful. Overall, this confirm that the model relies on semantically meaningful and intuitive cues.

## Question 2:

b. 
```{code-cell}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('resources/hw-5-q-2-data/variables.csv')

# Select columns related to 'slow', 'preferred', and 'fast' walking speeds
columns_of_interest = [col for col in df.columns if "slow" in col.lower() or "preferred"
in col.lower() or "fast" in col.lower()]
df_subset = df[columns_of_interest]
# Calculate the Pearson correlation matrix for the subset
corr_matrix_subset = df_subset.corr()
# Generate the clustered heatmap with hierarchical clustering applied to both rows and columns
clustermap = sns.clustermap(corr_matrix_subset, method='ward', cmap='coolwarm', figsize=
(15, 15), linewidths=.5)
plt.title('Clustered Heatmap of Correlated Features', pad=90)
plt.show()
```

c. Apply PCA on the 27 gait features listed below.

```{code-cell}
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

N_COMPONENTS = len(columns_of_interest)

scaler = StandardScaler()
X = df[columns_of_interest]
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=N_COMPONENTS)
gait_pca = pca.fit_transform(X_scaled)
```

(i) Plot a scatter plot where the axes are the first and second principal components scores (PC1 and PC2).
```{code-cell}
plt.figure(figsize=(8, 6))
plt.scatter(gait_pca[:, 0], gait_pca[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()
```

(ii) Plot the loadings of PC1 and PC2. Can you interprate PC1 and PC2 based on the loadings vectors?
```{code-cell}
loadings = pd.DataFrame(pca.components_.T[:, :2], columns=['PC1', 'PC2'], index=columns_of_interest)
plt.figure(figsize=(10, 10))
plt.scatter(loadings['PC1'], loadings['PC2'])

# Annotate each point with the feature name
for i, feature in enumerate(loadings.index):
    plt.annotate(feature, (loadings['PC1'].iloc[i], loadings['PC2'].iloc[i]), fontsize=9, alpha=0.75)

plt.xlabel('PC1 Loading')
plt.ylabel('PC2 Loading')
plt.title('Scatter Plot of PCA Loadings (PC1 vs PC2)')

plt.grid(True)
plt.show()
```
Based on the graph - we can see that the stride_length variables (for both slow, fast and preffered) have the strongest effect on both PC1 and PC2, whereas the stride_time variables have a minimal effect on PC1, but strong effect on PC2. 

(iii) Plot the explained variance ratio plot.
```{code-cell}
plt.figure(figsize=(9, 6))
x_range = range(1, N_COMPONENTS + 1)
plt.plot(x_range, pca.explained_variance_ratio_, label='Individual Component')
plt.plot(x_range, np.cumsum(pca.explained_variance_ratio_), label='Cumulative')

plt.legend(loc="best")

plt.xticks(x_range)
plt.xlabel("Number ofPrincipal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio by Number of Principal Components")
plt.tight_layout()
plt.show()
```

(iv) Apply k-means on PC1 and PC2 with and plot the results (Standardize first). Mark with an "X" the
clusters centers.
```{code-cell}
pc12_scores = scaler.fit_transform(gait_pca[:, :2])

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(pc12_scores)
centers = kmeans.cluster_centers_

plt.figure(figsize=(9, 7))
scatter = plt.scatter(pc12_scores[:, 0], pc12_scores[:, 1], c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.xlabel("Standardized PC1")
plt.ylabel("Standardized PC2")
plt.title("K-means (K=3) Clustering of StandardizedPC1 and PC2")
plt.grid(True)
plt.show()
```
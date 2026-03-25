---
title: Final_Project
authors: Estee Rebibo (949968879) and Eden Moran (209185107)
kernelspec:
  name: python3
  display_name: 'Python 3'
---
# 1. Loading the data set and asking basic questions on the dataset
```{code-cell}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score,classification_report, f1_score, ConfusionMatrixDisplay
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import seaborn as sns
from collections import Counter
import re

df = pd.read_json(r"resources/emoset_challenge_1000_augmented.json")

# unpacking the annotations dictionary and merging back into the main data frame
annotations_df = df['annotations'].apply(pd.Series)
df = pd.concat([df, annotations_df], axis=1) # sticking the new column we got side by side in the original table


# asking questions in order to understand the dataset
display(df.info(show_counts=True))
#pd.set_option('display.max_columns', None)
display("Duplicate image names:", df['image_name'].duplicated().sum()) # checking image names looking for bias
display("Embedding type:", type(df.loc[0, 'embedding'])) # checking embeddings
display("Embedding length summary:", df['embedding'].apply(len).describe()) # Check embedding length consistency
```

## Conclusions:
in the dataset there is 1000 images samples and 7 main columns. columns including: image identifiers, textual descriptions, viewer feeling descriptions, metadata annotations, and multiple embedding representations. the emotion label and another metadata are stored within a nested annotation dictionary. all embedding vectors have 512 dimensions implying there is consistency. No duplicating images where found - implying no bias in sampling.  

# 2. Asking the important questions regarding the dataset  
```{code-cell}
# Handle Missing Data

## brightness and colorfulness have a low % missing, so drop rows with missing values
df = df.dropna(subset=['brightness', 'colorfulness'])

df.info(show_counts=True)
```

```{code-cell}
# emotional label is inside a dictionary 'annotations', extracting it into its own columns  
df['emotion'] = df['annotations'].apply(lambda x: x['emotion'])
display("Emotion classes =", ", ".join(df['emotion'].unique()))
display("Number of emotion classes =", df['emotion'].nunique())
# counting samples per emotion
emotion_counts = df['emotion'].value_counts()
display(emotion_counts)
# Plotting class distribution
emotion_counts.plot(kind='bar')
plt.title("Emotion class distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of samples")
plt.show()
```

## Conclusions:
For missing data - brightness and colorfulness have a low % missing, so drop rows with missing values. The other items with missing data are categorical and can be one-hot encoded to include a "missing" variable.

the target variable in this project is emotion (extracted from 'annotation'). We found 8 emotions contained in the
dataset: anger, amusement, awe, contentment, disgust, excitement, fear, and sadness. each emotion has exactly 125 samples to represent the class that gives us a perfectly balanced dataset as we can see in the plot.  


# 3. Preparing the metadata
```{code-cell}
numeric_metadata = ['brightness', 'colorfulness']
categorical_metadata = ['facial_expression', 'human_action', 'scene']
display("Distribution of numeric metadata = ")
display(df[numeric_metadata].describe())
```

# 4.numeric and categorical metadata  
```{code-cell}
# 4.1 Distribution of numeric
df[numeric_metadata].hist(bins=30, figsize=(8, 4))
plt.suptitle("Distribution of numeric metadata features")
plt.show()
# checking if different emotions tend to have different brightness or colorfulness
df = df.loc[:, ~df.columns.duplicated()] # dropping the first emotions column
for feature in numeric_metadata:
    df.boxplot(column=feature, by='emotion')
    plt.title(f"{feature} by emotion")
    plt.suptitle("")  # removes automatic title
    plt.xlabel("Emotion")
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.show()

top_k = 10  # number of categories to keep

# 4.2 Distribution of categorical
for feature in categorical_metadata:
    display(f"\n--- {feature.upper()} DISTRIBUTION ---")
    # count occurrences (including NaNs)
    counts = df[feature].value_counts(dropna=False)
    display(counts)

    # If feature has many categories → limit to top-K
    if counts.dropna().shape[0] > top_k:
        top_categories = counts.dropna().head(top_k).index
        filtered_df = df[df[feature].isin(top_categories)]
    else:
        filtered_df = df
    by_emotion = pd.crosstab(filtered_df['emotion'], filtered_df[feature])
    display(by_emotion)

    # plotting
    by_emotion.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title(f"{feature.replace('_', ' ').title()} Distribution by Emotion")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(
        title=feature.replace('_', ' ').title(),
        bbox_to_anchor=(1.05, 1)
    )
    plt.tight_layout()
    plt.show()
```

# 5. Text EDA
```{code-cell}
text_columns = ['description', 'viewer_feelings']

for col in text_columns: #Are there NaNs and Is one column much sparser than the other?
    display(f"\n--- {col.upper()} ---")
    display(df[col].isna().value_counts())

for col in text_columns: #Text length distributions
    df[f'{col}_length'] = df[col].fillna("").apply(lambda x: len(x.split()))

#Plot histograms
for col in text_columns:
    plt.figure(figsize=(8, 4))
    df[f'{col}_length'].hist(bins=30)
    plt.title(f"Word Count Distribution: {col}")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.show()

#Compare statistics numerically
df[[f'{col}_length' for col in text_columns]].describe()
#Vocabulary size
import re
def get_vocab(text_series):
    words = []
    for text in text_series.dropna():
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words.extend(text.split())
    return set(words)

for col in text_columns:
    vocab = get_vocab(df[col])
    display(f"{col} vocabulary size: {len(vocab)}")
#Most frequent words
from collections import Counter
def top_words(text_series, top_k=20):
    words = []
    for text in text_series.dropna():
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words.extend(text.split())
    return Counter(words).most_common(top_k)
for col in text_columns:
    display(f"\nTop words in {col}:")
    for word, count in top_words(df[col]):
        display(f"{word}: {count}")
```

# 6. If we squish these high-dimensional embeddings into 2D, will images with the same emotion ends up closer  
```{code-cell}
def display_pca_by_emotion(n_components, x, y, title):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(x)

    #Plot and color by emotion
    plt.figure(figsize=(8, 6))

    for emotion in y.unique():
        idx = y == emotion
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=emotion,
            alpha=0.6
        )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    display("Explained variance ratio:", pca.explained_variance_ratio_)
    display("Total explained variance:", pca.explained_variance_ratio_.sum())
```

```{code-cell}
#Convert embeddings into a matrix
X_image = np.vstack(df['embedding'].values)
y = df['emotion']

display_pca_by_emotion(0.9, X_image, y, "PCA of Image Embeddings (colored by emotion)")

X_text = np.vstack(df['viewer_feelings_embedding'].values)
```

# 7. TF - IDF 
```{code-cell}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# SPLIT DATA
y = df['emotion']
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=y)

y_train = train_df['emotion']
y_test = test_df['emotion']

# TF-IDF PARAMETERS
vectorizer_params = {
    "sublinear_tf": True,
    'ngram_range': (1, 2),
    "stop_words": "english"
}

# ANALYSIS FUNCTION
def analyze_tfidf(vectorizer, X, y, title=""):
    print(f"\n--- {title} ---")
    print(f"TF–IDF shape: {X.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Top words per emotion
    for emotion in y.unique():
        idx = (y == emotion).values
        mean_tfidf = X[idx].mean(axis=0)
        top_indices = np.argsort(mean_tfidf.A1)[-10:]
        print(f"\nTop words for emotion: {emotion}")
        print(feature_names[top_indices])

    # PCA visualization
    X_dense = X.toarray()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_dense)
    plt.figure(figsize=(8, 6))
    for emotion in y.unique():
        idx = y == emotion
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=emotion,
            alpha=0.6
        )
    plt.title(f"PCA of TF-IDF ({title})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


# FEATURE EFFECT PLOT
def plot_feature_effects(vectorizer, X_train, y_train, clf, top_n=15, title=""):
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Mean TF-IDF across training data
    mean_tfidf = np.asarray(X_train.mean(axis=0)).ravel()

    # Compute effects
    effects = clf.coef_ * mean_tfidf

    # Select most important features
    top_indices = np.argsort(np.abs(effects).max(axis=0))[-top_n:]
    selected_features = feature_names[top_indices]
    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(clf.classes_):
        plt.barh(
            selected_features,
            effects[i, top_indices],
            alpha=0.6,
            label=class_label)
    plt.xlabel("Average Feature Effect")
    plt.title(f"Feature Effects ({title})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 7.1 VIEWER FEELINGS
print("\n==============================")
print("TF-IDF: VIEWER FEELINGS")
print("==============================")

feelings_vectorizer = TfidfVectorizer(**vectorizer_params)

X_train_feelings = feelings_vectorizer.fit_transform(train_df['viewer_feelings'])
X_test_feelings = feelings_vectorizer.transform(test_df['viewer_feelings'])

# Analysis
analyze_tfidf(feelings_vectorizer, X_train_feelings, y_train, "Viewer Feelings")

# Train model for interpretation
clf_feelings = LogisticRegression(max_iter=1000)
clf_feelings.fit(X_train_feelings, y_train)

# Feature effects plot 🔥
plot_feature_effects(
    feelings_vectorizer,
    X_train_feelings,
    y_train,
    clf_feelings,
    title="Viewer Feelings")

# 7.2 DESCRIPTION
print("\n==============================")
print("TF-IDF: DESCRIPTION")
print("==============================")

description_vectorizer = TfidfVectorizer(**vectorizer_params)

X_train_desc = description_vectorizer.fit_transform(train_df['description'])
X_test_desc = description_vectorizer.transform(test_df['description'])

# Analysis
analyze_tfidf(description_vectorizer, X_train_desc, y_train, "Description")

# Train model
clf_desc = LogisticRegression(max_iter=1000)
clf_desc.fit(X_train_desc, y_train)

# Feature effects plot 🔥
plot_feature_effects(
    description_vectorizer,
    X_train_desc,
    y_train,
    clf_desc,
    title="Description")

# 7.3 VOCABULARY OVERLAP
desc_vocab = set(description_vectorizer.vocabulary_.keys())
feel_vocab = set(feelings_vectorizer.vocabulary_.keys())
overlap = desc_vocab.intersection(feel_vocab)
print("\n--- Vocabulary Overlap ---")
print("Description vocab size:", len(desc_vocab))
print("Viewer feelings vocab size:", len(feel_vocab))
print("Shared vocabulary size:", len(overlap))
print("Overlap ratio (desc):", len(overlap) / len(desc_vocab))
print("Overlap ratio (feelings):", len(overlap) / len(feel_vocab))
```

# 8: Train a model that predicts the emotions based on tabular dataset. 
Features: metadata, words from description , words from viewer feeling (refer to assignment 5 q1). You are required to explain the model.

```{code-cell}
df.info(show_counts=True)

# One-hot encode categorical metadata columns
categorical_cols = ['facial_expression', 'human_action', 'scene'] ## todo: we left out object because its a list...should we leave out otjer columns that dont have a lot of datA?
numeric_cols = ['brightness', 'colorfulness']


# One-hot encoding for categorical variables (handle NaNs as a separate category)
metadata_categorical = pd.get_dummies(df[categorical_cols].fillna('missing'), prefix=categorical_cols)
metadata_numeric = df[numeric_cols]

# Concatenate numeric + one-hot encoded categorical
metadata_features = pd.concat([metadata_numeric, metadata_categorical], axis=1)

# Convert metadata_features to same shape/type for sparse matrix concatenation

# Ensure the metadata is float or int before converting to sparse to avoid dtype=object issues
metadata_float = metadata_features.astype(float)
metadata_sparse = sparse.csr_matrix(metadata_float.values)

# Combine all features: metadata, viewer_feelings_tfidf, description_tfidf
X = sparse.hstack([metadata_sparse, viewer_feelings_tfidf, description_tfidf])

# Create feature_names for all columns in X
metadata_feature_names = metadata_features.columns.tolist()
viewer_feelings_feature_names = ['viewer_feelings_tfidf_' + str(word) for word in feelings_vectorizer.get_feature_names_out()]
description_feature_names = ['description_tfidf_' + str(word) for word in description_vectorizer.get_feature_names_out()]
feature_names = metadata_feature_names + viewer_feelings_feature_names + description_feature_names

y = df[['emotion']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

display(f'Number of features: {len(feature_names)}')
```

```{code-cell}
classifiers = [ 
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    RidgeClassifier(tol=1e-2, solver="sparse_cg"),
    KNeighborsClassifier(n_neighbors=100),
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
]

USE_GRID_SEARCH = False

for clf in classifiers:
    print(f"Training {clf.__class__.__name__}...")
   
    if USE_GRID_SEARCH:
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 20]
        }

        cv = GridSearchCV(clf, param_grid, cv=5)
        cv.fit(X_train, y_train)
        model = cv.best_estimator_
        print("Best parameters:", cv.best_params_)
    else:
        model = clf

    _ = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate performance - since have balanced classes, can use accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    #print(classification_report(y_test, y_pred))
    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    #ax.xaxis.set_ticklabels()
    #ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(
        f"Confusion Matrix for {clf.__class__.__name__}\n({accuracy * 100:.2f}% Accuracy)"
    )
    
    # Print top features

    # importances = model.feature_importances_ if hasattr(model, "feature_importances_") else np.sum(np.abs(model.coef_), axis=0)
    # indices = np.argsort(importances)[::-1][:10]
    # print("\nTop Features:")
    # for i in indices:
    #     print(f"{feature_names[i]}: {importances[i]:.4f}")
```

## Model Explanation:

# 9: Train a model that predicts the emotions based on the embedding. 
Features: embedding , description_embedding , viewer_feelings_embedding . Dimensionality reduction recommended.

```{code-cell}
## 9.1: Create X so it contains all embeddings, and Y is the emotion column
EMBEDDING_COLS = [
    'embedding',
    'viewer_feelings_embedding',
    'description_embedding'
]

embeddings_stacked = [np.vstack(df[col].values) for col in EMBEDDING_COLS]
X = np.hstack(embeddings_stacked)

y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
```

```{code-cell}
# 9.2 Conduct PCA 
print(f"Number of features: {len(X_train)}")
pca_test = PCA(n_components=len(X_train))
pca_test.fit(X_train)

explained_variance = pca_test.explained_variance_ratio_
cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

pca_df = pd.DataFrame()
pca_df['Cumulative Variance Ratio'] = cumulative_variance
pca_df['Explained Variance Ratio'] = explained_variance
display(pca_df.iloc[::50].head(20))
```

```{code-cell}
# 9.3: Select PCA - we'll use 450 components to explain 95% of the variance
pca = PCA(n_components=450)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

```{code-cell}
# 9.4 Train on several classifiers, ultimately picking the best one 
## classifiers include random forest, gradient boosting, and support vector machine
classifiers = [ 
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    MLPClassifier(random_state=42),
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
]

for clf in classifiers:
    print(f"Training {clf.__class__.__name__}...")
    _ = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    # Evaluate performance - since have balanced classes, can use accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print(classification_report(y_test, y_pred))
```

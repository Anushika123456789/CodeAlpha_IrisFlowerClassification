# Iris Flower Classification

CodeAlpha Data Science Internship — Task 1.

Classifies an iris flower as *setosa*, *versicolor*, or *virginica* from
four measurements (sepal length/width, petal length/width), comparing three
classic classifiers head-to-head rather than just fitting one and stopping.

## Dataset

The classic Iris dataset (150 rows, 4 numeric features, 3 balanced classes,
`data/Iris.csv`), originally collected by Edgar Anderson and popularized by
Ronald Fisher's 1936 paper.

## Approach

1. **Data check** — nulls, duplicates, class balance.
2. **EDA** — scatter plots of sepal vs. petal dimensions by species, feature
   correlation heatmap.
3. **Modeling** — Logistic Regression, K-Nearest Neighbors, and a Decision
   Tree, each evaluated on both a held-out test split and 5-fold
   cross-validation (so one lucky/unlucky split doesn't decide the winner).
4. **Evaluation** — classification report and confusion matrix for the best
   model, plus a feature-importance view when applicable.

## Results

| Model | Test accuracy | 5-fold CV accuracy |
|---|---|---|
| Logistic Regression | 0.933 | **0.960** |
| K-Nearest Neighbors | 0.933 | 0.960 |
| Decision Tree | 0.900 | 0.953 |

Logistic Regression is the pick (ties KNN on test accuracy, edges ahead on
cross-validation, and gives a linear decision boundary that's easy to reason
about). The few misclassifications are all between *versicolor* and
*virginica* — the only two species whose petal measurements overlap at all;
*setosa* is separated perfectly by every model.

## Repo structure

```
CodeAlpha_IrisFlowerClassification/
├── data/Iris.csv                           # raw dataset
├── Iris_Flower_Classification.ipynb        # full notebook, pre-executed with outputs
├── src/iris_classification.py              # same analysis as a plain script
├── outputs/                                 # saved charts (EDA, confusion matrix, model comparison)
├── build_notebook.py                        # generates the notebook + script from one source of truth
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python src/iris_classification.py      # plain script, or
jupyter notebook Iris_Flower_Classification.ipynb   # or open in VS Code's Jupyter extension
```

`src/iris_classification.py` also opens directly in VS Code as an
interactive notebook (Jupyter extension) — it uses the same `# %%` cell
markers, so you get the identical cell-by-cell view without needing `.ipynb`.

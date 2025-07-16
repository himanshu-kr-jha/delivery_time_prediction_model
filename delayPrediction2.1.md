```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
```


```python
# Load the dataset
df = pd.read_csv("./Food_Delivery_Times.csv")
```


```python
# Drop the Order_ID column
df = df.drop(columns=["Order_ID"])

# Separate features and target
X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]
```


```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
```


```python
# Preprocessing for numerical data
numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", RobustScaler()),
    ("feature_selection", SelectKBest(score_func=f_regression, k=3))
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
```


```python
# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)
```


```python
# Define models to evaluate
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0)
}
```


```python
# Evaluate each model and store metrics
metrics_list = []

for name, modelR in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", modelR)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')

    metrics_list.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2_Mean": cv_scores.mean(),
        "CV_R2_Std": cv_scores.std()
    })

metrics_df = pd.DataFrame(metrics_list)
```


```python
metrics_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R2</th>
      <th>CV_R2_Mean</th>
      <th>CV_R2_Std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RandomForest</td>
      <td>6.847050</td>
      <td>9.582953</td>
      <td>0.795119</td>
      <td>0.696034</td>
      <td>0.052932</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GradientBoosting</td>
      <td>6.598749</td>
      <td>9.389548</td>
      <td>0.803306</td>
      <td>0.712950</td>
      <td>0.066977</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>8.047232</td>
      <td>10.949979</td>
      <td>0.732497</td>
      <td>0.650938</td>
      <td>0.027185</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Identify the best model based on CV_R2_Mean
best_model_name = metrics_df.sort_values(by="CV_R2_Mean", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]
```


```python
best_model
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-1 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(random_state=42)</pre></div> </div></div></div></div>




```python
# Define hyperparameter grid for the best model
param_grid = {}
if best_model_name == "RandomForest":
    param_grid = {
        "regressor__n_estimators": [50,100, 200],
        "regressor__max_depth": [5, 10, 20]
    }
elif best_model_name == "GradientBoosting":
    param_grid = {
        "regressor__n_estimators": [50,100, 200],
        "regressor__learning_rate": [0.01,0.05, 0.1],
        "regressor__max_depth": [3,4,5]
    }
elif best_model_name == "XGBoost":
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [3, 5]
    }
```


```python
# Perform GridSearchCV
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", best_model)
])
```


```python
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
```


```python
print("Model Comparison Metrics:")
print(metrics_df)
print(f"\nBest Model: {best_model_name}")
print("Best Hyperparameters from GridSearchCV:")
print(best_params)
print(f"Best Cross-Validated R2 Score: {best_score:.4f}")


```

    Model Comparison Metrics:
                  Model       MAE       RMSE        R2  CV_R2_Mean  CV_R2_Std
    0      RandomForest  6.847050   9.582953  0.795119    0.696034   0.052932
    1  GradientBoosting  6.598749   9.389548  0.803306    0.712950   0.066977
    2           XGBoost  8.047232  10.949979  0.732497    0.650938   0.027185
    
    Best Model: GradientBoosting
    Best Hyperparameters from GridSearchCV:
    {'regressor__learning_rate': 0.1, 'regressor__max_depth': 3, 'regressor__n_estimators': 50}
    Best Cross-Validated R2 Score: 0.7204
    


```python
import joblib

# Save the best model
joblib.dump(grid_search.best_estimator_, "best_gradient_boosting_model.pkl")
```




    ['best_gradient_boosting_model.pkl']




```python
# Load the model
loaded_model = joblib.load("best_gradient_boosting_model.pkl")
```


```python
loaded_model
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   RobustScaler()),
                                                                  (&#x27;feature_selection&#x27;,
                                                                   SelectKBest(k=3,
                                                                               score_func=&lt;function f_regression at 0x0000015109BFF240&gt;))]),
                                                  [&#x27;Distance_km&#x27;,
                                                   &#x27;Preparation_Time_min&#x27;,
                                                   &#x27;Courier_Experience_yrs&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehot&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  [&#x27;Weather&#x27;, &#x27;Traffic_Level&#x27;,
                                                   &#x27;Time_of_Day&#x27;,
                                                   &#x27;Vehicle_Type&#x27;])])),
                (&#x27;regressor&#x27;,
                 GradientBoostingRegressor(n_estimators=50, random_state=42))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;scaler&#x27;,
                                                                   RobustScaler()),
                                                                  (&#x27;feature_selection&#x27;,
                                                                   SelectKBest(k=3,
                                                                               score_func=&lt;function f_regression at 0x0000015109BFF240&gt;))]),
                                                  [&#x27;Distance_km&#x27;,
                                                   &#x27;Preparation_Time_min&#x27;,
                                                   &#x27;Courier_Experience_yrs&#x27;]),
                                                 (&#x27;cat&#x27;,
                                                  Pipeline(steps=[(&#x27;imputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehot&#x27;,
                                                                   OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                                  [&#x27;Weather&#x27;, &#x27;Traffic_Level&#x27;,
                                                   &#x27;Time_of_Day&#x27;,
                                                   &#x27;Vehicle_Type&#x27;])])),
                (&#x27;regressor&#x27;,
                 GradientBoostingRegressor(n_estimators=50, random_state=42))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;, SimpleImputer()),
                                                 (&#x27;scaler&#x27;, RobustScaler()),
                                                 (&#x27;feature_selection&#x27;,
                                                  SelectKBest(k=3,
                                                              score_func=&lt;function f_regression at 0x0000015109BFF240&gt;))]),
                                 [&#x27;Distance_km&#x27;, &#x27;Preparation_Time_min&#x27;,
                                  &#x27;Courier_Experience_yrs&#x27;]),
                                (&#x27;cat&#x27;,
                                 Pipeline(steps=[(&#x27;imputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehot&#x27;,
                                                  OneHotEncoder(handle_unknown=&#x27;ignore&#x27;))]),
                                 [&#x27;Weather&#x27;, &#x27;Traffic_Level&#x27;, &#x27;Time_of_Day&#x27;,
                                  &#x27;Vehicle_Type&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>num</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Distance_km&#x27;, &#x27;Preparation_Time_min&#x27;, &#x27;Courier_Experience_yrs&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RobustScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RobustScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SelectKBest</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.feature_selection.SelectKBest.html">?<span>Documentation for SelectKBest</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SelectKBest(k=3, score_func=&lt;function f_regression at 0x0000015109BFF240&gt;)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted"><pre>[&#x27;Weather&#x27;, &#x27;Traffic_Level&#x27;, &#x27;Time_of_Day&#x27;, &#x27;Vehicle_Type&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(handle_unknown=&#x27;ignore&#x27;)</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GradientBoostingRegressor</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a></div></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(n_estimators=50, random_state=42)</pre></div> </div></div></div></div></div></div>




```python
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance_km</th>
      <th>Weather</th>
      <th>Traffic_Level</th>
      <th>Time_of_Day</th>
      <th>Vehicle_Type</th>
      <th>Preparation_Time_min</th>
      <th>Courier_Experience_yrs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>521</th>
      <td>5.30</td>
      <td>Clear</td>
      <td>Low</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>16</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>737</th>
      <td>10.46</td>
      <td>Clear</td>
      <td>NaN</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>25</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>740</th>
      <td>4.04</td>
      <td>Rainy</td>
      <td>High</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>14</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>660</th>
      <td>3.33</td>
      <td>NaN</td>
      <td>Medium</td>
      <td>Evening</td>
      <td>Scooter</td>
      <td>24</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>411</th>
      <td>17.44</td>
      <td>NaN</td>
      <td>Low</td>
      <td>Night</td>
      <td>Car</td>
      <td>23</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>408</th>
      <td>15.62</td>
      <td>Rainy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Scooter</td>
      <td>23</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>332</th>
      <td>1.80</td>
      <td>Clear</td>
      <td>NaN</td>
      <td>Night</td>
      <td>Bike</td>
      <td>14</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>208</th>
      <td>7.39</td>
      <td>Rainy</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Scooter</td>
      <td>25</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>613</th>
      <td>9.70</td>
      <td>Snowy</td>
      <td>Low</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>78</th>
      <td>3.46</td>
      <td>Snowy</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Scooter</td>
      <td>23</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 7 columns</p>
</div>




```python
# testingdata
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Define the number of dummy rows
num_rows = 20

# Define dummy data options
weather_options = ['Clear', 'Rainy', 'Cloudy', 'Foggy', 'Snowy']
traffic_options = ['Low', 'Medium', 'High']
time_of_day_options = ['Morning', 'Afternoon', 'Evening', 'Night']
vehicle_types = ['Bike', 'Car', 'Truck', 'Scooter']

# Generate dummy data
data = {
    'Distance_km': np.random.uniform(1, 1000, num_rows).round(2),
    'Weather': np.random.choice(weather_options, num_rows),
    'Traffic_Level': np.random.choice(traffic_options, num_rows),
    'Time_of_Day': np.random.choice(time_of_day_options, num_rows),
    'Vehicle_Type': np.random.choice(vehicle_types, num_rows),
    'Preparation_Time_min': np.random.randint(5, 60, num_rows),
    'Courier_Experience_yrs': np.random.uniform(0, 10, num_rows).round(1)
}

# Create DataFrame
newdata = pd.DataFrame(data)

# Display the first few rows
# print(df.head())

```


```python
newdata
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance_km</th>
      <th>Weather</th>
      <th>Traffic_Level</th>
      <th>Time_of_Day</th>
      <th>Vehicle_Type</th>
      <th>Preparation_Time_min</th>
      <th>Courier_Experience_yrs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>375.17</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Car</td>
      <td>48</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>950.76</td>
      <td>Rainy</td>
      <td>Low</td>
      <td>Night</td>
      <td>Truck</td>
      <td>12</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>732.26</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>28</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>599.06</td>
      <td>Foggy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Scooter</td>
      <td>15</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>156.86</td>
      <td>Cloudy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Car</td>
      <td>55</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>156.84</td>
      <td>Foggy</td>
      <td>Low</td>
      <td>Night</td>
      <td>Bike</td>
      <td>21</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>59.03</td>
      <td>Foggy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Scooter</td>
      <td>12</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>866.31</td>
      <td>Clear</td>
      <td>High</td>
      <td>Evening</td>
      <td>Scooter</td>
      <td>39</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>601.51</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Scooter</td>
      <td>39</td>
      <td>9.7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>708.36</td>
      <td>Snowy</td>
      <td>Low</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>37</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>21.56</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Bike</td>
      <td>9</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>969.94</td>
      <td>Snowy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>46</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>832.61</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Evening</td>
      <td>Truck</td>
      <td>43</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>213.13</td>
      <td>Rainy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Bike</td>
      <td>45</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>182.64</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>32</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>184.22</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>11</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>304.94</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Truck</td>
      <td>13</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>525.23</td>
      <td>Rainy</td>
      <td>Low</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>12</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>432.51</td>
      <td>Rainy</td>
      <td>High</td>
      <td>Night</td>
      <td>Scooter</td>
      <td>16</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>291.94</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>38</td>
      <td>4.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make predictions
predictions = loaded_model.predict(newdata)  # new_data should be a DataFrame with the same structure as the training data

```


```python

results = newdata.copy()
results["Predicted_Delivery_Time_min"] = predictions

results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Distance_km</th>
      <th>Weather</th>
      <th>Traffic_Level</th>
      <th>Time_of_Day</th>
      <th>Vehicle_Type</th>
      <th>Preparation_Time_min</th>
      <th>Courier_Experience_yrs</th>
      <th>Predicted_Delivery_Time_min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>375.17</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Car</td>
      <td>48</td>
      <td>1.0</td>
      <td>102.058747</td>
    </tr>
    <tr>
      <th>1</th>
      <td>950.76</td>
      <td>Rainy</td>
      <td>Low</td>
      <td>Night</td>
      <td>Truck</td>
      <td>12</td>
      <td>3.7</td>
      <td>77.482075</td>
    </tr>
    <tr>
      <th>2</th>
      <td>732.26</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>28</td>
      <td>6.7</td>
      <td>93.256982</td>
    </tr>
    <tr>
      <th>3</th>
      <td>599.06</td>
      <td>Foggy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Scooter</td>
      <td>15</td>
      <td>6.7</td>
      <td>82.092476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>156.86</td>
      <td>Cloudy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Car</td>
      <td>55</td>
      <td>5.9</td>
      <td>96.055781</td>
    </tr>
    <tr>
      <th>5</th>
      <td>156.84</td>
      <td>Foggy</td>
      <td>Low</td>
      <td>Night</td>
      <td>Bike</td>
      <td>21</td>
      <td>2.7</td>
      <td>92.328129</td>
    </tr>
    <tr>
      <th>6</th>
      <td>59.03</td>
      <td>Foggy</td>
      <td>High</td>
      <td>Afternoon</td>
      <td>Scooter</td>
      <td>12</td>
      <td>5.6</td>
      <td>79.701987</td>
    </tr>
    <tr>
      <th>7</th>
      <td>866.31</td>
      <td>Clear</td>
      <td>High</td>
      <td>Evening</td>
      <td>Scooter</td>
      <td>39</td>
      <td>3.8</td>
      <td>108.738084</td>
    </tr>
    <tr>
      <th>8</th>
      <td>601.51</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Scooter</td>
      <td>39</td>
      <td>9.7</td>
      <td>92.610142</td>
    </tr>
    <tr>
      <th>9</th>
      <td>708.36</td>
      <td>Snowy</td>
      <td>Low</td>
      <td>Evening</td>
      <td>Bike</td>
      <td>37</td>
      <td>8.5</td>
      <td>91.366498</td>
    </tr>
    <tr>
      <th>10</th>
      <td>21.56</td>
      <td>Cloudy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Bike</td>
      <td>9</td>
      <td>7.2</td>
      <td>69.416442</td>
    </tr>
    <tr>
      <th>11</th>
      <td>969.94</td>
      <td>Snowy</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>46</td>
      <td>2.4</td>
      <td>104.705220</td>
    </tr>
    <tr>
      <th>12</th>
      <td>832.61</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Evening</td>
      <td>Truck</td>
      <td>43</td>
      <td>2.6</td>
      <td>103.417144</td>
    </tr>
    <tr>
      <th>13</th>
      <td>213.13</td>
      <td>Rainy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Bike</td>
      <td>45</td>
      <td>0.4</td>
      <td>101.855492</td>
    </tr>
    <tr>
      <th>14</th>
      <td>182.64</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>32</td>
      <td>7.1</td>
      <td>93.051168</td>
    </tr>
    <tr>
      <th>15</th>
      <td>184.22</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Afternoon</td>
      <td>Bike</td>
      <td>11</td>
      <td>1.1</td>
      <td>77.676971</td>
    </tr>
    <tr>
      <th>16</th>
      <td>304.94</td>
      <td>Foggy</td>
      <td>Medium</td>
      <td>Night</td>
      <td>Truck</td>
      <td>13</td>
      <td>4.4</td>
      <td>75.911530</td>
    </tr>
    <tr>
      <th>17</th>
      <td>525.23</td>
      <td>Rainy</td>
      <td>Low</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>12</td>
      <td>2.0</td>
      <td>77.573367</td>
    </tr>
    <tr>
      <th>18</th>
      <td>432.51</td>
      <td>Rainy</td>
      <td>High</td>
      <td>Night</td>
      <td>Scooter</td>
      <td>16</td>
      <td>9.0</td>
      <td>86.915899</td>
    </tr>
    <tr>
      <th>19</th>
      <td>291.94</td>
      <td>Clear</td>
      <td>Medium</td>
      <td>Morning</td>
      <td>Bike</td>
      <td>38</td>
      <td>4.8</td>
      <td>92.063784</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```

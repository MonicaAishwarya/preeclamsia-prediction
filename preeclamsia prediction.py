from google.colab import files

uploaded = files.upload()  # Click "Choose Files" and select your CSV
fname = next(iter(uploaded))  # gets the uploaded file name
print("Uploaded file:", fname)

import pandas as pd
data = pd.read_csv(fname)
data.head()


import pandas as pd

# Load dataset
data = pd.read_csv("Maternal Health Risk Data Set.csv")

# First few rows
print(data.head())

# Summary
print(data.info())
print(data['RiskLevel'].value_counts())


from google.colab import drive
drive.mount('/content/drive')


from sklearn.preprocessing import LabelEncoder

# Encode RiskLevel
le = LabelEncoder()
data['RiskLevel'] = le.fit_transform(data['RiskLevel'])

print(le.classes_)  # to see mapping


from sklearn.model_selection import train_test_split

X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train XGBoost
xgb = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb.predict(X_test)

# Evaluate
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

from sklearn.model_selection import cross_val_score
import numpy as np

# Random Forest CV
rf_scores = cross_val_score(RandomForestClassifier(random_state=42),
                            X, y, cv=5, scoring='accuracy')
print("Random Forest CV Accuracy:", np.mean(rf_scores))

# XGBoost CV
xgb_scores = cross_val_score(XGBClassifier(use_label_encoder=False,
                                           eval_metric='mlogloss',
                                           random_state=42),
                             X, y, cv=5, scoring='accuracy')
print("XGBoost CV Accuracy:", np.mean(xgb_scores))



from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Randomized search
rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,   # number of random combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

rf_random.fit(X, y)

print("Best Params:", rf_random.best_params_)
print("Best CV Accuracy:", rf_random.best_score_)



from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Best tuned Random Forest (from previous step)
best_rf = rf_random.best_estimator_

# XGBoost model
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# Soft voting ensemble
ensemble = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', xgb)],
    voting='soft'
)

# Cross-validation to check performance
cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')

print("Ensemble CV Accuracy:", np.mean(cv_scores))


import matplotlib.pyplot as plt

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(8,6))
plt.title("Feature Importances - Random Forest")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.show()




import shap

# Train XGB on full data
xgb.fit(X, y)

# Explain predictions
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X)

# Global importance (bar plot)
shap.summary_plot(shap_values, X, plot_type="bar")

# Detailed beeswarm plot
shap.summary_plot(shap_values, X)





import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
data = pd.read_csv("Maternal Health Risk Data Set.csv")

# Encode labels
le = LabelEncoder()
data['RiskLevel'] = le.fit_transform(data['RiskLevel'])

X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (you can replace with tuned RandomForest or ensemble)
best_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="log2",
    random_state=42
)
best_rf.fit(X_train, y_train)

# Prediction function

label_map = {0: "High Risk", 1: "Low Risk", 2: "Mid Risk"}

def predict(age, sbp, dbp, bs, temp, hr):
    input_data = pd.DataFrame([[age, sbp, dbp, bs, temp, hr]],
                              columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    pred = best_rf.predict(input_data)[0]
    return label_map[pred]

# Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("## Maternal Health Risk Predictor")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age")
            sbp = gr.Number(label="SystolicBP")
            dbp = gr.Number(label="DiastolicBP")
            bs = gr.Number(label="Blood Sugar")
            temp = gr.Number(label="Body Temperature")
            hr = gr.Number(label="Heart Rate")
            btn = gr.Button("Predict Risk Level ")

        with gr.Column():
            output = gr.Textbox(label="Prediction Result")

    btn.click(predict, inputs=[age, sbp, dbp, bs, temp, hr], outputs=output)

# Launch the app
demo.launch()

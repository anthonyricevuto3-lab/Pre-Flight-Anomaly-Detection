
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

#load preflight data
df = pd.read_csv('airplane_data.csv')
X = df[['rpm', 'temperature', 'pressure', 'voltage']]

#train unsupervised model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

#save model
joblib.dump(model, 'isolation_forest_model.pkl')
print("Model trained and saved.")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print("Starting the Intelligence System...")

# Load the data
train_df = pd.read_csv('train_complaints.csv')
test_df = pd.read_csv('test_complaints.csv')

# Preprocess: Handle missing text
train_df['complaint_text'] = train_df['complaint_text'].fillna('missing')
test_df['complaint_text'] = test_df['complaint_text'].fillna('missing')

# Encode Categories
le_primary = LabelEncoder()
le_secondary = LabelEncoder()

y_primary = le_primary.fit_transform(train_df['primary_category'])
y_secondary = le_secondary.fit_transform(train_df['secondary_category'])
y_severity = train_df['severity']

# Transform Text to Numbers (TF-IDF)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tfidf.fit_transform(train_df['complaint_text'])
X_test = tfidf.transform(test_df['complaint_text'])

# Train the Models
print("Training models (this might take a minute)...")
model_p = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_primary)
model_s = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_secondary)
model_v = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_severity)

# Generate Predictions
print("Generating predictions...")
pred_p = le_primary.inverse_transform(model_p.predict(X_test))
pred_s = le_secondary.inverse_transform(model_s.predict(X_test))
pred_v = np.clip(np.round(model_v.predict(X_test)), 1, 5).astype(int)

# Save the results
submission = pd.DataFrame({
    'complaint_id': test_df['complaint_id'],
    'primary_category': pred_p,
    'secondary_category': pred_s,
    'severity': pred_v
})

submission.to_csv('submission.csv', index=False)
print("Done! File 'submission.csv' is ready for management.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np

# Load the dataset
path = r"xy.csv"
df = pd.read_csv(path)

# Missing values fill
string_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(include=['float']).columns

df[string_columns] = df[string_columns].fill('Unknown')
df[numeric_columns] = df[numeric_columns].fill(df[numeric_columns].mean())

# Encode string columns using ordinal encoding
label_encoders = {}
for column in string_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

m_c = df.isnull().sum()
empty = m_c[m_c == len(df)].index
df = df.drop(empty, axis=1)

# Data Split
y = df['Converted']
X = df.drop(['Converted', 'Lead Number'], axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


confusion_matrix = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
print("\nOVERALL ACCURACY SCORE: ", acc)

sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True,
            fmt='.2%', cmap='Blues').set(title='Confusion matrix: Logistic Regression')

r = classification_report(y_test, y_pred)
print("\n CLASSIFICATION REPORT: \n", r)

# Load and preprocess new lead data
path2 = r"xyz.csv"
new_leads = pd.read_csv(path2)

new_leads[string_columns] = new_leads[string_columns].fill('Unknown')
new_leads[numeric_columns] = new_leads[numeric_columns].fill(df[numeric_columns].mean())
for col in string_columns:
    new_leads[column] = label_ecoders[column].transform(new_leads[column])

new_leads = new_leads.(empty, axis=1)
new_leads = new_leads.drop(['Lead Number'], axis=1)  # Exclude 'Lead Number'


new_leads_pred = model.(new_leads)
s = []
    s.append(i)
    
new_leads_predictions_df = pd.DataFrame({'Converted': new_leads_pred})

#new_leads_predictions_df.to_csv('final.csv')

# Print the predictions or use as needed
print(new_leads_predictions_df)

# Filter leads with values
print("\n Zero values:")
zero_val = new_leads_predictions_df[new_leads_predictions_df['Converted'] == 0]
print("\n",zero_val)
print("\n One values:")
one_val = new_leads_predictions_df[new_leads_predictions_df['Converted'] == 1]
print("\n",one_val)


s = new_predictions_df[new_leads_predictions_df['Converted']]
x = []
for i in new_leads:
    if new_leads[i] == s[i]:
        x.append("True")
     
        
print(x)

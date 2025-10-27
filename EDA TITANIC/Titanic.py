
# Titanic EDA - train.csv

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Basic Info
print("=== TRAIN INFO ===")
print(train.info())
print("\n=== TEST INFO ===")
print(test.info())

# Check missing values
print("\nMissing values in Train:\n", train.isnull().sum())
print("\nMissing values in Test:\n", test.isnull().sum())

# Statistical Summary
print("\n=== TRAIN DESCRIBE ===")
print(train.describe())

# Categorical value counts
print("\n=== VALUE COUNTS ===")
for col in ['Sex', 'Pclass', 'Embarked']:
    print(f"\n{col} value counts:\n", train[col].value_counts())

# Visualizations
sns.set(style="whitegrid")

# 1. Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=train, palette='viridis')
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=train, palette='cool')
plt.title("Survival by Gender")
plt.show()

# 3. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=train, palette='mako')
plt.title("Survival by Passenger Class")
plt.show()

# 4. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(train['Age'].dropna(), kde=True, color='purple')
plt.title("Age Distribution")
plt.show()

# 5. Boxplot - Age vs Pclass
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Age', data=train, palette='coolwarm')
plt.title("Age vs Passenger Class")
plt.show()

# 6. Heatmap - Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 7. Pairplot (numeric features)
sns.pairplot(train[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived', palette='husl')
plt.show()

# Observations
print("""
Observations:
1. Females had higher survival rates compared to males.
2. Passengers in higher classes (Pclass 1) had better survival chances.
3. Younger passengers tended to survive more.
4. Fare is positively correlated with survival probability.
5. Missing values exist in 'Age' and 'Cabin' columns — these need handling before modeling.
""")

# Summary of Findings
print("""
Summary:
The Titanic dataset shows clear relationships between survival and socio-economic/demographic factors.
- Class and gender are strong predictors of survival.
- Fare and age also influence survival likelihood.
EDA helps identify these patterns before any predictive modeling.
""")

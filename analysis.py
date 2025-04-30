import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set(style="whitegrid")

# Load data
df = pd.read_csv("ptb-xl/ptbxl_database.csv")

# Convert scp_codes from string to dict
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

# Extract first label
df['first_label'] = df['scp_codes'].apply(lambda x: list(x.keys())[0] if x else None)

# Calculate BMI
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# ECG Classification Distribution
plt.figure(figsize=(12, 6))
df['first_label'].value_counts().plot(kind='bar', color='red')
plt.title("ECG Classification Distribution")
plt.xlabel("First Label")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Average BMI by Classification
plt.figure(figsize=(12, 6))
df.groupby('first_label')['bmi'].mean().sort_values(ascending=False).plot(kind='bar', color='orange')
plt.title('Average BMI by Classification')
plt.xlabel('Classification')
plt.ylabel('Average BMI')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), bins=30, kde=True, color='blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Sex Distribution
plt.figure(figsize=(6, 6))
sex_counts = df['sex'].value_counts().sort_index()
plt.pie(sex_counts, labels=['Male', 'Female'], autopct='%1.1f%%', colors=['skyblue', 'pink'], startangle=90)
plt.title("Sex Distribution")
plt.show()

# BMI Distribution by Sex
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='bmi', data=df)
plt.title("BMI Distribution by Sex")
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()

# Correlation Heatmap of Numeric Features
plt.figure(figsize=(8, 6))
numeric_cols = ['age', 'height', 'weight', 'bmi']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Pacemaker Presence
plt.figure(figsize=(6, 4))
df['pacemaker'].value_counts().plot(kind='bar', color='green')
plt.title("Pacemaker Presence")
plt.xticks(ticks=[0, 1], labels=['No Pacemaker', 'Pacemaker'], rotation=0)
plt.ylabel("Count")
plt.show()

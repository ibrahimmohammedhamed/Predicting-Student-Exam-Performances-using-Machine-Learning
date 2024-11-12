#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('StudentPerformanceFactors.csv')


# In[3]:


df.head()


# In[4]:


missing_values = df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df[df['Distance_from_Home'].isnull()]


# In[8]:


df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0], inplace=True)


# In[9]:


df['Distance_from_Home'] = pd.to_numeric(df['Distance_from_Home'], errors='coerce')


# In[10]:


df.info()


# In[12]:


df.drop(columns=['Distance_from_Home'], inplace=True)



# In[13]:


print(df.info())


# In[14]:


print(df.describe())


# In[15]:


categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"{col} value counts:\n{df[col].value_counts()}\n")


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score')
plt.title('Relationship Between Hours Studied and Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.savefig('hours_studied_vs_exam_score.png')
plt.show()


# In[32]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Attendance', y='Exam_Score')
plt.title('Exam Scores Across Different Attendance Levels')
plt.xlabel('Attendance Level')
plt.ylabel('Exam Score')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.savefig('attendance_vs_exam_score_rotated.png')
plt.show()


# ###### plt.figure(figsize=(8, 6))
# sns.barplot(data=df, x='Family_Income', y='Exam_Score', ci=None)
# plt.title('Impact of Family Income on Exam Performance')
# plt.xlabel('Family Income Level')
# plt.ylabel('Average Exam Score')
# plt.savefig('family_income_vs_exam_score.png')
# plt.show()

# In[34]:


plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Motivation_Level', y='Exam_Score', ci=None)
plt.title('Influence of Motivation on Exam Performance')
plt.xlabel('Motivation Level')
plt.ylabel('Average Exam Score')
plt.savefig('motivation_level_vs_exam_score.png')
plt.show()


# In[35]:


plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='Parental_Involvement', y='Exam_Score', ci=None)
plt.title('Influence of Parental Involvement on Exam Score')
plt.xlabel('Parental Involvement')
plt.ylabel('Average Exam Score')
plt.savefig('parental_involvement_vs_exam_score.png')
plt.show()


# In[21]:


import seaborn as sns

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()


# In[22]:


from sklearn.linear_model import LinearRegression
import numpy as np

X = df[['Hours_Studied']]
y = df['Exam_Score']

model = LinearRegression()
model.fit(X, y)


# In[23]:


y_pred = model.predict(X)


# In[24]:


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Exam Scores')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.title('Linear Regression: Exam Score vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid()
plt.show()


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score



# In[37]:


y_pred = model.predict(X)


# In[38]:


mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)


# In[39]:


print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")


# In[40]:


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Exam Scores')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.title('Linear Regression: Exam Score vs. Hours Studied')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering 
# 
# Feature Engineering is the process of transforming raw data into meaningful features that can be used as input for advanceced visualisations or machine learning algorithms.
# 
# It involves selecting, creating, and transforming features to hopefully enhance the dataset.
# 
# Poorly designed features can lead to a disruptive dataset. 
# 

# ## Types of Feature Engineering
# 
# * **Handling Missing Values**
# 
#     Filling missing values with appropriate strategies, e.g., mean, median, or constant values.
# 
# * **Encoding Categorical Variables**
# 
#     Converting categorical data into numeric form, such as one-hot encoding or label encoding. Only needed if you are building a model
# 
# * **Binning Numeric Variables**
# 
#     Grouping continuous data into bins or categories to simplify the representation.
# 
# * **Feature Scaling**
# 
#     Scaling features to bring them to a similar range, e.g., Min-Max scaling or Standard scaling.
# 
# * **Creating New Features**
# 
#     Generating new features by combining or transforming existing ones.
# 
# * **Handling Outliers**
# 
#     Managing extreme values that can affect model performance.
# 
# * **Feature Joining**
# 
#     Creating new features by combining multiple existing features.

# ## Imports and Dataset

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df = pd.read_excel(r"C:\Users\mselala\Downloads\insurance_claims_raw.xlsx")


# In[7]:


df.head()


# ### Missing Values

# In[8]:


null_counts = df.isnull().sum()
null_counts


# In[9]:


df_new = df.drop("_c39", axis = 1)


# In[10]:


df_new.head()


# ### Binding Numeric Data

# In[11]:


df_new.describe()


# In[12]:


# Choose the column for the histogram
column_name = 'age'

# Plot the histogram
plt.hist(df[column_name], bins=3, edgecolor='black')

# Add labels and title
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.title(f'Histogram of {column_name}')

# Display the histogram
plt.show()


# In[31]:


bin_edges = [0, 30, 55, 100]  # Define the bin edges
bin_labels = ['Young Adult', 'Middle Aged', 'Elderly']  # Corresponding labels for each bin

# Create a new column based on the bin labels
df_new['ages_category'] = pd.cut(df_new['age'], bins=bin_edges, labels=bin_labels)


# In[13]:


df_new.head()


# In[38]:


bin_edges_customer = [0, 25, 150, 500]  # Define the bin edges
bin_labels_customer = ['New Client', 'Established Client', 'Long-Term Client']  # Corresponding labels for each bin

# Create a new column based on the bin labels
df_new['customer_category'] = pd.cut(df_new['months_as_customer'], bins=bin_edges_customer, labels=bin_labels_customer)


# In[39]:


df_new.head()


# ## Creating New Features

# In[40]:


df_new["Contract Years"] = df_new["months_as_customer"]/12


# In[14]:


df_new.head()


# ## Feature Joining

# In[15]:


df_new['total_premiums_paid'] = (df_new['policy_annual_premium']/12) * df_new['months_as_customer']


# In[16]:


df_new.head()


# In[17]:


df_new['net_value_of_customer'] = df_new['total_premiums_paid'] - df_new['total_claim_amount']


# In[43]:


df_new.head(10)


# ## Saving the csv for late

# In[51]:


df_new.to_csv('Advanced Features Claims Data.csv')


# ## Go wild
# 
# Go out a see what other features you can create that will be useful for our visualisations

# In[19]:


# Assuming 'df' is your DataFrame

# Impute missing values in numerical columns with their median
numerical_cols = ['age', 'policy_deductable', 'policy_annual_premium', 'capital-gains', 'total_claim_amount', 'injury_claim', 'property_claim']
for col in numerical_cols:
    median_val = df_new[col].median()
    df_new[col].fillna(median_val, inplace=True)



# In[23]:


# Assuming 'df' is your DataFrame containing the 'insurance_hobbies' column
most_common_hobbies = df_new['insured_hobbies'].value_counts().idxmax()
print("Most Common Hobby:", most_common_hobbies)


# In[24]:


hobbies_counts = df_new['insured_hobbies'].value_counts()
print(hobbies_counts)


# In[26]:


df_new['insured_hobbies'].fillna('reading', inplace=True)


# In[45]:


null_counts = df_new.isnull().sum()
null_counts


# In[29]:


authorities_contacted = df_new['authorities_contacted'].value_counts()
print(authorities_contacted)


# In[30]:


df_new['authorities_contacted'].fillna('Fire', inplace=True)


# In[34]:


insured_education_level = df_new['insured_education_level'].value_counts()
print(insured_education_level)


# In[35]:


df_new['insured_education_level'].fillna('College', inplace=True)


# In[37]:


incident_state = df_new['incident_state'].value_counts()
print(incident_state )


# In[38]:


df_new['incident_state'].fillna('NY', inplace=True)


# In[40]:


# Assuming 'df' is your DataFrame and 'total_premiums_paid' is the column
average_premiums_paid = df_new['total_premiums_paid'].mean()
print("Average Total Premiums Paid:", average_premiums_paid)


# In[41]:


# Replace NaN values in 'total_premiums_paid' column with the average
df_new['total_premiums_paid'].fillna(average_premiums_paid, inplace=True)


# In[44]:


net_value_of_customer_m = df_new['net_value_of_customer'].mean()
print("net_value_of_customer_m:", net_value_of_customer_m)
df_new['net_value_of_customer'].fillna(net_value_of_customer_m, inplace=True)


# In[46]:


df_new.to_csv('Advanced Features Claims Data.csv')


# In[ ]:





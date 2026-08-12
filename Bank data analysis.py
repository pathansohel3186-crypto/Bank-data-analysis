# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Read csv file 
data=pd.read_csv("banking data.csv")
data

# %%
# Convert to column integer data type
data["age"] = data["age"].astype("Int64")
data["credit_score"] = data["credit_score"].astype("Int64")

data

# %%
#*********** NUMPY ***********#      

# Convert columns into NumPy arrays
customer_id = data["customer_id"].to_numpy()
customer_name = data["customer_name"].to_numpy()
age = data["age"].to_numpy()
city = data["city"].to_numpy()
email = data["email"].to_numpy()
account_type = data["account_type"].to_numpy()
account_balance = data["account_balance"].to_numpy()
credit_score = data["credit_score"].to_numpy()
 


# %%
# null value check 

print("age null values:",np.isnan(age).sum())
print("account balance null values:",np.isnan(account_balance).sum())
print("credit score null values:",np.isnan(credit_score).sum())


# %%
# check unique values

print("unique id age:",np.unique(age).size)
print("unique id account_balance:",np.unique(account_balance).size)
print("unique id credite score:",np.unique(credit_score).size)

# %%
# credit score statistical analysis
print("Mean:", np.nanmean(credit_score))
print("Median:", np.nanmedian(credit_score))
print("Standard Deviation:", np.nanstd(credit_score))
print("Minimum:", np.nanmin(credit_score))
print("Maximum:", np.nanmax(credit_score))

# %%
# check average age
print("avg age:",np.mean(age[age<30]))

# check average account balance
print("avg account balance:",np.mean(account_balance[account_balance<25000]))

# check average credit score
print("avg credit score:",np.mean(credit_score[credit_score<550]))

# %%
# high-value customers data filter
high_value = (account_balance > 10000) & (credit_score >= 750) 

data[high_value]

# %%
# Risk customers data filter
risk = (credit_score < 600) & (account_balance> 10000)
data[risk]

# %%
# Identify the low balance customer
balance = account_balance < 10000 
data[balance]

# %%
# identify rows with null values
valid = ~np.isnan(account_balance) & ~np.isnan(credit_score)

clean_balance = account_balance[valid]
clean_credit = credit_score[valid]

np.corrcoef(clean_balance, clean_credit)

# %%
# higest 5 account balance
valid = ~np.isnan(account_balance)

valid_balance = account_balance[valid]

top_5 = np.argsort(valid_balance)[-5:][::-1]

valid_balance[top_5]

# %%
# Classifies data based on a condition
np.where(age>30,"senior","young")[:10]

# %%
#*********** PANDAS ***********# 

# Display first 5 rows
display(data.head())

# Display last 5 rows
display(data.tail())

# Display how many row & column
display(data.shape)

# %%
# Display all column name
display(data.columns)

# %%
# Display data type, non null count & column
display(data.info())

# %%
# to check numerical summury
display(data.describe())

# check data type
display(data.dtypes)

# %%
# to check every column and show null values
display(data.isnull().sum())

# remove all null value rows 
display(data.dropna())

# replace null value
display(data.fillna(0))


# %%
# Duplicate record count 
display(data.duplicated().sum())

# remove duplicate record row 
display(data.drop_duplicates())

# %%
# Show single columns
display(data["age"])

# show multiple column 
display(data[["customer_name", "age", "city"]])

# %%
# Filter the age  
display(data[data["age"] > 25])

# filter two column together 
display(data[(data["age"] > 25) & (data["city"] == "Pune")])

# %%
# it display data based on condition 
display(data.loc[data["age"] > 25])

# selects row or columns using their index
display(data.iloc[0:10])

# %%
# arrange the column data in order.
display(data.sort_values("age"))

# arrange data from largest to smallest.
display(data.sort_values("age",ascending=False))


# %%
# check unique value and show
display(data["city"].unique())

# count total no of unique value
display(data["city"].nunique())

# %%
# how many time each value is repeated
display(data["city"].value_counts())

# %%
# calculate average value
display(data.groupby("city")["age"].mean())
display(data.groupby("city")["age"].max())
display(data.groupby("city")["age"].min())
display(data.groupby("city")["age"].count())


# %%
# City wise acoont balance analysis.
display(data.groupby("city")["account_balance"].agg(["sum", "mean", "min", "max"]))


# %%
# Create new column
data["age_double"] = data["age"] * 2
data

# %%
# Rename column name  
data.rename(columns={"age_double": "age_group"}, inplace=True)
data


# %%
# permanently column remove 
data.drop(columns=["age_group"], inplace=True)
data

# %%
# convert to uppercase 
data["customer_name"] =data["customer_name"].str.upper()
data

# %%
# covert to lowercase
data["customer_name"]=data["customer_name"].str.lower()
data


# %%
# Pivot Table - group data and calculate
pd.pivot_table(data,
    values="credit_score",
    index="city",
    aggfunc="mean")


# %%
# Correlation check the reletionship between column 
data[["age","credit_score"]].corr()

# %%
# Save final data
data.to_csv("final_data.csv", index=False)
data

# %%
#*********** MATPLOTLIB ***********# 

# Line chart perfrom selected column data graphically visualizes  

x = data["age"]
y = data["account_balance"]

plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title("Age vs Account Balance")
plt.xlabel("Age")
plt.ylabel("Account Balance")
plt.show()

# %%
# Bar chart to compare data across different categories
city_balance = data.groupby("city")["account_balance"].mean()

plt.figure(figsize=(14, 5))
plt.bar(city_balance.index, city_balance.values)
plt.title("Average Account Balance by City")
plt.xlabel("City")
plt.ylabel("Average Account Balance")
plt.show()


# %%
# Scatter plot to show reletion between two variable 

x = data["age"]
y = data["account_balance"]

plt.figure(figsize=(8, 5))
plt.scatter(x, y)
plt.title("Age vs Account Balance")
plt.xlabel("Age")
plt.ylabel("Account Balance")
plt.show()

# %%
# Histogram to show the distribution of numerical data

plt.figure(figsize=(8, 5))
plt.hist(data["account_balance"], bins=10)
plt.title("Account Balance Distribution")
plt.xlabel("Account Balance")
plt.ylabel("Frequency")
plt.show()

# %%
# Box Plot to show the data distribution and identify outliers

plt.figure(figsize=(8, 5))
plt.boxplot(data["account_balance"].dropna())
plt.title("Account Balance Box Plot")
plt.ylabel("Account Balance")
plt.show()


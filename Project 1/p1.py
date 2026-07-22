import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset.csv")
print(df.head())
print(df.info())

print(df.isnull().sum())
print(df.describe())


df['TotalPrice'].hist()
plt.title('Distribution of Total Price')
plt.xlabel('Total Price')
plt.ylabel('Frequency')
plt.show()

df['PaymentMethod'].value_counts().plot(kind='bar')
plt.title('Payment Method Distribution')
plt.xlabel('Payment Method')
plt.ylabel('Count')
plt.show()


corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
plt.show()

sns.boxplot(x=df['TotalPrice'])
plt.title('Boxplot of Total Price')
plt.show()

df.groupby('Product')['TotalPrice'].sum().plot(kind='bar')
plt.title('Total Sales by Product')
plt.ylabel('Revenue')
plt.show()


df['Date'] = pd.to_datetime(df['Date'])
df['CouponCode'] = df['CouponCode'].fillna('NoCoupon')

num_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice']

#using iqr becs skewed dist and extreme values
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

#new features/columns
df['CalculatedTotal'] = df['Quantity'] * df['UnitPrice']
df['UsedCoupon'] = df['CouponCode'].apply(lambda x: 0 if x == 'NoCoupon' else 1)
df['AvgPricePerItem'] = df['TotalPrice'] / df['ItemsInCart']
df['OrderSize'] = pd.cut(df['Quantity'],bins=[0,2,5,10,100],labels=['Small','Medium','Large','Bulk'])
df.to_csv("cleaned_dataset.csv", index=False)


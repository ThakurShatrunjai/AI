import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure nltk data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load the datasets
apps_df = pd.read_csv('googleplaystore.csv')
reviews_df = pd.read_csv('googleplaystore_user_reviews.csv')

# Display the first few rows of the datasets
print("Apps DataFrame:")
print(apps_df.head())
print("\nReviews DataFrame:")
print(reviews_df.head())

# Clean the datasets
# Remove duplicates
apps_df.drop_duplicates(subset='App', keep='first', inplace=True)
reviews_df.drop_duplicates(subset=['package_name', 'review'], keep='first', inplace=True)

# Handle missing values in apps_df
apps_df.dropna(inplace=True)

# Convert data types if necessary
apps_df['Reviews'] = apps_df['Reviews'].astype(int)
apps_df['Installs'] = apps_df['Installs'].str.replace('+', '').str.replace(',', '').astype(int)

# Remove unwanted characters from the Size column and convert it to numeric
apps_df['Size'] = apps_df['Size'].replace('Varies with device', pd.NA)
apps_df['Size'] = apps_df['Size'].str.replace('M', 'e6').str.replace('k', 'e3')
apps_df['Size'] = pd.to_numeric(apps_df['Size'], errors='coerce')

# Convert 'Price' to numeric
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)

# Analysis
# Distribution of app categories
plt.figure(figsize=(15, 8))
sns.countplot(y='Category', data=apps_df, order=apps_df['Category'].value_counts().index)
plt.title('Distribution of App Categories')
plt.show()

# Average rating by category
plt.figure(figsize=(15, 8))
category_mean_rating = apps_df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=category_mean_rating, y=category_mean_rating.index)
plt.title('Average Rating by Category')
plt.show()

# Most common words in reviews (excluding stopwords)
stop_words = set(stopwords.words('english'))
reviews_df['review'] = reviews_df['review'].astype(str)
reviews_df['tokens'] = reviews_df['review'].apply(word_tokenize)
reviews_df['filtered_tokens'] = reviews_df['tokens'].apply(lambda tokens: [word for word in tokens if word.isalpha() and word.lower() not in stop_words])
all_words = [word for tokens in reviews_df['filtered_tokens'] for word in tokens]
word_freq = pd.Series(all_words).value_counts().head(20)
plt.figure(figsize=(15, 8))
sns.barplot(x=word_freq.values, y=word_freq.index)
plt.title('Most Common Words in Reviews')
plt.show()

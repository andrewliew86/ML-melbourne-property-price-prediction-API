import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Set up visualization style
sns.set_theme(style="whitegrid")

# Load in cleaned property dataset 
data = pd.read_csv("melbourne-realestate-clean.csv")

# Get summary statistics for numerical columns
summary_stats = data.describe()
print(summary_stats)

# Visualize missing data as a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing values heatmap')
plt.savefig('images/missing-values.png')


# Visualize distribution of house prices
plt.figure(figsize=(10, 6))
sns.histplot(data['price_clean'].dropna(), kde=True, bins=30, color='blue')
plt.title('Distribution of house prices')
plt.xlabel('Price (AUD)')
plt.ylabel('Frequency')
plt.savefig('images/house-price-distribution.png')

# Visualize distribution of year built
plt.figure(figsize=(10, 6))
sns.histplot(data['year_built_clean'].dropna(), kde=True, bins=20, color='blue')
plt.title('Distribution of year built')
plt.xlabel('Year built')
plt.ylabel('Frequency')
plt.savefig('images/year-built-distribution.png')


# Visualize the distribution of building types
plt.figure(figsize=(10, 6))
sns.countplot(y='building_type_clean', data=data, order=data['building_type_clean'].value_counts().index, palette='Set2')
plt.title('Building type distribution')
plt.xlabel('Count')
plt.ylabel('Building type')
plt.savefig('images/building-types-chart.png')


# Pairplot for selected numerical features to see relationships
selected_features = ['price_clean', 'year_built_clean', 'room_clean', 'shower_clean', 'car_clean', 'size_clean']
sns.pairplot(data[selected_features].dropna())
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.savefig('images/pairplot-numerical-data.png')


## Create bigram word cloud to determine frequently used bigrams in description of houses
# Combine all descriptions into one text
text = " ".join(data['description_clean'].dropna())

# Function to generate and display word clouds
def generate_wordcloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig('images/word-cloud.png')


# Bigram Word Cloud
vectorizer_bi = CountVectorizer(ngram_range=(2, 2), stop_words='english')
X_bi = vectorizer_bi.fit_transform(data['description_clean'].dropna())
word_freq_bi = dict(zip(vectorizer_bi.get_feature_names_out(), X_bi.toarray().sum(axis=0)))
generate_wordcloud(word_freq_bi, "Bigram word cloud")

print("EDA analysis complete!")
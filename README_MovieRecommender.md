
# 🎬 Movie Recommendation System using Content-Based Filtering

This project recommends similar movies based on the user's selected movie by analyzing metadata such as genres, cast, crew, keywords, and overview. It is built using Python and core Machine Learning and NLP techniques — ideal for placement and interview demonstrations.

---

## ✅ Problem Statement

Recommend top 5 similar movies using metadata like:
- Genres
- Keywords
- Cast & Crew
- Overview

---

## 📂 Dataset

- **TMDB 5000 Movies Dataset**
- **TMDB 5000 Credits Dataset**

---

## 🚀 What This Notebook Covers

- 🔧 Data Cleaning and Merging
- 🏗️ Feature Engineering (genres, cast, crew extraction)
- ✂️ Text Preprocessing (lowercase, remove spaces, stemming)
- 🧠 Vectorization using CountVectorizer
- 📐 Cosine Similarity for measuring movie similarity
- 🎯 Recommendation function to fetch top 5 similar movies

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK for Stemming
- Jupyter Notebook

---

## 🧠 Core ML/NLP Concepts

### ✅ Stemming
Applied using NLTK’s PorterStemmer to reduce similar words to their root form (e.g., "acting", "actor" → "act").

### ✅ Vectorization
Used CountVectorizer to convert text into numerical form so that cosine similarity can be applied.

### ✅ Cosine Similarity
Calculated the angle between vectors of different movies. Smaller angles mean higher similarity.

---

## 📌 Example Recommendation Output

```python
recommend("Avatar")
```

📽️ **Top 5 movies similar to Avatar:**
- John Carter  
- Guardians of the Galaxy  
- The Avengers  
- Star Trek  
- The Hobbit

---

## 📁 Project Workflow

1. Load and clean the TMDB datasets
2. Extract genres, keywords, cast, and crew
3. Preprocess text (split, lowercase, remove spaces)
4. Create a combined `tags` column for each movie
5. Apply stemming on tags
6. Vectorize using CountVectorizer (max 5000 features)
7. Compute similarity matrix using Cosine Similarity
8. Define and test the recommendation function

---

## 🧾 Sample Code Snippet

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(final_movies['tags']).toarray()

# Similarity Calculation
similarity = cosine_similarity(vectors)

# Recommendation Function
def recommend(movie):
    movie = movie.lower()
    if movie not in final_movies['title'].str.lower().values:
        print("Movie not found.")
        return
    index = final_movies[final_movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(final_movies.iloc[i[0]].title)
```
Similarity Score :
How does it decide which item is most similar to the item user likes? Here come the similarity scores.

It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.

How Cosine Similarity works?
Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.



---

## 👩‍💻 Author

**Vridhi Bansal**  
Data Science & Machine Learning Enthusiast  
📧 Email: vridhibansal@example.com  
🌐 GitHub: [github.com/vridhibansal(https://github.com/vridhibansal)

---

⭐ Star this repo if you liked it and want to see more ML projects!

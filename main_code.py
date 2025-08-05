1. Install Required Packages
!pip install pandas scikit-learn --quiet

2. Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

3. Load Dataset 
df = pd.read_csv("indian_courses_dataset.csv")

 Preview dataset
print(df.head())

4. Feature Engineering: Combine Course Title, Subject, and Instructors
df['content'] = df['Course Title'] + " " + df['Course Subject'] + " " + df['Instructors']

5. TF-IDF Vectorization on combined content
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])

6. Compute Cosine Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

7. Build a reverse map of indices and course titles for lookup
indices = pd.Series(df.index, index=df['Course Title']).drop_duplicates()

8. Recommendation function
def recommend(course_title, cosine_sim=cosine_sim):
    if course_title not in indices:
        return f"Course '{course_title}' not found in dataset."
    idx = indices[course_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort courses by similarity score descending and exclude the input course itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    course_indices = [i[0] for i in sim_scores]
    return df[['Course Title', 'Institution', 'Course Subject']].iloc[course_indices]

 9. Test recommendation
print("Recommendations for 'Machine Learning Basics':")
print(recommend("Machine Learning Basics"))

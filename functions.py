import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('./Financebench.csv')

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

def find_closest_match(question):
    """
    Finds the closest match for a question in the dataset using TF-IDF and cosine similarity.
    """
    query_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    max_index = similarity_scores.argmax()
    return max_index

def get_answer(question):
    """
    Returns the closest question's answer and evidence from the dataset.
    """
    index = find_closest_match(question)
    return df.iloc[index]['answer'], df.iloc[index]['evidence_text']
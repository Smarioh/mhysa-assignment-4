from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import os

app = Flask(__name__)

# Load dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Preprocess with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(documents)

# Apply LSA (Truncated SVD)
lsa = TruncatedSVD(n_components=100, random_state=42)
X_lsa = lsa.fit_transform(X)

def calculate_cosine_similarity(query_lsa, X_lsa):
    """
    Custom cosine similarity function without using sklearn's cosine_similarity.
    Handles cases where vectors have a zero norm to avoid division by zero.
    """
    dot_product = np.dot(X_lsa, query_lsa.T)
    norm_docs = np.linalg.norm(X_lsa, axis=1)
    norm_query = np.linalg.norm(query_lsa)

    # To avoid division by zero, return 0 similarity if the norm is 0
    norm_docs[norm_docs == 0] = 1e-10  # Small value to prevent division by zero
    norm_query = norm_query if norm_query != 0 else 1e-10

    return dot_product.flatten() / (norm_docs * norm_query)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    query_vec = vectorizer.transform([query])
    query_lsa = lsa.transform(query_vec)

    # Calculate cosine similarity using a custom function
    similarities = calculate_cosine_similarity(query_lsa, X_lsa)

    # Get the indices of the top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]

    # Retrieve the top 5 documents and their similarity scores
    top_docs = [documents[idx][:500] for idx in top_indices]
    top_similarities = similarities[top_indices]
    
    return top_docs, top_similarities.tolist(), top_indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form['query']
        print(f"Received query: {query}")  # Debugging statement

        documents, similarities, indices = search_engine(query)

        # Return documents, similarities, and indices for client-side rendering
        return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices})
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Print error to terminal for debugging
        return jsonify({'error': 'An error occurred on the server'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)

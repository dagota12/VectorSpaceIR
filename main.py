import os
import glob
import json
from nltk import PorterStemmer, download
from nltk.corpus import stopwords
from nltk.tokenize_rm_stopAndStem import word_tokenize
from collections import defaultdict, Counter
from math import sqrt, log2
import string
"""
`vector space model(vsm) ` implementation
used: 
    - cosine simmilarity for measuring similarity,
    - index for optimizing search
Note:
    -it only shows relevancy that is above 0% the assumption is
        - b/c below that the the docunment is completly irrelevant
"""
# Ensure stopwords are downloaded if not
# download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize stemmer
stemmer = PorterStemmer()

def load_corpus(directory):
    text_files_content = []
    text_files = glob.glob(f'{os.path.dirname(os.path.abspath(__file__))}/{directory}/*.txt')
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().replace('\n', " ")
            text_files_content.append((os.path.basename(file_path), content))
    return text_files_content

def tokenize_rm_stopAndStem(text):
    special_chars = string.punctuation + string.whitespace
    tokens = word_tokenize(text)
    tokens = [word.lower().strip(special_chars) for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def create_index_and_tfidf_vectors(docs):
    tokenized_docs = [(filename, tokenize_rm_stopAndStem(content)) for filename, content in docs]
    terms = set()
    for _, doc in tokenized_docs:
        terms |= set(doc)

    TF = defaultdict(lambda: defaultdict(int))
    DF = defaultdict(int)
    for i, (_, doc) in enumerate(tokenized_docs):
        inc = defaultdict(int)
        for word in doc:
            if inc[word] == 0:
                DF[word] += 1
                inc[word] = 1
            TF[i][word] += 1

    term_to_docs = defaultdict(set)
    for filename, tokens in tokenized_docs:
        for token in tokens:
            term_to_docs[token].add(filename)

    index_data = {term: list(doc_set) for term, doc_set in term_to_docs.items()}
    index_file_path = os.path.dirname(os.path.abspath(__file__)) + '/index/index.json'
    os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
    with open(index_file_path, 'w', encoding='utf-8') as index_file:
        json.dump(index_data, index_file)

    N = len(docs)
    doc_vectors = []
    for i, (_, doc) in enumerate(tokenized_docs):
        vector = []
        for term in terms:
            tf = TF[i][term]
            idf = log2(N / DF[term]) if DF[term] != 0 else 0
            vector.append(tf * idf)
        doc_vectors.append(vector)

    return terms, doc_vectors, DF, N, tokenized_docs

def load_index():
    index_file_path = os.path.dirname(os.path.abspath(__file__)) + '/index/index.json'
    with open(index_file_path, 'r', encoding='utf-8') as index_file:
        index_data = json.load(index_file)
    return index_data

def calculate_query_vector(query, terms, DF, N):
    preprocessed_query = tokenize_rm_stopAndStem(query)
    query_TF = Counter(preprocessed_query)
    query_vector = []
    for term in terms:
        tf = query_TF[term]
        idf = log2(N / DF[term]) if DF[term] != 0 else 0
        query_vector.append(tf * idf)
    return query_vector

def cosine_similarity(u, v):
    dotprod = sum(a * b for a, b in zip(u, v))
    u_mag = sqrt(sum(a ** 2 for a in u))
    v_mag = sqrt(sum(a ** 2 for a in v))
    return (dotprod / (u_mag * v_mag)) if u_mag != 0 and v_mag != 0 else 0

def search_and_display(query, terms, doc_vectors, DF, N, docs, index_data):
    preprocessed_query = tokenize_rm_stopAndStem(query)
    relevant_docs = set()
    for term in preprocessed_query:
        if term in index_data:
            relevant_docs.update(index_data[term])

    query_vector = calculate_query_vector(query, terms, DF, N)
    similarities = []
    for i, (filename, _) in enumerate(docs):
        if filename in relevant_docs:
            similarity = cosine_similarity(doc_vectors[i], query_vector)
            similarities.append((similarity, i))
    similarities.sort(reverse=True, key=lambda x: x[0])

    print("\nRelevant documents:")
    if not similarities:
        print("sorry, the data you are searching not found")
    for rank, (similarity, i) in enumerate(similarities):
        print(f"Rank {rank + 1} (match: {similarity:.2%}): {docs[i][0]}")

def main():
    directory_path = 'corpus'
    docs = load_corpus(directory_path)
    terms, doc_vectors, DF, N, tokenized_docs = create_index_and_tfidf_vectors(docs)
    index_data = load_index()

    while True:
        print("\nMenu:")
        print("[q] -> Quit")
        print("[s] -> Search")
        choice = input("Enter your choice: ").strip().lower()
        if choice == 'q':
            break
        elif choice == 's':
            query = input("query: ").strip()
            search_and_display(query, terms, doc_vectors, DF, N, docs, index_data)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

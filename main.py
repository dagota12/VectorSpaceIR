from nltk import PorterStemmer, download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from math import sqrt, log2
import string
import os
import glob

# note for reader, Download stopwords if needed(or not downloaded yet)
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

def tokenize(text):
    special_chars = string.punctuation + string.whitespace
    tokens = word_tokenize(text)
    tokens = [word.lower().strip(special_chars) for word in tokens if word not in stop_words and word not in string.punctuation]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

def create_index_and_tfidf_vectors(docs):
    tokenized_docs = [(filename, tokenize(content)) for filename, content in docs]
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

    index_file_path = os.path.dirname(os.path.abspath(__file__)) +'/index/index.txt'
    os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
    with open(index_file_path, 'w', encoding='utf-8') as index_file:
        for term, doc_set in sorted(term_to_docs.items()):
            index_file.write(f"{term}: {', '.join(sorted(doc_set))}\n")

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

def calculate_query_vector(query, terms, DF, N):
    preprocessed_query = tokenize(query)
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

def search_and_display(query, terms, doc_vectors, DF, N, docs):
    query_vector = calculate_query_vector(query, terms, DF, N)
    similarities = []
    for i, doc_vector in enumerate(doc_vectors):
        similarity = cosine_similarity(doc_vector, query_vector)
        similarities.append((similarity, i))
    similarities.sort(reverse=True, key=lambda x: x[0])

    print("\nRelevalt documents:")
    for rank, (similarity, i) in enumerate(similarities):
        print(f"Rank {rank + 1} (match: {similarity:.2%}): {docs[i][0]}")
#main driver program to run the code
def main():
    directory_path = 'corpus'
    docs = load_corpus(directory_path)
    terms, doc_vectors, DF, N, tokenized_docs = create_index_and_tfidf_vectors(docs)

    while True:
        print("\nMenu:")
        print("[q] -> Quit")
        print("[s] -> Search")
        choice = input("Enter your choice: ").strip().lower()
        if choice == 'q':
            break
        elif choice == 's':
            query = input("query: ").strip()
            search_and_display(query, terms, doc_vectors, DF, N, docs)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

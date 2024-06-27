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
            content = file.read().replace('\n'," ")
            
            text_files_content.append((os.path.basename(file_path), content))
    
    return text_files_content
#corpus directiory path
directory_path = 'trial_corpus'

docs = load_corpus(directory_path)

# print("Documents:", docs)
query = "The Renaissance"

# tokenizing and some other function
def tokenize(text):
    #remove any special charachers
    special_chars = string.punctuation + string.whitespace
    tokens = word_tokenize(text)
    tokens = [word.lower().strip(special_chars) for word in tokens if word not in stop_words and word not in string.punctuation]
    #may be the stemming is making error on  the simmilarity measure
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

# Preprocess documents and query by tokenizing
tokenized_docs = [(filename, tokenize(content)) for filename, content in docs]
# print("Preprocessed Docs:", tokenized_docs)
preprocessed_query = tokenize(query)
print(preprocessed_query)
# Extract terms from preprocessed documents and query
terms = set()
for _, doc in tokenized_docs:
    terms |= set(doc)
terms |= set(preprocessed_query)
# print("Terms:", terms)

# Initialize term frequency (TF) and document frequency (DF)
TF = defaultdict(lambda: defaultdict(int))
DF = defaultdict(int)

# Calculate TF and DF
for i, (_, doc) in enumerate(tokenized_docs):
    inc = defaultdict(int)
    for word in doc:
        if inc[word] == 0:
            DF[word] += 1
            inc[word] = 1
        TF[i][word] += 1

# print("TF:", dict(TF))
# print("DF:", dict(DF))

# index file
term_to_docs = defaultdict(set)

# Populate the dictionary
for filename, tokens in tokenized_docs:
    for token in tokens:
        term_to_docs[token].add(filename)

# Write the index to a file
index_file_path = os.path.dirname(os.path.abspath(__file__)) + '/index/index.txt'
os.makedirs(os.path.dirname(index_file_path), exist_ok=True)
with open(index_file_path, 'w', encoding='utf-8') as index_file:
    for term, doc_set in sorted(term_to_docs.items()):
        index_file.write(f"{term}: {', '.join(sorted(doc_set))}\n")

# print(f"Index file created at {index_file_path}")
# Number of documents
N = len(docs)

# Calculate TF-IDF for documents
doc_vectors = []
for i, (_, doc) in enumerate(tokenized_docs):
    vector = []
    for term in terms:
        tf = TF[i][term]
        idf = log2(N / DF[term]) if DF[term] != 0 else 0
        vector.append(tf * idf)
    doc_vectors.append(vector)

# print("Document Vectors:", doc_vectors)

# Calculate TF-IDF for query
query_vector = []
query_TF = Counter(preprocessed_query)
for term in terms:
    tf = query_TF[term]
    idf = log2(N / DF[term]) if DF[term] != 0 else 0
    query_vector.append(tf * idf)

# print("Query Vector:", query_vector)

# Cosine similarity function
def cosine_similarity(u, v):
    dotprod = sum(a * b for a, b in zip(u, v))
    u_mag = sqrt(sum(a ** 2 for a in u))
    v_mag = sqrt(sum(a ** 2 for a in v))
    return (dotprod / (u_mag * v_mag)) if u_mag != 0 and v_mag != 0 else 0

# Calculate cosine similarity between query and each document
similarities = []
for i, doc_vector in enumerate(doc_vectors):
    similarity = cosine_similarity(doc_vector, query_vector)
    similarities.append((similarity, i))

# Sort documents by similarity in descending order
similarities.sort(reverse=True, key=lambda x: x[0])

# Display documents based on match
print("\nDocuments sorted by relevance to the query:")
for rank,(similarity, i) in enumerate(similarities):
    # print(idx,similarity,i)
    print(f"Rank {rank+1} (match: {similarity:.2%}): {docs[i][0]}")  # Display file name
def main():
    query = input()
    """
    ## To do!
    do TF(),DF(),IDF(),simmilariy()
    and finally displaying the result
    while not quit:
        display menu
            [q] ->[quit]
            [s] ->[search]
            otherwise ask again
    -  
    """
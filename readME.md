## vector space model

`vector space model(vsm) ` in widely used information retrieval model that represents documents as vectors in a high-dimensional space, where each dimension corresponds to a term in the vocabulary. The VSM is based on the assumption that the meaning of a document can be inferred from the distribution of its terms, and that documents with similar content will have similar term distributions.

## what is the program about?

- this Program tries to show a simple vsm implementation
- user can input the query and the program tries to find the relevant document that match the query.
- using cosine simmilarity technique

- it also builds index file which relates the term with their respective occured document

## How to run?

first thing is first u should have nltk installed

    pip install nltk

next simple ''/ just run `main.py` in terminal

example:

- `python main.py`

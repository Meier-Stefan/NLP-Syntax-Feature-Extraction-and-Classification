# SpaCy Tokenizer

## What it Does

The tokenizer uses the spaCy library, so it is called `spacy_tokenizer`:

```python
def spacy_tokenizer(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
    ]
```

It takes the argument `text` which will be a speech.  
SpaCy would firstly run named entity recognition to find places, persons and organisation, and secondly dependency parsing to analyse grammatical relationships, which we both don't need. So we only use parts of the pipeline to speed it up: `nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])`.  
Each token will be lemmatised, which means that it returns the dictionary form of the token.  
Stop words are ignored, so that we don't have to do that later.  
Spaces, punctuation and numbers are ignored by using `token.is_alpha`, it is the same result as using the python function `token.text.isalpha()`.  

## How it Performs

The using the same classifiers as before and keeping the number of features to 3000, the Support Vector Machine achieved a Macro F1 score of 0.4692, which is better than the baseline of 0.4601. The performance could presumably be improved further, but it could already be shown that lemmatising and only using tokens that consist of alphabetic characters was helpful for the task of predicting the political party from
the text of the speech.  

Notes: To ensure that removing stop words with SpaCy instead of with the `TfidfVectorizer` does not have an influence, the two versions have been compared, as shown in the document `appendix.md`. The argument `token_pattern=None` does not have any impact on the code, it is just to silence a user warning. And `# type: ignore[arg-type]` is to silence a Pylance error that is shown because `token_pattern` is defined as type `str`.

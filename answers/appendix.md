# Appendix

## 2d

Results:

```bash
=== 2d ===


=== Best classifier: SVM (linear) ===

Macro F1: 0.4692
                          precision    recall  f1-score   support

           Conservative       0.77      0.94      0.84       312
                 Labour       0.70      0.57      0.63       156
       Liberal Democrat       0.00      0.00      0.00        18
Scottish National Party       0.71      0.29      0.41        42

               accuracy                           0.75       528
              macro avg       0.54      0.45      0.47       528
           weighted avg       0.71      0.75      0.72       528



=== 2d A===


=== Best classifier: SVM (linear) ===

Macro F1: 0.4692
                          precision    recall  f1-score   support

           Conservative       0.77      0.94      0.84       312
                 Labour       0.70      0.57      0.63       156
       Liberal Democrat       0.00      0.00      0.00        18
Scottish National Party       0.71      0.29      0.41        42

               accuracy                           0.75       528
              macro avg       0.54      0.45      0.47       528
           weighted avg       0.71      0.75      0.72       528
```

Adjusted code:

```python

def spacy_tokenizer(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
    ]


def spacy_tokenizer2(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
    ]

    results2d = train_and_evaluate(df, TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=3000, token_pattern=None)) # type: ignore[arg-type]
    print("\n\n=== 2d ===\n")
    best = max(results2d, key=lambda k: results2d[k]["macro_f1"])
    print(f"\n=== Best classifier: {best} ===\n")
    print(f"Macro F1: {results2d[best]['macro_f1']:.4f}")
    print("", results2d[best]["report"])

    results2da = train_and_evaluate(df, TfidfVectorizer(stop_words='english', tokenizer=spacy_tokenizer, max_features=3000, token_pattern=None)) # type: ignore[arg-type]
    print("\n\n=== 2d A===\n")
    best = max(results2da, key=lambda k: results2da[k]["macro_f1"])
    print(f"\n=== Best classifier: {best} ===\n")
    print(f"Macro F1: {results2d[best]['macro_f1']:.4f}")
    print("", results2d[best]["report"])
```

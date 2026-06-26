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

## 3a

### Setting up Ollama

1. **Install Ollama**:
    Download from [ollama.com/download](https://ollama.com/download) and open the app.

2. **Download the model**:

   ```bash
   ollama pull gemma3
   ```

3. **Verify the server is running**:

    ```bash
    curl http://localhost:11434/api/generate \
      -d '{"model": "gemma3", "prompt": "Say hello, I am here", "stream": false}'
    ```

   A JSON response means it is working.

## 3c – Few-shot demo length experiment

### One example

Results across demo lengths (1 example per party, 3 seeds):

| DEMO_LENGTH | seed=26 | seed=42 | seed=7 | mean |
|---|---|---|---|---|
| 400 | 0.5424 | 0.6548 | 0.4153 | 0.5375 |
| 500 | 0.6008 | 0.3854 | 0.5705 | 0.5189 |
| 600 | 0.3393 | 0.4038 | 0.3761 | 0.3731 |
| 700 | 0.5732 | 0.5417 | 0.4958 | 0.5369 |
| 800 | 0.5974 | 0.4218 | 0.4658 | 0.4950 |



### Two examples

Results across demo lengths (2 examples per party, 3 seeds):

| DEMO_LENGTH | seed=26 | seed=42 | seed=7 | mean |
|---|---|---|---|---|
| 400 | 0.3981 | 0.6556 | 0.4232 | 0.4923 |
| 500 | 0.4325 | 0.3897 | 0.3862 | 0.4028 |
| 600 | 0.4236 | 0.4073 | 0.6111 | 0.4807 |
| 700 | 0.5619 | 0.5571 | 0.4419 | 0.5203 |
| 800 | 0.4479 | 0.3914 | 0.4407 | 0.4267 |

## Qwen3

```bash
=== 3a ===


Model: qwen3:14b
Generation parameters: {'num_predict': 10, 'temperature': 0, 'seed': 26, 'num_ctx': 4096}
Parties: Conservative, Democratic Unionist Party, Labour, Scottish National Party


=== 3b – Zero-shot ===

Zero-shot prompt template:

Classify the political party of the following UK House of Commons speech.Choose exactly one: Conservative, Democratic Unionist Party, Labour, Scottish National Party.Answer with just the party name.

Speech: {text}

Party:



Zero-shot: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [09:10<00:00, 21.15s/it]
Macro F1: 0.9345
                         precision    recall  f1-score   support

           Conservative       1.00      0.87      0.93        15
                 Labour       0.78      1.00      0.88         7
Scottish National Party       1.00      1.00      1.00         4

               accuracy                           0.92        26
              macro avg       0.93      0.96      0.93        26
           weighted avg       0.94      0.92      0.93        26



=== 3c – Few-shot ===

Few-shot (seed=26): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [15:11<00:00, 35.04s/it]
Few-shot (seed=42): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [14:08<00:00, 32.64s/it]
Few-shot (seed=7): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/Few-shot (seed=7): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 26/26 [14:08<00:00, 32.65s/it]

--- seed=26 prompt: ---




--- seed=26 results ---

Macro F1: 0.5683
                           precision    recall  f1-score   support

             Conservative       0.81      0.87      0.84        15
Democratic Unionist Party       0.00      0.00      0.00         0
                   Labour       0.75      0.43      0.55         7
  Scottish National Party       0.80      1.00      0.89         4

                 accuracy                           0.77        26
                macro avg       0.59      0.57      0.57        26
             weighted avg       0.79      0.77      0.77        26


--- seed=42 prompt: ---




--- seed=42 results ---

Macro F1: 0.7762
                         precision    recall  f1-score   support

           Conservative       0.92      0.73      0.81        15
                 Labour       0.56      0.71      0.62         7
Scottish National Party       0.80      1.00      0.89         4

               accuracy                           0.77        26
              macro avg       0.76      0.82      0.78        26
           weighted avg       0.80      0.77      0.78        26


--- seed=7 prompt: ---




--- seed=7 results ---

Macro F1: 0.7333
                         precision    recall  f1-score   support

           Conservative       0.92      0.80      0.86        15
                 Labour       0.67      0.57      0.62         7
Scottish National Party       0.57      1.00      0.73         4

               accuracy                           0.77        26
              macro avg       0.72      0.79      0.73        26
           weighted avg       0.80      0.77      0.77        26

```

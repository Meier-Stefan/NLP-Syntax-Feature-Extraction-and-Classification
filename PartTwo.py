import sys
from pathlib import Path
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def spacy_tokenizer(text):
    doc = nlp(text)
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and token.is_alpha
    ]


def clean_data(csv="texts/hansard10000.csv"):
    df = pd.read_csv(csv)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    df = df[df["party"] != "Speaker"]
    top4 = df["party"].value_counts().index[:4]
    df = df[df["party"].isin(top4)]
    df = df[df["speech_class"] == "Speech"]
    df = df[df["speech"].str.len() >= 1000]
    return df


def train_and_evaluate(df, vectorizer):
    X_train, X_test, y_train, y_test = train_test_split(
        df["speech"], df["party"], stratify=df["party"], random_state=26
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = {}
    for name, clf in [
        ("Random Forest", RandomForestClassifier(n_estimators=300)),
        ("SVM (linear)", SVC(kernel='linear')),
    ]:
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        results[name] = {
            "macro_f1": f1_score(y_test, y_pred, average='macro'),
            "report": classification_report(y_test, y_pred, zero_division=0),
        }
    return results


from io_utils import _Tee

if __name__ == "__main__":
    out = Path.cwd() / "outputs" / "PartTwoOutput.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")
    old_stdout = sys.stdout
    sys.stdout = _Tee(f)

    df = clean_data()
    print("\n\n=== 2a ===\n")
    print(df.shape)

    results2b = train_and_evaluate(df, TfidfVectorizer(stop_words='english', max_features=3000))
    print("\n\n=== 2b ===\n")
    for name, metrics in results2b.items():
        print(f"\n=== {name} ===\n")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print("", metrics["report"])

    results2c = train_and_evaluate(df, TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1,3)))
    print("\n\n=== 2c ===\n")
    for name, metrics in results2c.items():
        print(f"\n=== {name} ===\n")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print("", metrics["report"])

    results2d = train_and_evaluate(df, TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=3000, token_pattern=None)) # type: ignore[arg-type]
    print("\n\n=== 2d ===\n")
    best = max(results2d, key=lambda k: results2d[k]["macro_f1"])
    print(f"\n=== Best classifier: {best} ===\n")
    print(f"Macro F1: {results2d[best]['macro_f1']:.4f}")
    print("", results2d[best]["report"])

    sys.stdout = old_stdout
    f.close()

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


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


if __name__ == "__main__":
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

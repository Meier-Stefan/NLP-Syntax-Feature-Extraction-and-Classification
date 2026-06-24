import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import pathlib

MODEL_NAME = "gemma3"
OLLAMA_URL = "http://localhost:11434/api/generate"
GEN_PARAMS = {"num_predict": 10, "temperature": 0, "seed": 26}


PARTIES = ["Conservative", "Labour", "Liberal Democrat", "Scottish National Party"]
ZERO_SHOT_PROMPT = (
    "Classify the political party of the following UK House of Commons speech."
    "Choose exactly one: Conservative, Labour, Liberal Democrat, Scottish National Party."
    "Answer with just the party name.\n\n"
    "Speech: {text}\n\nParty:"
)


def clean_data(csv="texts/hansard500.csv"):
    df = pd.read_csv(csv)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    df = df[df["party"] != "Speaker"]
    top4 = df["party"].value_counts().index[:4]
    df = df[df["party"].isin(top4)]
    df = df[df["speech_class"] == "Speech"]
    df = df[df["speech"].str.len() >= 1000]
    return df


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False} | GEN_PARAMS,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def parse_label(output):
    out = output.strip().lower()
    for party in PARTIES:
        if party.lower() == out or party.lower() in out:
            return party
    return "Unknown"


def classify_zero_shot(test_texts):
    predictions = []
    for text in tqdm(test_texts, desc="Zero-shot"):
        prompt = ZERO_SHOT_PROMPT.format(text=text)
        output = ask_ollama(prompt)
        predictions.append(parse_label(output))
    return predictions


def split_data(df):
    return train_test_split(
        df["speech"], df["party"], stratify=df["party"], random_state=26
    )


if __name__ == "__main__":
    df = clean_data()
    X_train, X_test, y_train, y_test = train_test_split(df["speech"], df["party"], stratify=df["party"], random_state=26)

    print("\n\n=== 3a ===\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Generation parameters: {GEN_PARAMS}")
    print(f"Test set size: {len(y_test)}")
    print("For details, see: ",pathlib.Path("answers/part3_comparison.md").resolve().as_uri())

    print("\n\n=== 3b ===\n")
    print("Zero-shot prompt template:\n")
    print(ZERO_SHOT_PROMPT)
    preds = classify_zero_shot(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

MODEL_NAME = "gemma3"
OLLAMA_URL = "http://localhost:11434/api/generate"
GEN_PARAMS = {"num_predict": 10, "temperature": 0, "seed": 26, "num_ctx": 4096}

NUM_EXAMPLES = 1
DEMO_LENGTH = 400
FEW_SHOT_SEEDS = [26, 42, 7]


def clean_data(csv="texts/hansard500.csv"):
    df = pd.read_csv(csv)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    df = df[df["party"] != "Speaker"]
    top4 = df["party"].value_counts().index[:4]
    df = df[df["party"].isin(top4)]
    df = df[df["speech_class"] == "Speech"]
    df = df[df["speech"].str.len() >= 1000]
    return df


def get_parties(df):
    return sorted(df["party"].unique().tolist())


def party_list_str(parties):
    return ", ".join(parties)


def build_zero_shot_prompt(parties):
    return (
        "Classify the political party of the following UK House of Commons speech."
        f"Choose exactly one: {party_list_str(parties)}."
        "Answer with just the party name.\n\n"
        "Speech: {text}\n\nParty:"
    )


def build_few_shot_prompt_template(parties):
    return (
        "The task is to classify the political party of a UK House of Commons speech.\n"
        f"Choose exactly one: {party_list_str(parties)}.\n\n"
        "Here are example speech starts (one per party):\n\n"
        "{demos}"
        "Classify the political party of the following speech.\n"
        "Speech: {text}\n\nParty:"
    )


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False} | GEN_PARAMS,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def parse_label(output, parties):
    out = output.strip().lower()
    for party in parties:
        if party.lower() == out or party.lower() in out:
            return party
    return "Unknown"


def classify_zero_shot(df):
    parties = get_parties(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df["speech"], df["party"], stratify=df["party"], random_state=26
    )
    prompt_template = build_zero_shot_prompt(parties)
    predictions = []
    for text in tqdm(X_test, desc="Zero-shot"):
        prompt = prompt_template.format(text=text)
        output = ask_ollama(prompt)
        predictions.append(parse_label(output, parties))
    return {
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "report": classification_report(y_test, predictions, zero_division=0),
    }


def build_few_shot_prompt(text, examples, parties):
    demo_block = ""
    for speech, party in examples:
        truncated = speech[:DEMO_LENGTH]
        demo_block += f"Speech: {truncated}...\nParty: {party}\n\n"
    template = build_few_shot_prompt_template(parties)
    return template.format(demos=demo_block, text=text)


def classify_few_shot(df):
    parties = get_parties(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df["speech"], df["party"], stratify=df["party"], random_state=26
    )
    train_df = X_train.to_frame("speech").join(y_train)

    results = {}
    for seed in FEW_SHOT_SEEDS:
        examples_list = get_examples(train_df, parties, seed)

        predictions = []
        for text in tqdm(X_test, desc=f"Few-shot (seed={seed})"):
            prompt = build_few_shot_prompt(text, examples_list, parties)
            output = ask_ollama(prompt)
            predictions.append(parse_label(output, parties))

        results[str(seed)] = {
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "report": classification_report(y_test, predictions, zero_division=0),
        }
    return results


def get_examples(train_df, parties, seed):
    examples = []
    for party in parties:
        subset = train_df[train_df["party"] == party]
        sampled = subset.sample(n=NUM_EXAMPLES, random_state=seed)
        examples.extend(zip(sampled["speech"], sampled["party"]))
    return examples


if __name__ == "__main__":
    df = clean_data()
    parties = get_parties(df)

    print("\n\n=== 3a ===\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Generation parameters: {GEN_PARAMS}")
    print(f"Parties: {party_list_str(parties)}")

    print("\n\n=== 3b – Zero-shot ===\n")
    zero_prompt = build_zero_shot_prompt(parties)
    print("Zero-shot prompt template:\n")
    print(zero_prompt)
    print("\n\n")
    results_zero = classify_zero_shot(df)
    print(f"Macro F1: {results_zero['macro_f1']:.4f}")
    print(results_zero["report"])

    print("\n\n=== 3c – Few-shot ===\n")
    results_few = classify_few_shot(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["speech"], df["party"], stratify=df["party"], random_state=26
    )
    train_df = X_train.to_frame("speech").join(y_train)

    for seed in FEW_SHOT_SEEDS:
        examples = get_examples(train_df, parties, seed)
        print(f"\n--- seed={seed} prompt: ---\n")
        print(build_few_shot_prompt(X_test.iloc[0], examples, parties))
        print("\n\n")
        print(f"--- seed={seed} results ---\n")
        print(f"Macro F1: {results_few[str(seed)]['macro_f1']:.4f}")
        print(results_few[str(seed)]["report"])

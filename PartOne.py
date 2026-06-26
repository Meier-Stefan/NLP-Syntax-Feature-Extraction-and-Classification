from collections import Counter
import math
import nltk
import re
import spacy
import sys
from pathlib import Path
import pandas as pd


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        pronunciations = d[word]
        phones = pronunciations[0]
        return sum(1 for p in phones if p[-1].isdigit())
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalpha()]
    if not words or not sentences:
        return 0.0
    total_syllables = sum(count_syl(w, d) for w in words)
    return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (total_syllables / len(words))




def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""

    rows = []

    for file_path in Path(path).glob("*.txt"):
        stem = file_path.stem
        raw_title, author, year = stem.split("-")
        title = raw_title.replace("_", " ")
        text = file_path.read_text(encoding="utf-8")
        rows.append({"text": text, "title": title, "author": author, "year": int(year)})
    df = pd.DataFrame(rows)
    df = df.sort_values("year").reset_index(drop=True)

    return df


def parse(df, store_path=Path.cwd() / "outputs", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["doc"] = list(nlp.pipe(df["text"], batch_size=1))
    df.to_pickle(store_path / out_name)
    return df


def load_parsed_df(path=Path.cwd() / "outputs" / "parsed.pickle"):
    """Loads the parsed dataframe from a pickle file."""
    return pd.read_pickle(path)


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    if not tokens:
        return 0.0
    num_types = len(set(tokens))
    num_tokens = len(tokens)
    return num_types / num_tokens


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def get_common_subjects(doc, n=10):
    subjects = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and not token.is_stop:
            subjects.append(token.lemma_.lower())
    counts = Counter(subjects)
    return counts.most_common(n)


def get_subject_verb_pmi(doc, subject, n=10):
    pairs = []
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
            verb = token.head.lemma_.lower()
            if not verb.isalpha():
                continue
            pairs.append((token.lemma_.lower(), verb))

    N = len(pairs)
    if N == 0:
        return []

    verb_counts = Counter(verb for _, verb in pairs)
    subj_count = sum(1 for s, _ in pairs if s == subject)
    if subj_count == 0:
        return []

    subj_verb_counts = Counter(verb for s, verb in pairs if s == subject)
    results = []
    for verb, subj_c in subj_verb_counts.items():
        v_c = verb_counts[verb]
        pmi = math.log2((subj_c * N) / (v_c * subj_count))
        results.append((verb, round(pmi, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]



from io_utils import _Tee

if __name__ == "__main__":
    out = Path.cwd() / "outputs" / "PartOneOutput.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w")
    old_stdout = sys.stdout
    sys.stdout = _Tee(f)

    novels_path = Path.cwd() / "texts" / "novels"
    pickle_path = Path.cwd() / "outputs" / "parsed.pickle"

    if pickle_path.exists():
        df = load_parsed_df(pickle_path)
        print(df[["title", "author", "year"]])
    else:
        df = read_novels(novels_path)
        print(df)
        print("\nParsing texts with spaCy...")
        parse(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Tokens in first doc: {len(df['doc'].iloc[0])}")

    print("\nType-token ratios:")
    for title, ttr in get_ttrs(df).items():
        print(f"  {title}: {ttr:.4f}")

    print("\nFlesch-Kincaid Reading Ease Score:")
    for title, score in get_fks(df).items():
        print(f"  {title}: {score}")

    print("\nMost common syntactic subjects:\n")
    for _, row in df.iterrows():
        subjects = get_common_subjects(row["doc"], n=10)
        print(f"=== {row['title']} ===")
        for word, count in subjects:
            print(f"    {word}: {count}")

    print("\nVerbs with highest PMI for subject 'he':")
    for _, row in df.iterrows():
        verbs = get_subject_verb_pmi(row["doc"], "he", n=10)
        print(f"=== {row['title']} ===")
        for verb, pmi in verbs:
            print(f"    {verb}: {pmi:.4f}")

    print("\nVerbs with highest PMI for subject 'she':")
    for _, row in df.iterrows():
        verbs = get_subject_verb_pmi(row["doc"], "she", n=10)
        print(f"=== {row['title']} ===")
        for verb, pmi in verbs:
            print(f"    {verb}: {pmi:.4f}")

    sys.stdout = old_stdout
    f.close()

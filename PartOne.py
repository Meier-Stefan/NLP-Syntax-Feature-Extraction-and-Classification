import nltk
import spacy
from pathlib import Path
import pandas as pd


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass




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


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass


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


#.. add functions for part (e) here



if __name__ == "__main__":
    path = Path.cwd() / "texts" / "novels"
    df = read_novels(path)
    print(df)
    print("\nType-token ratios:")
    for title, ttr in get_ttrs(df).items():
        print(f"  {title}: {ttr:.4f}")

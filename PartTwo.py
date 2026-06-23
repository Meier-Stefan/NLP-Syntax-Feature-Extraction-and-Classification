import pandas as pd

def clean_data(csv="texts/hansard10000.csv"):
    df = pd.read_csv(csv)
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")
    df = df[df["party"] != "Speaker"]
    top4 = df["party"].value_counts().index[:4]
    df = df[df["party"].isin(top4)]
    df = df[df["speech_class"] == "Speech"]
    df = df[df["speech"].str.len() >= 1000]
    return df


if __name__ == "__main__":
    df = clean_data()
    print(df.shape)

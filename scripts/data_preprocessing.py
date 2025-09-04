import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath="data/IMDB_dataset.csv"):
    df = pd.read_csv(filepath)
    df['review'] = df['review'].fillna("")
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    return train_texts, test_texts, train_labels, test_labels

if __name__ == "__main__":
    train_texts, test_texts, train_labels, test_labels = load_and_split_data()
    print(f"Train size: {len(train_texts)}, Test size: {len(test_texts)}")

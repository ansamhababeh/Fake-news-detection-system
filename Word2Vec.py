import numpy as np
import pandas as pd
import ast
# pip install gensim
from gensim.models import Word2Vec


def safe_literal_eval(value):
    """Safely converts string representations of lists to actual lists."""
    try:
        return ast.literal_eval(value) if isinstance(value, str) and value.startswith('[') and value.endswith(']') else []
    except (SyntaxError, ValueError):
        return []


def load_data(file_path, encoding='utf-8'):
    """Loads CSV data and applies safe conversion to lists."""
    print("Loading data...")
    data = pd.read_csv(file_path, encoding=encoding)
    data['tkn_stp_lm'] = data['tkn_stp_lm'].apply(safe_literal_eval)
    print("Data loading completed. Shape:", data.shape)
    return data


def train_word2vec(data, vector_size=100, window=5, min_count=5, workers=4):
    """Initializes and trains a Word2Vec model."""
    print("Training Word2Vec model...")
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.build_vocab(corpus_iterable=data['tkn_stp_lm'])
    model.train(corpus_iterable=data['tkn_stp_lm'], total_examples=model.corpus_count, epochs=model.epochs)
    print("Word2Vec training completed.")
    return model


def save_model(model, model_path="word2vec_model.bin"):
    """Saves the trained Word2Vec model."""
    if model:
        model.save(model_path)
        print(f"Model saved to {model_path}.")


def get_word_vector(model, word):
    """Returns the vector of a given word if it exists in the model's vocabulary."""
    if word in model.wv:
        return model.wv[word]
    print(f"'{word}' not found in vocabulary.")
    return None


def get_sentence_vector(model, tokens, vector_size):
    """Computes sentence embedding by averaging word vectors."""
    vectors = [model.wv[word] for word in tokens if isinstance(word, str) and word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)


def generate_embeddings(data, model, vector_size):
    """Generates embeddings for each document and saves to CSV."""
    print("Generating sentence embeddings...")
    data['word2vec_embedding'] = data['tkn_stp_lm'].apply(lambda x: get_sentence_vector(model, x, vector_size).tolist())
    data[['word2vec_embedding']].to_csv("word2vec_features.csv", index=False)
    print("Embeddings saved to 'word2vec_features.csv'.")


# Usage
if __name__ == "__main__":
    file_path = "Word2VecData.csv"
    vector_size = 100

    data = load_data(file_path)
    model = train_word2vec(data, vector_size=vector_size)
    save_model(model)

    # Example word vector retrieval
    word = "president"
    vector = get_word_vector(model, word)
    if vector is not None:
        print(f"Vector representation of '{word}':\n", vector)

    generate_embeddings(data, model, vector_size)

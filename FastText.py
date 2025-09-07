import os
import pandas as pd
# pip install fasttext
import fasttext
import numpy as np


def load_data(file_path):
    print("🔄 Loading data...")
    data = pd.read_csv(file_path)
    print("✅ Data loaded. Shape:", data.shape)
    return data


def prepare_fasttext_format(data, output_txt_path):
    print("🔄 Preparing FastText training file...")
    data['formatted_text'] = '__label__' + data['label'].astype(str) + ' ' + data['text']
    data['formatted_text'].to_csv(output_txt_path, index=False, header=False)
    print(f"✅ FastText formatted file saved to: {output_txt_path}")


def train_fasttext_model(train_txt_path, model_path="fasttext_model.bin"):
    print("🔧 Training FastText model...")
    model = fasttext.train_supervised(train_txt_path, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1)
    model.save_model(model_path)
    print(f"✅ FastText model saved to: {model_path}")
    return model


def extract_embeddings_with_labels(data, model):
    print("🧠 Generating sentence embeddings with labels...")
    embeddings = np.array([model.get_sentence_vector(str(text)) for text in data["text"].fillna("")])
    labels = data["label"].values.reshape(-1, 1)
    final_data = np.hstack((embeddings, labels))
    return final_data


def save_embeddings_with_labels(final_data, output_path="fasttext_features.csv"):
    print("💾 Saving embeddings + labels...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    columns = [f"ft_feature_{i}" for i in range(final_data.shape[1] - 1)] + ["label"]
    df = pd.DataFrame(final_data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"✅ Full feature file with labels saved to: {output_path}")


# === MAIN ===
if __name__ == "__main__":
    input_file = "FastTextData.csv"
    train_txt_path = "fasttext_train.txt"
    model_path = "fasttext_model.bin"
    output_path = "fasttext_features.csv"

    data = load_data(input_file)
    prepare_fasttext_format(data, train_txt_path)
    model = train_fasttext_model(train_txt_path, model_path)
    full_data = extract_embeddings_with_labels(data, model)
    save_embeddings_with_labels(full_data, output_path)

from flask import Flask, request, jsonify, render_template
from lime.lime_text import LimeTextExplainer
import numpy as np
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback
import random  

from utils import (
    clean_text,
    preprocess_for_word2vec_BiLSTM,
    predict_using_w2v_bilstm
)

app = Flask(__name__)
app.secret_key = "1492002key"  # 🔐 اكتبي أي قيمة سرية


# Load the trained BiLSTM model
bilstm_model = load_model("my_model.h5")

class_names = ['Fake', 'Real']
explainer = LimeTextExplainer(class_names=class_names)


def explain_prediction(article_text):
    def predict_proba(texts):
        results = []
        for text in texts:
            cleaned = clean_text(text)
            vectors, _ = preprocess_for_word2vec_BiLSTM(cleaned)
            # vectors = pad_sequences([vectors], maxlen=300, dtype='float32', padding='post', truncating='post')
            pred = predict_using_w2v_bilstm(bilstm_model, vectors)
            prob = float(pred)
            print("Prediction for texttttttttttttttttttttttttttttttttt:", text[:30], "=>", prob)

            results.append([1 - prob, prob])
        return np.array(results)

    explanation = explainer.explain_instance(article_text, predict_proba, num_features=10)
    return explanation.as_list()


def generate_detailed_explanation(explanation_list, label):
    top_words = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)[:5]
    insights = []
    for word, weight in top_words:
        insight = {
            "word": word,
            "impact": weight,
            "reason": "commonly found in fake news" if weight < 0 else "frequently used in reliable articles"
        }
        insights.append(insight)
    if label == "Fake":
        summary = (
            "The model suspects this article might be fake due to terms like "
            + ", ".join(f'\"{i["word"]}\"' for i in insights[:3])
            + " that often appear in misleading content."
        )
    else:
        summary = (
            "This article appears trustworthy thanks to words like "
            + ", ".join(f'\"{i["word"]}\"' for i in insights[:3])
            + ", typically seen in credible sources."
        )
    return summary, insights


# SHAP Explanation
def explain_with_shap(vectors, tokens, model):
    MAX_SEQUENCE_LEN = 150
    EMBEDDING_DIM = 100

    # padding للمقالة الواحدة (vectors)
    padded = np.zeros((1, MAX_SEQUENCE_LEN, EMBEDDING_DIM), dtype='float32')
    length = min(len(vectors), MAX_SEQUENCE_LEN)
    if length > 0:
        padded[0, :length, :] = vectors[:length]

    # نكررها 10 مرات كـ background
    background = np.repeat(padded, 10, axis=0)  # shape (10, 150, 100)

    # التفسير باستخدام SHAP
    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(padded)  # shape (1, 150, 100)

    word_shap_pairs = []
    for i in range(min(len(tokens), len(shap_values[0][0]))):
        impact = float(np.linalg.norm(shap_values[0][0][i]))
        word_shap_pairs.append((tokens[i], impact))


    return sorted(word_shap_pairs, key=lambda x: abs(x[1]), reverse=True)



@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

from utils import get_daily_articles_from_csv

def normalize_source_name(source):
    return source.lower().replace(" ", "_").replace("-", "_")


@app.route("/daily-news")
def daily_news():
    try:
        articles = get_daily_articles_from_csv("C:\\Users\\HP\\OneDrive\\Desktop\\Graduation-Project-main\\News_Crawl\\news_data.csv")

        results = []
        for article in articles:
            cleaned = clean_text(article["content"])
            vectors, tokens = preprocess_for_word2vec_BiLSTM(cleaned)
            prediction = predict_using_w2v_bilstm(bilstm_model, vectors)
            prediction_value = float(prediction)
            label = "Real" if prediction_value > 0.5 else "Fake"

            results.append({
            "title": article["title"],
            "content": article["content"],
            "url": article["url"],
            "label": label,
            "source": normalize_source_name(article.get("source", "default")),

            "confidence": round(prediction_value * 100 if label == "Real" else (1 - prediction_value) * 100, 2)
        })
          


            

        random.shuffle(results)
        return render_template("daily_news.html", articles=results)

    except Exception as e:
        print("❌ Error in /daily-news:", e)
        return "Internal Server Error", 500



import smtplib
from email.mime.text import MIMEText
from flask import request, redirect, flash

@app.route("/send-message", methods=["POST"])
def send_message():
    try:
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        # إعداد الرسالة
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg = MIMEText(body)
        msg["Subject"] = "New Contact Message from NewsScope"
        msg["From"] = email
        msg["To"] = "israajdali9@gmail.com"  # ✉️ حطي إيميلك هون

        # إرسال البريد (باستخدام Gmail كمثال)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        smtp_username = "israajdali9@gmail.com"  # بريدك
        smtp_password = "lqyo dvve puil xfpm"  

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)

        flash("Message sent successfully!", "success")
        return redirect("/contact")

    except Exception as e:
        print("❌ Error sending message:", e)
        flash("Error sending message. Try again later.", "error")
        return redirect("/contact")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        article = data.get("article", "")
        print("Received article:", article)

        cleaned = clean_text(article)
        print("Cleaned article:", cleaned)

        vectors, tokens = preprocess_for_word2vec_BiLSTM(cleaned)
        #vectors = pad_sequences([vectors], maxlen=300, dtype='float32', padding='post', truncating='post')

        print("Vectors shape:", vectors.shape, "Tokens:", tokens[:5])

        prediction = predict_using_w2v_bilstm(bilstm_model, vectors)
        prediction_value = float(prediction)
        label = "Real" if prediction_value > 0.5 else "Fake"

        shap_insights = explain_with_shap(vectors, tokens, bilstm_model)

        top_words = shap_insights[:5]
        word_insights = [{
            "word": word,
            "impact": impact,
            "reason": "commonly found in fake news" if impact < 0 else "frequently used in reliable articles"
        } for word, impact in top_words]

        if label == "Fake":
            human_explanation = (
                "The model suspects this article might be fake due to terms like "
                + ", ".join(f'\"{i["word"]}\"' for i in word_insights[:3])
                + " that often appear in misleading content."
            )
        else:
            human_explanation = (
                "This article appears trustworthy thanks to words like "
                + ", ".join(f'\"{i["word"]}\"' for i in word_insights[:3])
                + ", typically seen in credible sources."
            )

        return jsonify({
            "prediction": prediction_value,
            "label": label,
            "shap_insights": shap_insights,
            "human_explanation": human_explanation,
            "insights": word_insights
        })

    except Exception as e:
        print("❌ Error in /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True)
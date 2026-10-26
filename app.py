
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("fake_news_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "").strip()

    if not news:
        return render_template(
            "index.html",
            error="Please enter a news article."
        )

    # Predict news category
    prediction = model.predict([news])[0]

    # Get model probability estimates
    probabilities = model.predict_proba([news])[0]

    # Confidence for the predicted class
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    confidence = round(
        float(probabilities[class_index]) * 100, 2
    )

    if prediction == 0:
        result = "Likely Fake News"
        result_type = "fake"
    else:
        result = "Likely Real News"
        result_type = "real"

    return render_template(
        "index.html",
        prediction=result,
        result_type=result_type,
        confidence=confidence,
        news=news
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
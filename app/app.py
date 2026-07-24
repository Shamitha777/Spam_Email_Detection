from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    recommendation = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]

        # Convert message into TF-IDF features
        vector = vectorizer.transform([message])

        # Predict
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = round(max(probability) * 100, 2)

        # Convert numeric prediction to text
        prediction = "Spam" if prediction == 1 else "Not Spam"

        # Recommendation
        if prediction == "Spam":
            recommendation = "⚠️ Avoid clicking links or sharing personal information."
        else:
            recommendation = "✅ This message appears to be safe."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        recommendation=recommendation,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
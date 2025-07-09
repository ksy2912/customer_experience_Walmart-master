from flask import Flask, request, jsonify
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize

# Initial Setup
nltk.download('punkt')
app = Flask(__name__)

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.route('/summarize-reviews', methods=['POST'])
def summarize_reviews():
    data = request.get_json()
    reviews = data.get("reviews", [])

    if not reviews or not isinstance(reviews, list):
        return jsonify({"error": "Please provide a list of reviews."}), 400

    try:
        # Join and chunk for summarization
        joined_reviews = " ".join(reviews)
        sentences = sent_tokenize(joined_reviews)
        input_text = " ".join(sentences[:min(len(sentences), 10)])

        summary_output = summarizer(input_text, max_length=60, min_length=15, do_sample=False)
        summary = summary_output[0]['summary_text']

        # Sentiment analysis
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total = len(reviews)

        for review in reviews:
            score = sentiment_analyzer.polarity_scores(review)
            compound = score['compound']
            if compound >= 0.05:
                sentiment_counts['positive'] += 1
            elif compound <= -0.05:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1

        # Sentiment result
        percentages = {k: round((v / total) * 100) for k, v in sentiment_counts.items()}
        dominant = max(percentages, key=percentages.get)
        if percentages['positive'] > 30 and percentages['negative'] > 30:
            dominant = "mixed"

        return jsonify({
            "summary": summary,
            "sentiment": dominant,
            "score": percentages
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)

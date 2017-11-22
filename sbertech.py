from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

app = Flask(__name__)

stop_words = set(stopwords.words("russian"))
vectorizer = joblib.load("models/tfidf_matrix.pkl")
predictor = joblib.load("models/model_v1.pkl")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.json:
        json_ = request.json
    elif request.form:
        json_ = {'Review': request.form['Review'], 'Title': request.form['Title']}

    preprocessed_data = preprocess(json_)

    result = list(evaluate(preprocessed_data))
    print(result)
    if request.json:
        return jsonify({'Rating': str(result)})
    else:
        return render_template('result.html', result=result)


def preprocess(data):
    text = data['Review'] + " " + data['Title']
    lambda_1 = lambda t: ' '.join(CountVectorizer().build_tokenizer()(t.lower()))
    lambda_2 = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    return lambda_1(lambda_2(text))


def evaluate(comment):
    vector = vectorizer.transform([comment])
    return predictor.predict(vector)


if __name__ == '__main__':
    vectorizer = joblib.load("models/tfidf_matrix.pkl")
    predictor = joblib.load("models/model_v1.pkl")
    app.run(debug=True)

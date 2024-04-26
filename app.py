import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


from flask import Flask, render_template, request, redirect, url_for, session
from models import db, User
from newspaper import Article
from textblob import TextBlob
from joblib import load

model = load("ml_model.joblib")


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return False
    elif n == 1:
        return True
    
def manual_testing(news, model):
    vectorization = TfidfVectorizer()
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = model.predict(new_xv_test)

    return output_lable(pred_RFC[0])

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database for simplicity
db.init_app(app)

# Initialize database
with app.app_context():
    db.create_all()

# Refactor sentiment analysis into a separate function
def get_sentiment(text):
    """
    Perform sentiment analysis on the given text.
    Returns the sentiment as a string: Positive, Negative, or Neutral.
    """
    text_blob = TextBlob(text)
    try:
        sentiment = text_blob.sentiment
        return "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"
    except Exception as e:
        print("Sentiment analysis error:", e)
        return "Unknown"

# Route to handle news submission
@app.route('/submit_news', methods=['POST'])
def submit_news():
    if 'username' in session:
        username = session['username']
        url = request.form.get('url')
        news_text = request.form.get('news_text')
        
        if url:
            article = Article(url)
            article.download()
            article.parse()
            news = {"headline": article.title, "content": article.text}
            result = {"result": manual_testing(article.text, model)}
            if result[result]:
                print("It is a Real News")
            else:
                print("It is a fake News")

        elif news_text:
            news = {"headline": "Custom News", "content": news_text}
            result = {"result": manual_testing(news_text, model)}
            if result[result]:
                print("It is a Real News")
            else:
                print("It is a fake News")
        
        # Perform sentiment analysis
        news["sentiment"] = get_sentiment(news["content"])
        
        # Store the submitted news in the session
        session['submitted_news'] = news
        
        return redirect(url_for('dashboard'))  # Redirect to dashboard to display the submitted news
    return redirect(url_for('login'))

# Route for the dashboard
@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        news = session.pop('submitted_news', None)  # Retrieve submitted news from session
        return render_template('dashboard.html', username=username, news=news)
    return redirect(url_for('login'))

# Route to handle user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('signup.html', message="Username already exists")
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        session['username'] = username  # Automatically log in the user after signup
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

# Route to handle user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')

# Route to handle user logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)













    




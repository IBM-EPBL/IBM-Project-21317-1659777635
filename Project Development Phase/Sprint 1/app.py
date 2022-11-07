from flask import Flask, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URL'] = 'sqlite:///test.db'
# db = SQLAlchemy(app)

# class Todo(db.Model):
#     id = db.Column(dc.Integer, primary_key = True)
#     content = db.Column(db.String(200), nullable = False)
#     completed = dv.Column(db.Integer, default = 0)
#     date_created = db.Column(db.DateTime, default = datetime.utcnow)

#     def __repr__(self):
#         return '<Task %r>'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile.html')
def profile():
    return render_template('profile.html')

if __name__ == "__main__":
    app.run(debug=True)

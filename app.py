# import some stuff here


# create the Flask application


# routes go here


# script initialization





from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route('/')

def home():
    return 'Hello World!'

@app.route('/train')
def train():
    return "I'm training!!!!"

@app.route('/titanic')
def titanic():
    return render_template('index.html')

@app.route('/wine')
def wine():
    return render_template('wine.html')

if __name__ == '__main__':
    app.run(debug=True)
    
# titanic dataset predict survival
    # sex age

# create a plot and save it and call it

# create a step by step and tell them i think we'll only get here
# UCI
# wine quality one
# bike share data

# create a flask app
# create different routes
# train a model
# deploy a model
# create and return an image
# use templates (jinja2)

from flask import Flask
from flask import render_template

application = Flask(__name__)

@application.route('/')
def start():
    return 'Hello World!'

@application.route('/train')
def train():
    return "I'm training!!!!"

@application.route('/titanic')
def titanic():
    return render_template('index.html')

@application.route('/wine')
def wine():
    return render_template('wine.html')

if __name__ == '__main__':
    application.debug = True
    application.run()
    
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

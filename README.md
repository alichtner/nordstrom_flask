# Flask and Data Science Workshop

*September 6, 2017*

Aaron Lichtner, Data Scientist @ Nordstrom

---

Flask is a web framework for Python. In other words, it's a way to use python to create websites, web apps and APIs.

###Workshop Topics
 - [Creating your flask dev environment](#Creating your Flask Environment)
 - [Deploying a basic flask app](#Basic Flask App
)
 - [Using Flask to train and predict with a model](#Now let's do some Data Science
)
 - Using a Bootstrap templates to add some style
 - jinja templating
 
# Creating your Flask Environment
### python 3

- Install python 3
`brew install python3`

- Set up the virtual environment
`virtualenv -p python3 venv3`

- Activate the virtual environment
`source venv3/bin/activate`

### python 2

- Install python 2
`brew install python2`

- Set up the virtual environment
`virtualenv -p python2 venv2`

- Activate the virtual environment
`source venv2/bin/activate`

- Install flask
`pip install flask`

#Basic Flask App

1. In your root directory, create your `app.py` file, import flask and create your flask object.
```python
from flask import Flask

# create the flask object
app = Flask(__name__)
```

2. Add your first route. Routes connect web pages to unique python functions. In this example, the root page of the site, `yoursite/`, should trigger the `home() function and return 'Hello World!' as the server response to the webpage. 
```python
@app.route('/')
def home():
    return 'Hello World!'
```

3. Now just add some code at the bottom to tell python what to do when the script is run.
```python
if __name__ == '__main__':
    app.run(debug=True)
```
- *Note: `debug=True` allows for quicker development since you don't have to keep restarting your app when you change it.*

4. Run `python app.py`

5. Check it out!

- [Your Locally Running App](http://127.0.0.1:5000/)

6. Let's add another route. Rather than just pure text, let's return some HTML.

```python
@app.route('/jazzhands')
def jazzhands():
    return "<h1>Here's some <i>Pizzazz!</i></h1>"
```

7. One final route in our basic app. This route will use what's known as **dynamic routing**. These allow more flexibility with your urls and the ability to pass variables straight through the url. Variables must be passed in between angled brackets '<>'.

```python
@app.route('/twice/<int:x>') # int says the expected data type
def twice(x):
    output = 2 * x
    return 'Two times {} is {}'.format(x, output)
```
 - *Note: if you only wish to pass a string in dynamically, say someone's username, you only need `/<username>`, the `/<int:>` part isn't necessary.*

#Now let's do some Data Science

We will be building a survival classifier using the [Titanic Survival Dataset](https://www.kaggle.com/c/titanic/data). 

The code to read in the data, split it up and train the model has already been written for you. We're going to focus on how to implement a training function 

When submitting a form, make sure to allow the POST method to be used with your route.

```python
pip install sklearn 
```

```python 
from flask import Flask, request
```

1. train model without using a route, predict with it, then run it through a template

You are now able to train/load a model into a flask app and predict with it. Now on to some more advanced topics.

## Basic Flask App Organization
1. Create your file and folder structure.
```python
# the most basic flask app
 yourapp/
    |
    | - app.py
```
- Example of a more complex application structure
```python
 yourapp/
    |
    | - app.py
    | - static/
            | - css/
            | - resources/
    | - templates/
            | - index.html
    | - data/
```

# Get some HTML Templates 
[Bootstrap Templates](https://startbootstrap.com/)

```python 
from flask import Flask, request, render_template
```

# Keep Going!
- plot a visualization and present it on the webpage
- train a different type of model and create a new route to its results

Further Questions: 
- Aaron Lichtner, Data Scientist @ Nordstrom
- LinkedIn: https://www.linkedin.com/in/aaronlichtner/
- Email: [aaron.lichtner@nordstrom.com](aaron.lichtner@nordstrom.com)
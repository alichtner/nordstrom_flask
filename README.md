# Flask and Data Science Workshop

*September 6, 2017*

Aaron Lichtner, Data Scientist @ Nordstrom

---

Flask is a web framework for Python. In other words, it's a way to use python to create websites, web apps and APIs.

### Workshop Topics
 - [Creating your flask dev environment](#creating-your-flask-environment)
 - [Deploying a basic flask app](#basic-flask-app)
 - [Using Flask to train and predict with a model](#now-lets-do-some-data-science)
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

# Basic Flask App

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

# Now let's do some Data Science

We will be building a survival classifier using the [Titanic Survival Dataset](https://www.kaggle.com/c/titanic/data). Our goal is to create an interface for a user to make and view predictions.

The code to read in the data, split it up and train the model has already been written for you. We're going to focus on how to implement the and predict with it through the flask web interface. 


### Install and import additional packages 
1. First make sure you have all the required packages in your virtualenv.

```python
# python 2
pip install pandas, sklearn, scipy

# python 3
pip3 install pandas, sklearn, scipy
```

2. Import the libraries required for modeling. `render_template` and `request` are needed for us to get data from the web interface to the flask app and then to present the results in a more visually appealing way than basic text.
```python
from flask import Flask     # you already should have this 
from flask import render_template, request

# modeling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

### Build the model
1. The following code will read in the `titanic_data.csv`, clean it up, split it into test and training sets and then train a simple logistic regression to predict the probability of survival. Paste this code right at the start of the script initialization. The model will be available in the namespace of the flask app.

```python
# read in data and clean the gender column
if __name__ == '__main__':
    # build a basic model for titanic survival
    titanic_df = pd.read_csv('data/titanic_data.csv')
    titanic_df['sex_binary'] = titanic_df['sex'].map({'female': 1, 'male': 0})
    
    # choose our features and create test and train sets
    features = [u'pclass', u'age', u'sibsp', u'parch', u'fare', u'sex_binary', 'survived']
    train_df, test_df = train_test_split(titanic_df)
    train_df = train_df[features].dropna()
    test_df = test_df[features].dropna()
    
    features.remove('survived')
    X_train = train_df[features]
    y_train = train_df['survived']
    X_test = test_df[features]
    y_test = test_df['survived']
    
    # fit the model
    L1_logistic = LogisticRegression(C=1.0, penalty='l1')
    L1_logistic.fit(X_train, y_train)
    
    # check the performance
    target_names = ['Died', 'Survived']
    y_pred = L1_logistic.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # start the app
    app.run(debug=True)
```

- Rerunning the script now should show us the classification_report from the logistic regression model in the terminal. We haven't hooked up any flask routes to the model however. Let's change that.

### Rendering HTML templates from Flask
I've created a basic HTML template where we can build a user interface for predicting with our amazing model.

```python
@app.route('/titanic', methods=['GET','POST'])
def titanic():
    return render_template('titanic.html')
```

### Getting prediction inputs into flask

We use the following variables to predict whether someone will survive the titanic:
- Ticket Class
- Age
- \# Siblings & Spouses 
- \# Children & Parents
- Ticket Fare
- Gender

In order to hook up the web interface with the model we have to allow the user to input all of the required model parameters. The easiest way to do this is with a simple web form. 

![](static/resources/form.png)

In the `titanic.html` file, add the following
```html
            <form action="/titanic" method="post" id="titanic_predict">
                 <div>
                    <label for="name">Ticket Class: 1, 2, or 3</label>
                    <input type="text" id="class" name="predict_class" value=1>
                </div>
                <div>
                    <label for="name">Age: 0 - 100 </label>
                    <input type="text" id="age" name="predict_age" value=25>
                </div>
                <div>
                    <label for="name"># Siblings and Spouses</label>
                    <input type="text" id="sibsp" name="predict_sibsp" value=1>
                </div>
                <div>
                    <label for="name"># Children and Parents</label>
                    <input type="text" id="parch" name="predict_parch" value=0>
                </div>
                <div>
                    <label for="name">Ticket Fare: 0 - 500 ($) </label>
                    <input type="text" id="fare" name="predict_fare" value=250>
                </div>
                <div>
                    <label for="name">Gender: M or F</label>
                    <input type="text" id="sex" name="predict_sex" value='F'>
                </div>
            </form>
            <button class="btn" type="submit" form="titanic_predict" value="Submit">Predict</button>
```
When you go check out your page you should see the web form there for you. Press the predict button though and you'll get an error saying that method isn't allowed. Flask routes by default enable the 'GET' method but if we want to allow any additional functionality, such as submitting data to the flask server via a webform we'll need to enable those explicitly.

```python
@app.route('/titanic', methods=['GET','POST'])
def titanic():
    return render_template('titanic.html')
```

```html
                {% if data.prediction %}
                    <h1>{{data.prediction}}</h1>
                    <h5>Ticket Class: {{data.predict_class}}</h5>
                    <h5>Age: {{data.predict_age}}</h5>
                    <h5>Siblings & Spouses: {{data.predict_sibsp}}</h5>
                    <h5>Children & Parents: {{data.predict_parch}}</h5>
                    <h5>Ticket Fare: ${{data.predict_fare}}</h5>
                    <h5>Gender: {{data.predict_sex}}</h5>
                {% endif %}
```

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
            | - js/
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
- plot a visualization of the data and present it on the web page
- have the `titanic/` route return a representative image based on the prediction of survive or not survive
- train an entirely different model and create a new route to its results

### Further Questions: 
- Aaron Lichtner, Data Scientist @ Nordstrom
- LinkedIn: https://www.linkedin.com/in/aaronlichtner/
- Email: [aaron.lichtner@nordstrom.com](aaron.lichtner@nordstrom.com)
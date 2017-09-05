from flask import Flask
from flask import render_template, request

# modeling packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

app = Flask(__name__)


# home page
@app.route('/')
def home():
    return 'Hello World!'


# you can directly pass html inside
@app.route('/jazzhands')
def jazzhands():
    return "<h1>Here's some <i>Pizzazz!</i></h1>"


# dynamically generate urls and functionality
@app.route('/twice/<int:x>')  # int says the expected data type
def twice(x):
    output = 2 * x
    return 'Two times {} is {}'.format(x, output)


# ------------------------------------ #
# -------- DATA SCIENCE TIME --------- #
# ------------------------------------ #


@app.route('/titanic/predict', methods=['GET','POST'])
def titanic_predict():
    data = {}
    if request.form:
        # get the input data
        form_data = request.form
        data['form'] = form_data
        predict_class = float(form_data['predict_class'])
        predict_age = float(form_data['predict_age'])
        predict_sibsp = float(form_data['predict_sibsp'])
        predict_parch = float(form_data['predict_parch'])
        predict_fare = float(form_data['predict_fare'])
        predict_sex = form_data['predict_sex']

        # convert the sex from text to binary
        if predict_sex == 'M':
            sex = 0
        else:
            sex = 1
        input_data = np.array([predict_class, predict_age, predict_sibsp, predict_parch, predict_fare, sex])

        # get prediction
        prediction = L1_logistic.predict_proba(input_data.reshape(1, -1))
        prediction = prediction[0][1] # probability of survival
        data['prediction'] = '{:.2f}% Chance of Survival'.format(prediction)
        data['predict_class'] = predict_class
        data['predict_age'] = predict_age
        data['predict_sibsp'] = predict_sibsp
        data['predict_parch'] = predict_parch
        data['predict_fare'] = predict_fare
        data['predict_sex'] = predict_sex
    return render_template('titanic.html', data=data)


if __name__ == '__main__':
    # build a basic model for titanic survival
    titanic_df = pd.read_csv('data/titanic_data.csv')
    titanic_df['sex_binary'] = titanic_df['sex'].map({'female': 1, 'male': 0})
    train_df, test_df = train_test_split(titanic_df)

    features = [u'pclass', u'age', u'sibsp', u'parch', u'fare', u'sex_binary', 'survived']
    train_df = train_df[features].dropna()
    test_df = test_df[features].dropna()

    features.remove('survived')
    X_train = train_df[features]
    y_train = train_df['survived']
    X_test = test_df[features]
    y_test = test_df['survived']

    L1_logistic = LogisticRegression(C=1.0, penalty='l1')
    L1_logistic.fit(X_train, y_train)

    app.run(debug=True)

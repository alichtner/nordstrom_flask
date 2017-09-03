from flask import Flask
from flask import render_template, request
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

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
@app.route('/twice/<int:x>') # int says the expected data type
def twice(x):
    output = 2 * x
    return 'Two times {} is {}'.format(x, output)


# ------------------------------------ #
# -------- DATA SCIENCE TIME --------- #
# ------------------------------------ #


@app.route('/train')
def train_model():
	global model
	model = linear_model.LinearRegression()
	X = np.array([[1, 1, 2, 3, 4, 5, 2, 5]])
	y = np.array([[0, 1, 2, 2, 4, 3, 5, 6]])
	model.fit(X, y)
	message = "<p>Reading in data --</p><p>Training model --</p>"
	return message

@app.route('/predict')
def predict():
    #x = np.array([[float(x)]])
    #print(x)
    prediction = regr.predict(diabetes_X_test)
    return prediction


@app.route('/titanic')
def titanic():
    return render_template('index.html')

@app.route('/titanic/train')
def titanic_train():
    return

@app.route('/titanic/predict',  methods=['POST'])
def titanic_predict():
	result = request.form
	return render_template('titanic.html', result=result)


@app.route('/wine')
def wine():
    return render_template('wine.html')

if __name__ == '__main__':

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print(diabetes_X_test)

    # # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # # The mean squared error
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
    #
    # # Plot outputs
    # plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    # plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    #
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()
    app.run(debug=True)
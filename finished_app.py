from flask import Flask
from flask import render_template, request

application = Flask(__name__)

@application.route('/')
def start():
    return 'Hello World!'

@application.route('/train')
def train_model():
	# read data
	# split data
	# train model
	# get accuracy
	message = "<p>Reading in data --</p><p>Training model --</p><p>Accuracy is 80%--</p>"
	return message
	
@application.route('/predict')
def predict():
	# prediction = model.predict()
	pass
	

@application.route('/titanic')
def titanic():
    return render_template('index.html')

@application.route('/titanic/train')
def titanic_train():
    return

@application.route('/titanic/predict',  methods=['POST'])
def titanic_predict():
	result = request.form
	return render_template('titanic.html', result=result)


@application.route('/wine')
def wine():
    return render_template('wine.html')

if __name__ == '__main__':
    application.debug = True
    application.run()
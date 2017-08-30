# nordstrom_flask
Flask for data science

## python 3

Install python 3
`brew install python3`

Set up the virtual environment
`virtualenv -p python3 venv3`

Activate the virtual environment
`source venv3/bin/activate`

Install flask
`pip3 install flask`

# A Basic Flask App
Import the needed things.
```python
from flask import Flask
application = Flask(__name__)
```

Add your first route. Routes are tell flask what to do for different URLs.
```python
@application.route('/')
def start():
    return 'Hello World!'
```

Initialize your flask app
```python
if __name__ == '__main__':
    application.debug = True
    application.run()
```

Start the application
`python application.py`

Check it out!

[Your App](http://127.0.0.1:5000/)
from flask import Flask

application = Flask(__name__)

@application.route('/')
def start():
    return 'Hello World!'

if __name__ == '__main__':
    application.debug = True
    application.run()
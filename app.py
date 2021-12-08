from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
import store

mail_settings = {
        "MAIL_SERVER":'smtp.gmail.com',
        "MAIL_PORT": 465,
        "MAIL_USE_TLS": False,
        "MAIL_USE_SSL": True,
        "MAIL_USERNAME": store.gmail,
        "MAIL_PASSWORD": store.password
     }

db = SQLAlchemy()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'e591658b7c3d40649bc86aeca3590e99'
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{store.user}:{store.password}@{store.hostname}/dashboard'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.update(mail_settings)
login_manager = LoginManager(app)
login_manager.login_view = 'main.login'
login_manager.login_message_category = 'info'

mail = Mail(app)
db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'main.login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))

from main import main
app.register_blueprint(main)

from dashboard import dashboard
app.register_blueprint(dashboard)


if __name__ == '__main__':
    app.run(debug=True)

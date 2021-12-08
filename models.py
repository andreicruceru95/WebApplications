from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SubmitField, EmailField, SelectField, PasswordField, BooleanField
from wtforms.validators import Length, Email, DataRequired, ValidationError, EqualTo, InputRequired
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask_login import UserMixin, current_user
from app import db, app, login_manager

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(20), nullable=False)
    last_name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(30), nullable=False)
    password = db.Column(db.String(20))
    role = db.Column(db.String(20), nullable=False)
    image = db.Column(db.String(20), default='default.jpg')

    def __init__(self, email, first_name, last_name, password, role, image):
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.role = role
        self.password = password
        self.image = image

    def get_reset_token(self, expires_sec = 3600):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)


class RegistrationForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email(), Length(min=5, max=30)])
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=30)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=30)])
    role = SelectField('Role', validators=[DataRequired()], choices=['Associate Analyst', 'Data Analyst',
                                                                     'Data Scientist', 'Admin'])
    submit = SubmitField('Register User')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email is already in use!')

class LoginForm(FlaskForm):
    email = EmailField('Username',validators=[DataRequired(), Email(), Length(min=5, max=30)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RequestResetForm(FlaskForm):
    email = EmailField('Username',validators=[DataRequired(), Email(), Length(min=5, max=30)])
    submit = SubmitField('Request Password Reset')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account registered with this email!')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6, max=20)])
    password_confirm = PasswordField('Confirm password',
                                     validators=[InputRequired(), Length(min=6, max=20), EqualTo('password')])
    submit = SubmitField('Reset Password')

class UpdateUserInfo(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email(), Length(min=5, max=30)])
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=30)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=30)])
    picture = FileField('Change Profile Picture', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
    submit = SubmitField('Update Info')

    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('This email is already in use. Please use a different email.')


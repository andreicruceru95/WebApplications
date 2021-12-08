import pandas as pd
from flask import Blueprint, render_template, url_for, redirect, flash, request
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user


main = Blueprint('main', __name__)

from app import db
from models import User, RegistrationForm, LoginForm, RequestResetForm, ResetPasswordForm, UpdateUserInfo
import mail

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and (check_password_hash(user.password, form.password.data)):
            login_user(user, remember=form.remember.data)

            next_page = request.args.get('next')
            print(next_page, request.args)
            return redirect(next_page) if next_page else redirect(url_for('dashboard.summary'))
        else:
            flash('Login unsuccessful! Email or password incorrect!', 'danger')
    return render_template('login.html', form=form)


@main.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            flash(f'This email is already used', 'danger')
            return redirect(url_for('main.register'))

        flash(f'Account created for {form.first_name.data} {form.last_name.data}!', 'success')
        new_user = User(email=form.email.data, first_name=form.first_name.data,
                        last_name=form.last_name.data, role=form.role.data, image='default.jpg',
                        password=generate_password_hash('28yr99c8cnu324wcer233rc'))
        db.session.add(new_user)
        db.session.commit()
        mail.send_registration_email(new_user)
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@main.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    form = RequestResetForm()
    if form.validate_on_submit():
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user:
                mail.send_reset_email(user)
                flash('An email has been sent with instructions', 'info')
                return redirect(url_for('main.login'))
    return render_template('reset_request.html', form=form)


@main.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    user = User.verify_reset_token(token)
    if user is None:
        flash('Invalid or expired token', 'warning')
        return redirect(url_for('main.reset_request'))

    form = ResetPasswordForm(request.form)
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, 'sha256')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now ready to login')
        return redirect(url_for('main.login'))

    return render_template('reset_password.html', form=form)


@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


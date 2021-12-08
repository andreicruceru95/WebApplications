from flask_mail import Message
from flask import url_for
from app import mail, app

def send_registration_email(user):
    token = user.get_reset_token(86400) # 24h
    msg = Message(subject="Welcome",
                  sender = ('donotreply', 'Insights App'),
                  recipients=[user.email],
                  html=f"""\
<!Doctype html>
<html lang="en" style:"background-color: #484a49;">
<head>
    
</head>
<body>
    <div style:"background-color: #484a49;">
        <div style="background-color: #484a49; text-align: center;">
        <br><br>
        <h2 style="color:white">Insights App</h2>
        <br><br>     
        </div>
        <div style="background-color: #484a49;">
            <br><br><br><br>
            <h3 style="color:white;">Welcome, {user.first_name},</h3>
            <br>
            <h3 style="color:white";>Thank you for joining the Insights team.</h3>
            <h3 style="color:white";>Please use the link below to reset your password and access our application.</h3>
            <br><br>
            <a href="{url_for('main.reset_token', token=token, _external=True)}" style="color:orange; margin:left: 10%;">Reset my password</a>     
            <br><br>            
            <h3 style="color:white";>You will not be able to access the application without resetting your password. This link is available for the next 24 hours.</h3>
            <h3 style="color:white";>Alternatively, you can request a new reset token by accessing our <a href="{url_for('main.login')}" style="color:orange; margin:left: 10%;">Login</a> page => 'Forgot my Password'.</h3>
            <br><br><br><br><br><br><br><br><br><br>
        </div>
        <div style="background-color: #484a49; text-align: center;">
        <h3 style="color:white">Insights App</h3>
        <p style="color:white">Copyright@2021 #AndreiCruceru #CO550: Web Application for Data Science</p>
        <br><br>
        </div>
    </div>
</body>
</html>
"""
                  )
    mail.send(msg)

def send_reset_email(user):
    token = user.get_reset_token(21600) # 6h
    msg = Message(
        subject='Password Reset Request',
        sender=('donotreply', 'Insights App'),
        recipients=[user.email],
        html=f"""\
<!Doctype html>
<html lang="en" style:"background-color: #484a49;">
<head>
    
</head>
<body>
    <div style:"background-color: #484a49;">
        <div style="background-color: #484a49; text-align: center;">
        <br><br>
        <h2 style="color:white">Insights App</h2>
        <br><br>     
        </div>
        <div style="background-color: #484a49;">
            <br><br><br><br>
            <h3 style="color:white;">Hello {user.first_name},</h3>
            <br>
            <h3 style="color:white";>You recently requested to reset your password for your Insights account. Use the button below to change it.</h3>
            <br><br>
            <a href="{url_for('main.reset_token', token=token, _external=True)}" style="color:orange; margin:left: 10%;">Change my password</a>     
            <br><br>       
            <h3 style="color:white";>If you did not make this request, please ignore this email.</h3>
            <h3 style="color:white";>Your password won't change until you access the link above and create a new one. The link is available for the next 6 hours.</h3>
            <br><br><br><br><br><br><br><br><br><br>
        </div>
        <div style="background-color: #484a49; text-align: center;">
        <h3 style="color:white">Insights App</h3>
        <p style="color:white">Copyright@2021 #AndreiCruceru #CO550: Web Application for Data Science</p>
        <br><br>
        </div>
    </div>
</body>
</html>
"""
    )
    mail.send(msg)
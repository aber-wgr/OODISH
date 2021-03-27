import smtplib, ssl
from email.mime.text import MIMEText

def SendEmail(subject='Test Email', text='This is a test', recipient='target@email.com'):
    port = 587
    host = 'host.smtp.server'
    sender = "my@email.com"
    username = "myemailusername"
    password = "myemailpassword"


    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = "my@email.com"
    msg['To'] = recipient
    receivers = [recipient]

    with smtplib.SMTP(host,port) as server:
            server.starttls()
            server.login(username,password)
            server.sendmail(sender, receivers[0], msg.as_string())
            print("email sent")

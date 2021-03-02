import smtplib, ssl
from email.mime.text import MIMEText

def SendEmail(subject='Test Email', text='This is a test', recipient='wgr@aber.ac.uk'):
    sender = 'wgr@aber.ac.uk'
    port = 587
    host = 'relay.plus.net'
    sender = "will@chalicier.plus.com"
    username = "chalicier"
    password = "oojaxooz"


    msg = MIMEText(text)
    msg['Subject'] = subject
    msg['From'] = "will@chalicier.plus.com"
    msg['To'] = recipient
    receivers = [recipient]

    with smtplib.SMTP(host,port) as server:
            server.starttls()
            server.login(username,password)
            server.sendmail(sender, receivers[0], msg.as_string())
            print("email sent")

import smtplib
import socket
from email.header import Header
from email.mime.text import MIMEText


class Gmail:
    def __init__(self):
        self._user = "UFOdestiny@gmail.com"
        self._smtpcode = "bbsigfeiispxrnjk"
        self._smtp = self._connect(self._user, self._smtpcode)

        self._sender = "UFOdestiny@gmail.com"
        self._reciver = "1976438440@qq.com"

    def _connect(self, user, smtpcode):
        try:
            smtp = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465)
            smtp.login(user, smtpcode)
            return smtp
        except socket.gaierror:
            print("HOST ERROR")

    def _message(self, msg, from_, to, subject):
        msg = MIMEText(msg, 'plain', 'utf-8')
        msg['From'] = Header(from_, 'utf-8')
        msg['To'] = Header(to, 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        return msg.as_string()

    def send(self, msg="MESSAGE IS EMPTY", sender=None, to=None, subject=None):
        from_ = sender if sender else self._sender
        to = to if to else self._reciver
        subject = subject if subject else msg
        msg = self._message(msg=msg, from_=from_, to=to, subject=subject)
        try:
            self._smtp.sendmail(from_, to, msg)
        except smtplib.SMTPException:
            print("SEND FAILURE")


if __name__ == "__main__":
    g = Gmail()
    g.send(msg="The Task is Done!")

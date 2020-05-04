import logging
from logging.handlers import BufferingHandler
import sys
import datetime
import smtplib
import os
import string


class BufferingSMTPHandler(BufferingHandler):

	def __init__(self, mailhost, fromaddr, toaddrs, subject, capacity, credentials):

		BufferingHandler.__init__(self, capacity)
		self.mailhost = mailhost[0]
		self.mailport = mailhost[1]
		self.fromaddr = fromaddr
		self.toaddrs = toaddrs
		self.subject = subject
		self.username = credentials[0]
		self.password = credentials[1]
		self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

	def flush(self):
		if len(self.buffer) > 0:
			try:
				port = self.mailport
				if not port:
					port = smtplib.SMTP_PORT
				smtp = smtplib.SMTP_SSL(self.mailhost, port)
				msg = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (
				self.fromaddr, string.join(self.toaddrs, ","), self.subject)
				for record in self.buffer:
					s = self.format(record)
					msg = msg + s + "\r\n"
				if self.username:
					smtp.login(self.username, self.password)
				smtp.sendmail(self.fromaddr, self.toaddrs, msg)
				smtp.quit()
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				self.handleError(self.buffer[0])  # no particular record
			self.buffer = []


def get_smtp_handler(subject_email='', capacity=1):
	smtp_handler = BufferingSMTPHandler(
		mailhost=('smtp.gmail.com', '465'),
		fromaddr='yoannp@adikteev.com',
		toaddrs=['yoannp@adikteev.com'],
		subject=subject_email,
		capacity=capacity,
		credentials=(
			os.environ.get("SMTP_CREDENTIALS_EMAIL"),
			os.environ.get("SMTP_CREDENTIALS_PASSWORD")))
	smtp_handler.setLevel(logging.ERROR)
	return smtp_handler


def get_console_handler():
	formatter_console_handler = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter_console_handler)
	return console_handler


def get_file_handler(
		file_handler_name='temp.log'):
	formatter_file_handler = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
	file_handler = logging.handlers.TimedRotatingFileHandler(
		file_handler_name, when='midnight')
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(formatter_file_handler)
	return file_handler


def get_logger(
		logger_name='__name__',
		file_handler_name='temp.log',
		subject_email='',
		capacity=1):
	file_handler_name = 'data/log/' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '__' + file_handler_name
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(get_console_handler())
	logger.addHandler(get_file_handler(file_handler_name=file_handler_name))
	logger.addHandler(get_smtp_handler(
		subject_email=subject_email,
		capacity=capacity))
	return logger

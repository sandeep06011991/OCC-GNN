import logging
logging.basicConfig(filename='main.log',
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

class LogFile():

    def __init__(self, actor, id):
        self.actor = actor
        self.id = id

    def log(self, msg):
        logging.debug(" {} {} {}".format(self.actor, self.id,msg))

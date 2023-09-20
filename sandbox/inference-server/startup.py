# Setup Logging
# --------------------------------------------------------------------------------
import logging, sys
class DefaultLogger(logging.Logger):
    logging.addLevelName(logging.INFO,      "\x1b[38;20mINFO \x1b[0m")
    logging.addLevelName(logging.WARNING,   "\x1b[33;20mWARN \x1b[0m")
    logging.addLevelName(logging.ERROR,     "\x1b[31;20mERROR\x1b[0m")
    def __init__(self, name):
        super().__init__(name)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)-12s %(message)s", "%Y-%m-%d %H:%M:%S.%03d")
        )
        self.addHandler(self.handler)
    def warn(self, message):
        self.warning(message)

log = DefaultLogger('startup')

# Update packages
# --------------------------------------------------------------------------------
import subprocess
log.info("pip install packages...")
try:
    subprocess.run(["pip", "install", "-r", "requirements.txt"], 
        stdout=sys.stdout, stderr=sys.stderr)
except:
    pass


"""
import logging
class CustomLogger(logging.Logger):
    class ColorFormatter(logging.Formatter):
        format = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)-12s %(message)s"
        FORMATS = {
            logging.INFO:       f"\x1b[38;20m{format}\x1b[0m",
            logging.WARNING:    f"\x1b[33;20m{format}\x1b[0m",
            logging.ERROR:      f"\x1b[31;20m{format}\x1b[0m",
        }
        
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    logging.addLevelName(logging.WARNING, "WARN")

    def __init__(self, name):
        super().__init__(name)
        self.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.ColorFormatter())
        self.addHandler(console_handler)

# Create a logger instance using the CustomLogger class
log = CustomLogger("My_app")

log.info("pip install packages...")
log.warning("pip install packages...")
log.error("pip install packages...")
"""
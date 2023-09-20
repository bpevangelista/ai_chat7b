import logging
class DefaultLogger(logging.Logger):
    logging.addLevelName(logging.INFO,      "\x1b[38;20mINFO \x1b[0m")
    logging.addLevelName(logging.WARNING,   "\x1b[33;20mWARN \x1b[0m")
    logging.addLevelName(logging.ERROR,     "\x1b[31;20mERROR\x1b[0m")
    def __init__(self, name):
        super().__init__(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)-12s %(message)s"))
        self.addHandler(handler)
    def warn(self, message):
        self.warning(message)


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
__version__ = '0.1.0'
__author__ = 'Bruno Evangelista'
__license__ = 'Private'

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
"""
import subprocess
log.info("pip install packages...")
try:
    subprocess.run(["pip", "install", "-r", "requirements.txt"], 
        stdout=sys.stdout, stderr=sys.stderr)
except:
    pass
"""

from .inference import InferenceEngine
from .models import InferenceModel
from .personas import ChatPersonas
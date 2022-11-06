__version__ = '0.3.0'
__author__ = 'Oleg Kostromin, Marco Pasini, Iryna Kondrashchenko'

from faulthandler import disable

from falcon.main import initialize, AutoML
from falcon.utils import disable_warnings

disable_warnings()
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sp


#import other files
import readinput
import output
import florianfit
import classes

readinput.param['filename']='test1_150nm'
output.output(readinput.param)
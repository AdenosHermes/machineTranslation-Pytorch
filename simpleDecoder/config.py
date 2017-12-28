from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import time
import math
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 40
use_cuda = torch.cuda.is_available()
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

teacher_forcing_ratio = 0.5

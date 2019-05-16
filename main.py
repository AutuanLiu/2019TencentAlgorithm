import pandas as pd
import numpy as np
from models.utils import *
from deepctr.models import xDeepFM
from sklearn.model_selection import train_test_split

path = './data/'
train_data = pd.read_csv(f'{path}train_set_final.csv', low_memory=False, encoding='utf-8')
test_data = pd.read_csv(f'{path}test_set_final.csv', low_memory=False, encoding='utf-8')

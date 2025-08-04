from os import environ

DATA_DIR = environ['VSC_DATA'] if 'VSC_DATA' in environ else "/esat/stadiustempdatasets/jquan" 
EAVG_DEFAULT_EXP = 0.3
EPOCHS_AT_DEPTH = [50, 55, 60]
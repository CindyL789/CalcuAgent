# Add your **utility data** code below:

from pathlib import Path

_PATH_2APP = Path(__file__).resolve().parents[3]
PATH_2NEWSDATA = _PATH_2APP / 'data'
PATH_2KAGGLE   = _PATH_2APP / 'kaggle'
PATH_2MODELS   = _PATH_2APP / 'models'
PATH_2REPORTS  = _PATH_2APP / 'reports'
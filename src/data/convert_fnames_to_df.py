import sys
from pathlib import Path
import pandas as pd

BASE_PATH = Path.cwd()
sys.path.append(str(BASE_PATH / "src" / "data"))  # for clean_csv and train_test_split
# from utils import *

print(list(Path.glob(BASE_PATH / "data" / "raw" / "recordings", "*")))

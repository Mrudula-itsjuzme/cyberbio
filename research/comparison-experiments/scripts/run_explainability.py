import pandas as pd
import numpy as np
import os
import sys

# Add materials_adv path to import model logic
sys.path.append(os.path.abspath("materials-adversarial/src"))
from materials_adv.data.tokenizer import tokenize

# We will mock the model prediction based on the previous drift values, but the attribution occlusion requires running the model.
# Since we might not have the model loaded easily here without the full PyTorch stack, let's see if we can load the model.

import os
import urllib.request

# Download the resnet_model.py file to the current library directory
dir_path = os.path.dirname(os.path.realpath(__file__))
_MODEL_PY_URL = 'https://raw.githubusercontent.com/tensorflow/models/v1.4.0/official/resnet/resnet_model.py'
urllib.request.urlretrieve(_MODEL_PY_URL, 'resnet_model.py')
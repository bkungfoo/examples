import argparse
import os
from subprocess import call
import sys
import urllib.request

REMOTE_FILE = 'http://download.tensorflow.org/models/official/resnet50_2017_11_30.tar.gz'

def dl_progress(count, blockSize, totalSize):
  '''Displays percentage of file complete'''
  percent = int(count * blockSize * 100 / totalSize)
  sys.stdout.write('\rdownloading...%d%%' % percent)
  sys.stdout.flush()


parser = argparse.ArgumentParser()
parser.add_argument(
  '--src',
  '-s',
  required=True,
  help='Source URL for checkpoint file'
)
parser.add_argument(
  '--dst',
  '-d',
  required=True,
  help='Where to store checkpoint file locally'
)
args = parser.parse_args()

src = args.src
model_dir = args.dst
filename = os.path.basename(src)
local_file = os.path.join(model_dir, filename)

call(['mkdir', '-p', model_dir])

urllib.request.urlretrieve(
  src,
  local_file,
  reporthook=dl_progress
)

print('\n')
call(["tar", "-zxvf", local_file, '-C', model_dir])

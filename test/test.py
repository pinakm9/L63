# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath()))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')
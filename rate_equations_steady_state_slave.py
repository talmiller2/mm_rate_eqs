import argparse
import ast
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--settings', help='settings for the rate equations solver algorithm',
                    type=str, required=True)
args = parser.parse_args()

settings = ast.literal_eval(args.settings)

sys.path.append(settings['code_dir'])
from relaxation_algorithm_functions import find_rate_equations_steady_state

state = find_rate_equations_steady_state(settings)

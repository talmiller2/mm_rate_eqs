import argparse
import ast

from mm_rate_eqs.relaxation_algorithm_functions import find_rate_equations_steady_state

parser = argparse.ArgumentParser()
parser.add_argument('--settings', help='settings for the rate equations solver algorithm',
                    type=str, required=True)
args = parser.parse_args()

settings = ast.literal_eval(args.settings)

state = find_rate_equations_steady_state(settings)

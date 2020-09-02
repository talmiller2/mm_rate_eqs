import os


def get_script_rate_eqs_slave():
    script_path = os.path.dirname(os.path.abspath(__file__)) + '/rate_eqs_slave.py'
    return script_path

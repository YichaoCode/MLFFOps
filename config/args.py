import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Deep Potential Generator")
    parser.add_argument("-p", "--param", type=str, default="input/par.json", help="The parameter file in JSON format")
    parser.add_argument("-m", "--machine", type=str, default="input/machine.json", help="The machine configuration file in JSON format")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-v", "--version", action="version", version="v1.0.0")
    parser.add_argument("--clean", action="store_true", help="Clean up files and folders before running")
    parser.add_argument("-r", "--restart", type=str, default=None,
                        help="Restart from a specific task, format: ITER.TASK, e.g., 000000.06")
    return parser.parse_args()
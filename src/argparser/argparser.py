"""
Argparser for the project
"""

import argparse

parser = argparse.ArgumentParser(
    description="Variational Autoencoder for generating flowers images."
)

parser.add_argument("-a", "--arg1", type=int, help="Description of arg1")
parser.add_argument("-b", "--arg2", type=str, help="Description of arg2")
parser.add_argument("-c", "--arg3", type=float, help="Description of arg3")

args = parser.parse_args()

print(args.arg1)
print(args.arg2)
print(args.arg3)

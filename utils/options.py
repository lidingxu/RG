import argparse
import os
"""
args
"""

parser = argparse.ArgumentParser(description='RG')

parser.add_argument(
    '--results_dir',
    metavar='RESULTS_DIR',
    default='./results',
    help='results dir')

parser.add_argument(
    '--dataset',
    default='Ionosphere',
    type=str,
    help='dataset, default: Ionosphere')

parser.add_argument(
    '--model',
    '-a',
    metavar='MODEL',
    default='convex',
    help='model architecture')

parser.add_argument(
    '--seed', 
    default=0, 
    type=int, 
    help='random seed, set to 0 to disable')


parser.add_argument(
    '--silent', 
    default=True, 
    type=bool, 
    help='detailed log output, set to True to disable')



args = parser.parse_args()
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
    '--logs_dir',
    metavar='LOG_DIR',
    default='./logs',
    help='logs dir')

parser.add_argument(
    '--dataset',
    default='Ionosphere',
    type=str,
    help='dataset, default: Ionosphere')

parser.add_argument(
    '--model',
    '-m',
    metavar='MODEL',
    default='convex',
    help='model architecture')
    
parser.add_argument(
    '--round',
    '-r',
    default=0, 
    type=int, 
    help='Int rounds')

parser.add_argument(
    '--seed', 
    default=0, 
    type=int, 
    help='random seed, set to 0 to disable')


parser.add_argument(
    '--verbose', 
    default=0, 
    type=int, 
    help='detailed log output level, set to 0 to disable')

parser.add_argument(
    '--val',
    default=0, 
    type=float, 
    help='percentage of validation data set, 0 for disable')

parser.add_argument(
    '--lambda0',
    default=0.001, 
    type=float, 
    help='lambda0')

parser.add_argument(
    '--lambda1',
    default=0.001, 
    type=float, 
    help='lambda1')

parser.add_argument(
    '--lambda2',
    default=0.00, 
    type=float, 
    help='lambda2')

parser.add_argument(
    '--filter_eps',
    default=0.00, 
    type=float, 
    help='filter threshold')


args = parser.parse_args()

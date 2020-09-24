# Python-specific imports
import numpy as np
import argparse
import time
import pandas as pd

# Scikit-learn imports
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import collections


def get_args():
    '''
    get_args() method is to parse the arguments from the command line.
    It is called by initialization() method

    :return: Parsed arguments
    '''
    parser = argparse.ArgumentParser(
        description="Experiments with several PUF architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--puf', metavar="PUF", nargs='?',
                        default="DAPUF", help='Type of arbiter (xor, DAPUF)')
    parser.add_argument('--runs', metavar="R", type=int,
                        default=10, help='Number of runs')
    parser.add_argument('--num_neurons', metavar='num_neurons', type=int,
                        default=32, help='')

    return parser.parse_args()


def transform_features(C):
    parity = 2 * C - 1
    parity = np.cumprod(parity, axis=1, dtype=np.int8)
    return parity

def model_training(args, model):
    metrics = {m: [] for m in ['acc', 'time']}

    # Start running the experiment for the number of runs that indicated in the command line argument (--runs)
    for run_id in range(args.runs):
        print("\n___--|[ Run %d ]|--___ \n" % (run_id + 1))

        """
        STEP1: Generate CRPs
        """
        print('[1] Extract CRPs ...')
        df = pd.read_csv('3Double64bit_1M.txt', sep=";", header=None, names=["C", "r"],low_memory=False)

        C = df['C']
        r = df['r']
        C = [list(C[i]) for i in range(len(C))]
        C = np.asarray(C, dtype=np.int8)
        r = np.asarray(r, dtype=np.int8)
        print(collections.Counter(r))


        """
        STEP2: Transform the challenge set to parity features using cumulative product
        """
        print('[2] Transform CRPs as parity features ...')

        C = transform_features(C)

        """
        STEP3: Start model training
        """
        print('[3] Model Training in progress ...')
        tr_C, ts_C, tr_r, ts_r = train_test_split(C, r, train_size=.85)

        start = time.time()

        print("\t>> Train Accuracy= %.4f%%" % model.fit(tr_C, tr_r).score(tr_C, tr_r))

        end = time.time()

        # Predict and calculate the accuracy scores
        prd_r = model.predict(ts_C)
        test_acc = accuracy_score(ts_r, prd_r) * 100.

        print("\t>> Test Accuracy= %.2f%%" % test_acc)

        train_time = end - start

        # Store the accuracy results
        metrics['acc'].append(test_acc)
        metrics['time'].append(train_time)

        # Get the average accuracy so far
        avg_acc = np.mean(metrics['acc'])
        avg_time = np.mean(metrics['time'])
        best_acc = np.max(metrics['acc'])

        """
        STEP4: Print the results
        """
        print("[4] RESULTS for Run (%s):" % (run_id + 1))
        print("************************************")
        print("\t>> Average Test Accuracy= %.2f%%" % avg_acc)
        print("\t>>  Best Test Accuracy= %.2f%% " % best_acc)
        print("\t>> Average Training time= %.3f sec" % avg_time)
        print("hidden layer size  = (%s, %s, %s)" % (args.num_neurons, args.num_neurons, args.num_neurons))
        print("----------------------------------------")


def initialization():
    '''
    This method is to initialize the desired PUF

    :return: args, model
    '''

    model = None  # Holds an object form the models module
    args = get_args()

    # initialize based on the puf type

    if args.puf == 'DAPUF':
        model = MLPClassifier(hidden_layer_sizes=(args.num_neurons, args.num_neurons, args.num_neurons), max_iter=1000,
                              learning_rate_init=1e-3,
                              learning_rate='adaptive', batch_size=100,
                              solver='adam', activation='relu', early_stopping=True)

    return args, model


"""
*************************************************************************************
*                                   Main Function                                   *
*************************************************************************************
"""

if __name__ == "__main__":
    '''
    PART_1:
    Start initializing the simulation
    '''
    args, model = initialization()

    '''
    PART_2:
    Model Training
    '''
    model_training(args, model)

#!/usr/bin/env python3

import argparse
import sklearn.model_selection
import pandas as pd


def read_json_lines(file_path):
    # Initialize an empty list to store the parsed JSON objects
    data = []
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON object from each line
            json_obj = pd.read_json(line, typ='series', convert_dates=False)
            data.append(json_obj)
    
    # Create a DataFrame from the parsed JSON objects
    df = pd.DataFrame(data)
    
    return df


def main(arguments):
    reviews = read_json_lines("Movies_and_TV_trimmed.json")
    train, test = sklearn.model_selection.train_test_split(reviews, test_size = 0.2)
    # print("Train is: ", train)
    # print("Test is: ", test)

def _load_args():
    parser = argparse.ArgumentParser(description='Generate a recommendation list consisting of 10 items for each user in the testing set.')

    parser.add_argument('--param', dest='param', required=False,
                        action='store', type=str, help='adjustment for hyperparameter.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args())

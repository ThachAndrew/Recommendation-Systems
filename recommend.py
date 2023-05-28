#!/usr/bin/env python3

import argparse
import multiprocessing

import lenskit.algorithms.user_knn
import lenskit.algorithms.basic
import lenskit.algorithms.ranking
import sklearn.model_selection
import pandas as pd


# Deprecated, since it's slow.
def read_json_lines(file_path):
    # Initialize an empty list to store the parsed JSON objects
    data = []
    
    # Read the file line by line
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            print("On line: ", i)
            # Parse the JSON object from each line
            json_obj = pd.read_json(line, typ='series', convert_dates=False)
            data.append(json_obj)
            i += 1
    
    # Create a DataFrame from the parsed JSON objects
    df = pd.DataFrame(data)
    
    return df

def parse_json_line(line):
    return pd.read_json(line, typ='series', convert_dates=False)

# Define the worker function for parallel processing
def worker(line):
    return parse_json_line(line)

def read_json_lines_parallel(filename, ignore_keys=None):
    # Initialize an empty list to store the parsed JSON objects
    data = []

    # Read the file line by line
    with open(filename, 'r') as file:
        # Create a multiprocessing Pool
        pool = multiprocessing.Pool()

        # Map the worker function to process the lines in parallel
        results = pool.map(worker, file)

        # Close the multiprocessing Pool
        pool.close()

        # Combine the results
        data.extend(results)

    # Create a DataFrame from the parsed JSON objects
    df = pd.DataFrame(data)

    # Exclude keys specified in ignore_keys list
    if ignore_keys:
        df = df.drop(ignore_keys, axis=1, errors='ignore')

    return df

def load_data(filename):
    # TODO: Make a condition to check if the pickle is there.
    # ratings = read_pickle("serialized_ratings")

    ratings = read_json_lines_parallel("Movies_and_TV_trimmed.json")
    ratings = ratings.rename(columns = {"reviewerID": "user", "asin": "item", "overall": "rating"})
    ratings.to_pickle("serialized_ratings")

    return ratings


def main(arguments):
    print("Loading data...")
    ratings = load_data("Movies_and_TV_trimmed.json")

    train, test = sklearn.model_selection.train_test_split(ratings, test_size = 0.2)

    print("Fitting User User...")
    classifier = lenskit.algorithms.user_knn.UserUser(3)
    classifier.fit(train)

    print("Fitting Candidate Selector...")
    unrated_items = lenskit.algorithms.basic.UnratedItemCandidateSelector()
    unrated_items.fit(ratings)

    topn = lenskit.algorithms.ranking.TopN(classifier, unrated_items)

    print(topn.recommend("A2PANT8U0OJNT4"))

def _load_args():
    parser = argparse.ArgumentParser(description='Generate a recommendation list consisting of 10 items for each user in the testing set.')

    parser.add_argument('--param', dest='param', required=False,
                        action='store', type=str, help='adjustment for hyperparameter.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args())

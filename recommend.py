#!/usr/bin/env python3

import argparse

import lenskit.algorithms.user_knn
import lenskit.algorithms.basic
import lenskit.algorithms.ranking
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

def load_data(filename):
    # TODO: Make a condition to check if the pickle is there.
    # ratings = read_pickle("serialized_ratings")

    ratings = read_json_lines("Movies_and_TV.json")
    ratings = ratings.rename(columns = {"reviewerID": "user", "asin": "item", "overall": "rating"})
    ratings.to_pickle("serialized_ratings")

    return ratings


def main(arguments):
    ratings = load_data("Movies_and_TV_trimmed.json")

    train, test = sklearn.model_selection.train_test_split(ratings, test_size = 0.2)

    classifier = lenskit.algorithms.user_knn.UserUser(3)
    classifier.fit(train)

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

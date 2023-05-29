#!/usr/bin/env python3

import argparse
import multiprocessing

# Rating Prediction
import lenskit.algorithms.user_knn
import lenskit.algorithms.item_knn
import lenskit.algorithms.mf_common
import lenskit.algorithms.als
import lenskit.algorithms.basic

# Ranking Eval
import lenskit.algorithms.ranking
import lenskit.topn
import lenskit.metrics.topn
import lenskit.metrics.predict

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
    ratings = pd.read_pickle("serialized_ratings")
    # ratings = pd.read_pickle("serialized_ratings_trimmed")

    # ratings = read_json_lines_parallel("Movies_and_TV.json")
    # ratings = ratings.rename(columns = {"reviewerID": "user", "asin": "item", "overall": "rating"})
    # ratings.to_pickle("serialized_ratings")

    return ratings

def predict_ratings_for_users(classifier, test):
    predictions = []
    for index, review in test.iterrows():
        # print("Looking at test review: \n", review)
        predictions.append(classifier.predict_for_user(review["user"], [review["item"]])[0])
        
    return predictions


def main(arguments):
    print("Loading data...")
    ratings = load_data("Movies_and_TV.json")

    train, test = sklearn.model_selection.train_test_split(ratings, test_size = 0.2)
    # print("test_columns are: ", test.columns)

    # print("Fitting User User...")
    # classifier = lenskit.algorithms.user_knn.UserUser(30)
    print("Fitting Item Item...")
    classifier = lenskit.algorithms.item_knn.ItemItem(30)

    # print("Matrix Factorizing...")
    # classifier = lenskit.algorithms.als.BiasedMF(10)

    classifier.fit(train)

    print("Predicting ratings for test users")
    test_ratings_pred = predict_ratings_for_users(classifier, test)
    test_ratings_ground = list(ratings['rating'])

    # print("test_ratings_pred are: ", test_ratings_pred)
    # print("test_ratings_ground are: ", test_ratings_ground)
    rmse = lenskit.metrics.predict.rmse(test_ratings_pred, test_ratings_ground)
    mae = lenskit.metrics.predict.mae(test_ratings_pred, test_ratings_ground)
    print("rmse is: ", rmse)
    print("mae is: ", mae)


    # print("Fitting Candidate Selector...")
    # candidate_selector = lenskit.algorithms.basic.UnratedItemCandidateSelector()
    # candidate_selector.fit(ratings)

    # topn = lenskit.algorithms.ranking.TopN(classifier, candidate_selector)

    # analyzer = lenskit.topn.RecListAnalysis()
    # analyzer.add_metric(lenskit.metrics.topn.precision, k = 10)
    # analyzer.add_metric(lenskit.metrics.topn.recall, k = 10)

    # print("Test users: \n", set(test['user']))

    # print("candidate_selector is: \n", candidate_selector)
    # print("Recommending items for user... \n")

    # rec_frame = topn.recommend("A1Z0Y3THM81OY2", n=10)
    # rec_frame["user"] = "A1Z0Y3THM81OY2"
    # scores = analyzer.compute(rec_frame, test) 
    # print(scores)

def _load_args():
    parser = argparse.ArgumentParser(description='Generate a recommendation list consisting of 10 items for each user in the testing set.')

    parser.add_argument('--param', dest='param', required=False,
                        action='store', type=str, help='adjustment for hyperparameter.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args())

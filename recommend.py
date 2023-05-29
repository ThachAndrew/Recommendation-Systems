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
    ratings = pd.read_pickle("serialized_ratings")

    # If not serialized, then uncomment these to re-serialize them.
    # ratings = read_json_lines_parallel("Movies_and_TV.json")
    # ratings = ratings.rename(columns = {"reviewerID": "user", "asin": "item", "overall": "rating"})
    # ratings.to_pickle("serialized_ratings")

    return ratings

def predict_ratings_for_users(classifier, test):
    predictions = []
    for index, review in test.iterrows():
        predictions.append(classifier.predict_for_user(review["user"], [review["item"]])[0])
        
    return predictions

def recommend_for_all_users(classifier, users):
    whole_rec_frame = pd.DataFrame()
    for user in users:
        user_rec_frame = classifier.recommend(user, n=10)
        whole_rec_frame = pd.concat([user_rec_frame, whole_rec_frame])

    return whole_rec_frame


def main(arguments):
    print("Loading data...")
    ratings = load_data("Movies_and_TV.json")

    train, test = sklearn.model_selection.train_test_split(ratings, test_size = 0.2)

    ########################################
    ## User based Collaborative Filtering ##
    ########################################
    # print("Fitting User User...")
    # classifier = lenskit.algorithms.user_knn.UserUser(30)

    ########################################
    ## Item based Collaborative Filtering ##
    ########################################
    # print("Fitting Item Item...")
    # classifier = lenskit.algorithms.item_knn.ItemItem(30)

    ##########################
    ## Matrix Factorization ##
    ##########################
    print("Matrix Factorizing...")
    classifier = lenskit.algorithms.als.BiasedMF(30)

    classifier.fit(train)

    #######################
    ## Rating Prediction ##
    #######################
    # print("Predicting ratings for test users")
    test_ratings_pred = predict_ratings_for_users(classifier, test)
    test_ratings_ground = list(ratings['rating'])

    rmse = lenskit.metrics.predict.rmse(test_ratings_pred, test_ratings_ground)
    mae = lenskit.metrics.predict.mae(test_ratings_pred, test_ratings_ground)
    print("rmse is: ", rmse)
    print("mae is: ", mae)


    ###########################
    ## Ranking Unrated Items ##
    ###########################
    print("Fitting Candidate Selector...")
    candidate_selector = lenskit.algorithms.basic.UnratedItemCandidateSelector()
    candidate_selector.fit(ratings)

    topn = lenskit.algorithms.ranking.TopN(classifier, candidate_selector)


    print("Generating top 10 for each user")
    recs = recommend_for_all_users(topn, test['user'])
    recs.to_pickle("serialized_recs")

    analyzer = lenskit.topn.RecListAnalysis()
    analyzer.add_metric(lenskit.metrics.topn.precision, k = 10)
    analyzer.add_metric(lenskit.metrics.topn.recall, k = 10)

    scores = analyzer.compute(rec_frame, test) 
    print(scores)

def _load_args():
    parser = argparse.ArgumentParser(description='Generate a recommendation list consisting of 10 items for each user in the testing set.')

    parser.add_argument('--param', dest='param', required=False,
                        action='store', type=str, help='adjustment for hyperparameter.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args())

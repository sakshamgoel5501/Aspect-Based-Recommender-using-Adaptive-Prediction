from pandas import DataFrame
import pandas as pd

predicted_item_id = "B004MWZLYC"
predicted_rating = 2.0

expected_item_id = "B000IVUL64"
expected_rating = 4.5

item_df = pd.read_csv('out.csv',sep='\t', index_col=None)
# item_df.columns = ['MovieID', 'Name','Rating','Genre','Details']
# item_df = item_df.astype({'Rating':'float','Genre':'string','MovieID':'string'})
# item_df.dropna(how='any')
item_df = item_df.dropna()

print(len(item_df))


item_df.head()
print(item_df[item_df['Genre'].str.contains('Suspense')][:3])


def matching_actuals_or_recommendations(predicted_item_id, predicted_rating, item_df):
    shortlisted_item_df = pd.DataFrame(columns=item_df.columns, index=None)
    # predicted_movie_features = item_df[item_df['MovieID']==predicted_item_id]['Genre'].tolist()
    # # print(predicted_movie_features)

    # for pred_mov_feature in predicted_movie_features[0].split(","):
    #   # print(pred_mov_feature)
    #   items_to_consider_df = item_df[item_df['Genre'].str.contains(pred_mov_feature, case=False)]
    #   # print(len(items_to_consider_df))
    #   items_to_consider_df = items_to_consider_df.drop_duplicates(subset=['Name','Rating'])
    #   shortlisted_item_df = shortlisted_item_df.append(items_to_consider_df,ignore_index=True)
    #   shortlisted_item_df = shortlisted_item_df.drop_duplicates(subset=['Name','Rating'])

    # shortlisted_item_df = shortlisted_item_df[shortlisted_item_df['Rating']>=predicted_rating]
    shortlisted_item_df = item_df[item_df['Rating'] >= predicted_rating].drop_duplicates(subset=['Rating'])

    return shortlisted_item_df[['MovieID', 'Rating']]


matching_recommendations = matching_actuals_or_recommendations(predicted_item_id,predicted_rating,item_df)

# for recommendation in matching_recommendations:
#   print(recommendation)

predictions_df = pd.DataFrame(matching_recommendations)

# Convert rating column to float
predictions_df['Rating'] = predictions_df['Rating'].astype(float)

print(len(predictions_df))


matching_actuals = matching_actuals_or_recommendations(expected_item_id,expected_rating,item_df)

actuals_df = pd.DataFrame(matching_actuals)
# Convert rating column to float
actuals_df['Rating'] = actuals_df['Rating'].astype(float)

print(len(actuals_df))


def precision_recall_f1_scores(predictions_df, actuals_df, k, rating_threshold):
    sorted_predictions_df = predictions_df.sort_values(by='Rating', ascending=False)
    sorted_actuals_df = actuals_df.sort_values(by='Rating', ascending=False)

    # Number of relevant items
    # n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
    num_rel_items = len(sorted_actuals_df[sorted_actuals_df['Rating'] >= rating_threshold])

    print("Num of relevant items {}".format(num_rel_items))

    # Number of recommended items in top k
    # n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
    num_sorted_predictions = len(sorted_predictions_df)
    num_rec_k_items = len(sorted_predictions_df[:k]) if num_sorted_predictions >= k else num_sorted_predictions

    print("Num of recommended k items {}".format(num_rec_k_items))

    # Number of relevant and recommended items in top k
    num_rel_and_rec_k_items = len(sorted_predictions_df[sorted_predictions_df['Rating'] >= rating_threshold][
                                  :k]) if num_rec_k_items >= k else len(
        sorted_predictions_df[sorted_predictions_df['Rating'] >= rating_threshold])

    print("Num of relevant and recommended k items {}".format(num_rel_and_rec_k_items))

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    precision = num_rel_and_rec_k_items / num_rec_k_items if num_rec_k_items != 0 else 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.

    recall = num_rel_and_rec_k_items / num_rel_items if num_rel_items != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, sorted_predictions_df[:k]


k = 3
precision, recall, f1, predictions_k = precision_recall_f1_scores(predictions_df, actuals_df, k, rating_threshold=expected_rating)

print("Precision@{0}: {1}, Recall@{0}: {2} and F1@{0} Score: {3}".format(k,precision,recall,f1))
print("Expected Rating Threshold {0} and Predicted Ratings {1}".format(expected_rating, predictions_k))




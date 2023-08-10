from pandas import DataFrame
import pandas as pd

predicted_item_id = "B004MWZLYC"
predicted_rating = 4.3

expected_item_id = "B000IVUL64"
expected_rating = 4.8


item_df=pd.read_csv('out.csv',sep='\t', index_col=None)
item_df.columns = ['MovieID', 'Name','Rating','Genre','Details']
# item_df = item_df.astype({'Rating':'float','Genre':'string','MovieID':'string'})
# item_df.dropna(how='any')
item_df = item_df.dropna()


item_df.head()
print(item_df[item_df['Genre'].str.contains('Suspense')][:3])


def recommendations(predicted_item_id, predicted_rating, item_df):
    predicted_movie_features = item_df[item_df['MovieID'] == predicted_item_id]['Genre'].tolist()
    # print(predicted_movie_features)
    shortlisted_item_df = pd.DataFrame(columns=item_df.columns, index=None)
    for pred_mov_feature in predicted_movie_features[0].split(","):
        # print(pred_mov_feature)
        items_to_consider_df = item_df[item_df['Genre'].str.contains(pred_mov_feature, case=False)]
        # print(len(items_to_consider_df))
        items_to_consider_df = items_to_consider_df.drop_duplicates(subset=['Name', 'Rating'])
        shortlisted_item_df = shortlisted_item_df.append(items_to_consider_df, ignore_index=True)
        shortlisted_item_df = shortlisted_item_df.drop_duplicates(subset=['Name', 'Rating'])

    shortlisted_item_df = shortlisted_item_df[shortlisted_item_df['Rating'] >= predicted_rating]
    shortlisted_item_df = shortlisted_item_df.sort_values(by=['Rating'], ascending=False)

    return zip(shortlisted_item_df['MovieID'], shortlisted_item_df['Rating'])



matching_recommendations = recommendations(predicted_item_id,predicted_rating,item_df)

for recommendation in matching_recommendations:
  print(recommendation)

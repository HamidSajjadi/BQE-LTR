import pandas as pd
import data
df = pd.read_csv('./data/recall/expanded_and_labeled_queries_with_pubmed_recall_n100_200candidates.csv')
fold_queries = data.get_fold_queries(fold_number=1, split= "train", path='./data/queries_5_fold.csv')
fold_df = df[df["Q_name"].isin(fold_queries)]
out_df = data.create_pos_neg_pair_df(fold_df, max_sample_per_query=100)





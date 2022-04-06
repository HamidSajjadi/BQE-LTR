import pandas as pd
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv('./data/precision/expanded_and_labeled_queries_with_pubmed_precision_n1000_200candidates.csv')

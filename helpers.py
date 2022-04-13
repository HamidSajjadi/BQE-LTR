from pymed import PubMed
from typing import List, Union
from elasticsearch import Elasticsearch

from backoff import on_exception, expo, constant
from ratelimit import RateLimitException, limits
from sklearn.metrics import precision_score

pubmed = PubMed(
        tool="PubMedSearcher",
        email="sample@sample.com",
    )

es = Elasticsearch()
DEFAULT_SIZE = 1000
MAX_AVAILABLE_SIZE = 10000

@on_exception(expo, RateLimitException, max_tries=10)
@limits(calls=10, period=1)
def search_pubmed(query: str, max_results: int = 10000) -> List[int]:
    pubmed = PubMed(
        tool="PubMedSearcher_hamid",
        email="s.hamid.sajjadi@gmail.com",
        api_key="135b3c9a802d2b6fb3e6bc715ab14e634b09",
    )

    if not query:
        print("query is None")
        return []
    if not max_results:
        max_results = 10e6
    article_ids = pubmed._getArticleIds(query=query, max_results=max_results)
    articleInfo = []

    for article in article_ids:
        articleInfo.append(int(article))

    return articleInfo

def search_elastic(keywords: Union[str, List[str]], max_results=1000) -> List[int]:
    if isinstance(keywords, list):
        query = ' '.join(keywords)
    else:
        query = keywords

    query_dictionary = {
        "bool": {
            "should": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "text"],
                    "minimum_should_match" :"20%"
                }
            }
        }
    }
    if max_results and max_results <= MAX_AVAILABLE_SIZE:
        pmids = []
        res = es.search(index='cord_medline', query=query_dictionary, fields=[], _source=False, size=max_results,
                        filter_path=['hits.hits._id', 'took'],
                        track_total_hits=False,
                        request_timeout=500)
        hits = res['hits']['hits']
        for h in hits:
            try:
                pmids.append(int(h['_id']))
            except ValueError:
                pass
        # print(f"ES took: {res['took']}, Search Time {t1 - t0}, List Time: {t2 - t1}")
    else:
        pmids = __elastic_search_pit(query_dictionary, max_results)

    return pmids


def __elastic_search_pit(query, max_results):
    if max_results:
        size = min(max_results, MAX_AVAILABLE_SIZE)
    else:
        size = MAX_AVAILABLE_SIZE

    pit = es.open_point_in_time(index='cord_medline', keep_alive="5m")
    pmids = []

    def append_to_pmids(results):
        sa = None
        for idx, h in enumerate(results):
            if idx == len(hits) - 1:
                sa = h["sort"]
            try:
                pmids.append(int(h["_id"]))
            except Exception:
                continue
        return sa

    res = es.search(
        index='cord_medline',
        query=query,
        fields=["pmid"],
        size=size,
        request_timeout=500,
        pit=pit,
        sort=["_score"],
        _source=False,
        filter_path=["hits.hits._id", "hits.total.value", "hits.hits.sort"],
        track_total_hits=True,
    )
    hits = res["hits"]["hits"]
    total_hits = res["hits"]["total"]["value"]
    if not max_results:
        max_results = total_hits

    search_after = append_to_pmids(hits)

    while len(pmids) < max_results:
        size = min(MAX_AVAILABLE_SIZE, max_results - len(pmids))
        res = es.search(
            query=query,
            fields=["pmid"],
            size=size,
            request_timeout=500,
            pit=pit,
            sort=["_score"],
            track_total_hits=False,
            search_after=search_after,
        )
        hits = res["hits"]["hits"]
        search_after = append_to_pmids(hits)

    es.close_point_in_time(body=pit)
    return pmids


def precision(related_set, search_result, k=None):
    if not search_result or not len(search_result):
        return 0
    related_found = related_set.intersection(search_result)
    return len(related_found) / len(search_result)


def recall(related_set, search_result):
    if not search_result or not len(search_result):
        return 0
    related_found = related_set.intersection(search_result)
    return len(related_found) / len(related_set)

def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def evaluate_search_result(
    topic_id: str, search_result: List[int], qrels, metric: str = "r", n = 100
):
    topic_id_qrels = qrels[str(topic_id)]
    related_set = set(topic_id_qrels["related"])

    if metric == "p" or metric == "precision":
        return precision(related_set, search_result)
    elif metric == "r" or metric == "recall":
        return recall(related_set, search_result)
    elif metric == "ap" or metric == "average_precision":
        return apk(related_set, search_result, n)
    else:
        raise NotImplementedError("only precision and recall are supported")

@on_exception(expo, RateLimitException, max_tries=10)
@limits(calls=10, period=1)
def search_pubmed_for_labeled_data(query: str, list_of_labeled_pmids: List[int], top_n: int = 100,
                                   max_results: int = 100_000) -> List[int]:
    pubmed = PubMed(tool="PubMedSearcher_hamid",
                    email="s.hamid.sajjadi@gmail.com", api_key="135b3c9a802d2b6fb3e6bc715ab14e634b09")

    if not query:
        print('query is None')
        return []

    found = []
    total_results = pubmed.getTotalResultsCount(query)
    article_ids = pubmed._getArticleIds(
        query=query, max_results=max_results)

    for article_id in article_ids:
        article_id = int(article_id)
        if article_id in list_of_labeled_pmids:
            found.append(article_id)
            if len(found) >= top_n:
                return found[:top_n]
    if total_results > max_results:
        return search_pubmed_for_labeled_data(query, list_of_labeled_pmids, top_n, max_results * 2)
    else:
        return found



if __name__ == '__main__':
    related_set = [1,2,3,4,5]
    search_result = [1,8,3]
    print(precision(set(related_set), search_result,k=1))
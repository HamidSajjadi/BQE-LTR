import asyncio
import json

from helpers import (
    search_pubmed,
    search_pubmed_for_labeled_data,
    evaluate_search_result,
    search_elastic,
)


class Rewarder:
    def __init__(
        self,
        qrels_file,
        n_results=100,
        search_engine="pubmed",
        metric="recall",
    ):
        self.n_results = n_results
        self.metric = metric
        with open(qrels_file, "r") as qrels_file:
            self.qrels = json.load(qrels_file)
        if search_engine != "pubmed" and search_engine != "elastic":
            raise f"`{search_engine}` not implemented. Please use `pubmed` or `elastic`"
        self.se = search_engine
        self.qrels_file = qrels_file

    def batch_reward(self, sentences, ids):
        rewards = []

        for s, id in zip(sentences, ids):
            rewards.append(self.reward_one(s, id))
        return rewards

    async def async_batch_reward(self, sentences, ids):
        loop = asyncio.get_event_loop()
        futures = []
        response = []

        for s, id in zip(sentences, ids):
            futures.append(loop.run_in_executor(None, self.reward_one, s, id))
        for f in futures:
            response.append(await f)

        return response

    def reward_one(self, sample: str, id: str) -> float:
        search_function = search_pubmed if self.se == "pubmed" else search_elastic
        reward = 0.0
        try:
            result = search_function(sample, max_results=self.n_results)
            reward = evaluate_search_result(id, result, self.qrels, metric=self.metric, n=self.n_results)
        except Exception as e:
            print("error!")
            print(e)
        return reward

    def reward_with_labeled_data(self, sample: str, id: str):
        rels = self.qrels[id]
        list_of_pmids = [*rels["related"], *rels["unrelated"]]
        result = search_pubmed_for_labeled_data(
            sample, list_of_pmids, self.n_results, max_results=10 * self.n_results
        )
        reward = evaluate_search_result(id, result, self.qrels, metric=self.metric)
        return reward


if __name__ == "__main__":

    rewarder = Rewarder(
        t, "./data/cord_qrels.json", n_results=5000, search_engine="pubmed"
    )
    rewarder.reward_one("test", "test")

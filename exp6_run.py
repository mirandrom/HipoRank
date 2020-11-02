from hipo_rank.dataset_iterators.pubmed import PubmedDataset

from hipo_rank.embedders.bert import BertEmbedder
from hipo_rank.embedders.rand import RandEmbedder
from hipo_rank.embedders.sent_transformers import SentTransformersEmbedder

from hipo_rank.similarities.cos import CosSimilarity

from hipo_rank.directions.edge import EdgeBased
from hipo_rank.directions.order import OrderBased

from hipo_rank.scorers.add import AddScorer

from hipo_rank.summarizers.default import DefaultSummarizer
from hipo_rank.evaluators.rouge import evaluate_rouge

from pathlib import Path
import json
import time
from tqdm import tqdm

"""
Effect of document length
"""

DEBUG = False

DATASETS = [
    ("arxiv_nosections_test_0_3000",
     PubmedDataset, {"file_path": "data/arxiv-release/test.txt",
                     "max_words": 3000,
                     "no_sections": True}
     ),
    ("arxiv_nosections_test_3000_6000",
     PubmedDataset, {"file_path": "data/arxiv-release/test.txt",
                     "min_words": 3000, "max_words": 6000,
                     "no_sections": True}
     ),
    ("arxiv_nosections_test_6000_9000",
     PubmedDataset, {"file_path": "data/arxiv-release/test.txt",
                     "min_words": 6000, "max_words": 9000,
                     "no_sections": True}
     ),
    ("arxiv_nosections_test_9000_0",
     PubmedDataset, {"file_path": "data/arxiv-release/test.txt",
                     "min_words": 9000,
                     "no_sections": True}
     ),
    ("pubmed_nosections_test_0_2000",
     PubmedDataset, {"file_path": "data/pubmed-release/test.txt",
                     "max_words": 2000,
                     "no_sections": True}
     ),
    ("pubmed_nosections_test_2000_4000",
     PubmedDataset, {"file_path": "data/pubmed-release/test.txt",
                     "min_words": 2000, "max_words": 4000,
                     "no_sections": True}
     ),
    ("pubmed_nosections_test_4000_6000",
     PubmedDataset, {"file_path": "data/pubmed-release/test.txt",
                     "min_words": 4000, "max_words": 6000,
                     "no_sections": True}
     ),
    ("pubmed_nosections_test_6000_0",
     PubmedDataset, {"file_path": "data/pubmed-release/test.txt",
                     "min_words": 6000,
                     "no_sections": True}
     ),
        
]
EMBEDDERS = [
    ("rand_200", RandEmbedder, {"dim": 200}),
    ("pacsum_bert", BertEmbedder,
     {"bert_config_path": "models/pacssum_models/bert_config.json",
      "bert_model_path": "models/pacssum_models/pytorch_model_finetuned.bin",
      "bert_tokenizer": "bert-base-uncased",
      }
     )
]
SIMILARITIES = [
    ("cos", CosSimilarity, {}),
]
DIRECTIONS = [
    ("edge", EdgeBased, {}),
    ("order", OrderBased, {}),

]

SCORERS = [
    ("add_f=0.0_b=1.0_s=0.5", AddScorer, {"section_weight": 0.5}),
]


experiment_time = int(time.time())
# results_path = Path(f"results/{experiment_time}")
results_path = Path(f"results/exp6")

for embedder_id, embedder, embedder_args in EMBEDDERS:
    Embedder = embedder(**embedder_args)
    for dataset_id, dataset, dataset_args in DATASETS:
        DataSet = dataset(**dataset_args)
        docs = list(DataSet)
        if DEBUG:
            docs = docs[:5]

        if dataset_id.startswith("arxiv"):
            Summarizer = DefaultSummarizer(num_words=220)
        elif dataset_id.startswith("pubmed"):
            Summarizer = DefaultSummarizer(num_words=200)

        print(f"embedding dataset {dataset_id} with {embedder_id}")
        embeds = [Embedder.get_embeddings(doc) for doc in tqdm(docs)]
        for similarity_id, similarity, similarity_args in SIMILARITIES:
            Similarity = similarity(**similarity_args)
            print(f"calculating similarities with {similarity_id}")
            sims = [Similarity.get_similarities(e) for e in embeds]
            for direction_id, direction, direction_args in DIRECTIONS:
                print(f"updating directions with {direction_id}")
                Direction = direction(**direction_args)
                sims = [Direction.update_directions(s) for s in sims]
                for scorer_id, scorer, scorer_args in SCORERS:
                    Scorer = scorer(**scorer_args)
                    experiment = f"{dataset_id}-{embedder_id}-{similarity_id}-{direction_id}-{scorer_id}"
                    experiment_path = results_path / experiment
                    try:
                        experiment_path.mkdir(parents=True)

                        print("running experiment: ", experiment)
                        results = []
                        references = []
                        summaries = []
                        for sim, doc in zip(sims, docs):
                            scores = Scorer.get_scores(sim)
                            summary = Summarizer.get_summary(doc, scores)
                            results.append({
                                "num_sects": len(doc.sections),
                                "num_sents": sum([len(s.sentences) for s in doc.sections]),
                                "summary": summary,

                            })
                            summaries.append([s[0] for s in summary])
                            references.append([doc.reference])
                        rouge_result = evaluate_rouge(summaries, references)
                        (experiment_path / "rouge_results.json").write_text(json.dumps(rouge_result, indent=2))
                        (experiment_path / "summaries.json").write_text(json.dumps(results, indent=2))
                    except FileExistsError:
                        print(f"{experiment} already exists, skipping...")
                        pass


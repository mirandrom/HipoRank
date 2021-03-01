# HipoRank
Unsupervised and extractive long document summarization with **Hi**erarchichal and **Po**sitional information. Contains code for [Discourse-Aware Unsupervised Summarization of Long Scientific Documents](https://arxiv.org/abs/2005.00513) accepted at EACL 2021.

# Requirements
## Libraries
`pip install -r requirements.txt`

## Datasets
https://github.com/armancohan/long-summarization

## ROUGE
`pyrouge_set_rouge_path /absolute/path/to/ROUGE-1.5.5/directory`

## Model Files
- Biomed w2v: http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin
- PacSum BERT: https://drive.google.com/file/d/1wbMlLmnbD_0j7Qs8YY8cSCh935WKKdsP/view?usp=sharing
- BioBERT: https://github.com/dmis-lab/biobert
- sentence-transformers: https://github.com/UKPLab/sentence-transformers 


# Directory Structure

## hipo_rank
Library to define summarization pipeline as modular components with standard interfaces for ease of experimentation;

- `hipo_rank/dataset_iterators` yield document sections and sentences in standard typed format;
- `hipo_rank/embedders` generate vector representations for sentences and sections;
- `hipo_rank/similarities` compute similarities to weigh edges in document graph;
- `hipo_rank/directions` introduce directionality in document graph based on discourse structure;
- `hipo_rank/scorers` compute node centrality for sentences in document graph;
- `hipo_rank/summarizers` generate a summary from a document graph with node centrality scores;
- `hipo_rank/evaluators` evaluate summary with different metrics;


## Experiments
- `exp1_run.py` hyperparameter tuning and ablation study on PubMed val set;
- `exp2_run.py` cnndm validation set;
- `exp3_run.py` pubmed test set;
- `exp4_run.py` arxiv test set with pubmed hyperparams;
- `exp5_run.py` lead baseline;
- `exp6_run.py` effect of document length;
- `exp7_run.py` hyperparameter tuning on arxiv val set;
- `exp8_run.py` pacsum baseline;
- `exp9_run.py` similarity thresholding;
- `exp10_run.py` short sentence removal;
- `exp11_run.py` TextRank baseline;

## Human evaluation
- `human_eval_sample.ipynb` Code to sample examples for human evaluation;
- `human_eval_samples.jsonl` Sampled examples for human evaluation;
- `human_eval_data.jsonl` Results of human evaluation from Prodigy.ai annotation tool;
- `human_eval_results.ipynb` Code to generate human evaluation metrics;

## Plotting
- `plot_ablation.ipynb` Code to plot results of ablation study;
- `plot_sentence_positions.ipynb` Code to plot sentence positions in original document;














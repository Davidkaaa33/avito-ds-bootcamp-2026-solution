# Avito Data Science Bootcamp 2026

## Candidate Generation for Services

Final Recall@50: **0.831562**

This repository contains my solution for the Avito Data Science Bootcamp candidate-generation task.

## Task

For every search query, retrieve 50 relevant service listings.

The target metric is Recall@50.

## Final approach

The final candidate-generation system combines:

- BM25 lexical retrieval
- BAAI/bge-m3 semantic retrieval
- exact-location retrieval
- alternative-location retrieval
- geographic priors
- microcategory prediction
- joint location x microcategory retrieval
- exact-query historical positives
- historical positive-item embedding prototypes
- weighted Reciprocal Rank Fusion

## Main features

Query features:

- search_query
- search_infm_params_text
- search_location_id
- historical query-item interactions

Item features:

- item_title_raw
- item_description_raw
- item_infm_params_text
- item_location_id
- item_microcat_id

## Validation

Most experiments used query-disjoint validation.

Separate slices were used for tuning and independent confirmation.

Only changes that transferred to an independent validation slice were kept.

## Important improvements

- geographic prior and alternative locations
- alternative-location retrieval
- joint location x microcategory retrieval
- historical-item prototype retrieval

## Tested but rejected

- generic cross-encoder reranking
- fine-tuned reranker
- field-specific BM25
- conditional geo x microcategory prior
- query-to-query transfer

These approaches either reduced Recall@50 or did not transfer reliably to independent validation.

## Repository structure

- solution.py - final submission pipeline
- build_benchmark_assets.py - benchmark retrieval assets
- microcat_classifier.py - query-to-microcategory model
- check_submission.py - submission format validation
- answer.csv - final submitted answer
- experiments/ - selected validation experiments
- data/README.md - expected data layout

## Installation

Create a virtual environment and install:

    pip install -r requirements.txt

## Run

Place the required task data and generated artifacts into data/.

Then run:

    python solution.py

Validate the generated submission:

    python check_submission.py

## Final result

Recall@50: **0.831562**

SHA-256 of final answer.csv:

`093072acc21b856f79a982cf67b1d7ffa9f57252387c1b99b4cdea7c66f05cd6`

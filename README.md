# Mammography AI Analysis

## Overview

This repository contains data from mammography exams along with AI scores for the respective images. The aim is to evaluate the diagnostic performance of the AI algorithm in predicting the presence of cancer.

## Dataset Description

- **Data Source**: The dataset is provided in a [CSV file](https://github.com/medxiaorudan/AI-scores-analysis-of-mammography/blob/main/data/anon_dataset_POSTDOC_TASK_220323.csv).
- **Content**: Each row in the CSV corresponds to one mammography image. Typically, one exam consists of four images since two views (CC and MLO) of each breast are acquired during screening.
- **Identification**: Images from the same exam can be identified by matching the `anon_patient-id` (indicating the same person) and the `exam_date`.
- **AI Scores**: An AI model has processed all images and produced a score per breast. For both the CC and MLO images of the right breast, there's a single score, and similarly for the left breast.
- **Ground Truth**: The scores can be compared against a parameter named `x_groundtruth`. A value of `0` indicates women who were not diagnosed with cancer, while a value of `1` indicates women who were diagnosed with cancer.

## Code for Tasks

1. **Relevance of AI Score**: Determine which AI score from a single exam (comprising four images) is most relevant in deciding whether a woman should be recalled for further diagnostic work-up with the objective of diagnosing cancer. The metrics to consider are:
   - Average AI score for an exam
   - Maximum AI score for an exam

2. **Diagnostic Performance Metrics**: Calculate metrics that describe the diagnostic performance of the AI algorithm using the AI score suggested in the first task. For suggestions on which metrics to consider, you can refer to the attached publication or conduct your own research.

3. **AI model selection**: Use various AI models to explore the peculiarities or patterns that contribute to the mammography screening.



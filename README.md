# TechnicalProject - A Comparative Analysis of Mathematical Reasoning in Large Language Models

## Abstract
This report explores the importance of LLMs within mathematical reasoning. Mathematical problem-solving provides a benchmark for AI models to test their innate reasoning abilities relative to humans. The rise of LLMs in the past few years has sparked an interest in understanding the capabilities of these models. As they grow in scale (current state-of-the-art LLMs have billions of parameters). Questions arise about whether the sole path to increasing performance is through increased parameter size or if targeted training methods may enhance models further at a fraction of the training cost. To understand this relationship further, two compact models are studied (\texttt{TinyLLM} and \texttt{DistilGPT2}, with 10M and 82M, respectively). To provide context to these results, Machine Learning algorithms provide a baseline benchmark and larger LLMs upper-bound performance. The GSM8K benchmark is employed to quantify the models' ability to reason.

The baseline benchmark achieves 3 \% accuracy, where larger LLMs achieve up to 80 \%. In contrast, the base models underperform (under 2 \%). Through several training methods: Supervised Fine Tuning (SFT), Chain of Thought (CoT), and a custom Code Evaluator (CE) on a custom built dataset for this report GSM8K-v2. \texttt{DistilGPT2} accuracy rises to over 10 \% (\texttt{TinyLLM} to 2.2 \%), rivalling some LLMs with nearly 100 times the number of parameters. Overall, the results demonstrate that targeted training methods critically shape LLM reasoning. These findings suggest that democratising AI  through targeted training can foster cost-effective, energy-efficient solutions across diverse domains.

## REPOSITORY Structure
- **POSTER/**: Code to generate plots and data for the poster.
- **REPORT/**: All other code related to running experiments, plots, etc.

## POSTER
### Overview
This folder contains code specifically designed to generate the necessary plots and data for the poster presentation. These visualizations summarise key findings and results from the experiments in the report. 

You can view the poster here:
[Poster - A Comparative Analysis of Reasoning in LLMs](Poster/Poster.pdf)

### File Descriptions
- `dataset_accuracy.py`: Script to inital data on model ability to resonate.
- `plotposter.ipynb`: Plot summary of results (corresponds to figure 3 in poster).

## REPORT
### Overview
This folder contains the core experiments, methodology, and additional code required for the detailed analysis in the report. It includes everything from data preprocessing to model training, evaluation, and result analysis.

### File Descriptions
#### Introduction

#### Literature Review

#### Methodology

#### Results

#### Discussion

#### Conlsuion
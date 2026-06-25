# Norwegian Sentiment-Aware Summarisation

A Norwegian NLP project for generating concise, sentiment-aware summaries of user-generated book reviews.

The project combines targeted sentiment analysis, parameter-efficient fine-tuning of Norwegian large language models, and human evaluation. It was originally developed as a course project and is now being refactored into a more reusable model-development pipeline.

While the current domain is Norwegian book reviews, the project demonstrates a transferable workflow for developing AI-powered summarisation components. With domain-specific data, evaluation, and appropriate safeguards, the same architecture could be adapted for document archives, email threads, support tickets, or case overviews.

> **Note:** The fine-tuned models were trained and evaluated for Norwegian summarisation tasks. They are not production-ready for administrative or sensitive documents without domain-specific evaluation, access control, privacy assessment, and human review.

## Overview

The system takes Norwegian book reviews and discussions for a given title and produces an abstractive summary of how readers received the book.

Rather than passing all raw reviews directly to the summarisation model, the pipeline first identifies positive and negative sentiment targets. These are then used to construct a compact, sentiment-aware input for the language model.

```text
Norwegian book reviews and discussions
        ↓
Data cleaning and grouping by title
        ↓
Targeted sentiment analysis
        ↓
Positive and negative sentiment snippets
        ↓
Prompt construction
        ↓
Fine-tuned Norwegian LLM
        ↓
Abstractive, sentiment-aware review summary
        ↓
Human evaluation and error analysis
```

## Why This Project?

Long, unstructured text can be difficult to navigate. In many workflows, users need a quick overview of:

* What a document or collection of texts is about
* The most important positive and negative points, where sentiment is relevant
* What information may be useful for follow-up, prioritisation, or decision-making

This project explores one reusable component in that broader workflow: turning multiple pieces of Norwegian text into a short, coherent, source-informed overview.

## Data

The project uses user-generated Norwegian book reviews and discussions from Bokelskere.no, obtained through the National Library of Norway.

The original corpus contains reviews and comments with metadata such as:

* Review text
* Book title
* Author
* User rating
* Publication date
* User and post identifiers

For this project, the data was filtered to reviews associated with the most-reviewed books.

### Preprocessing

The preprocessing pipeline:

* Removes incomplete or unsuitable records
* Cleans URLs, digits, and non-standard symbols
* Filters records missing key metadata
* Groups reviews by book title
* Preserves relevant metadata such as author and average rating
* Aggregates multiple reviews into one book-level input

## Targeted Sentiment Analysis

The pipeline uses [`ltg/norbert3-large_tsa`](https://huggingface.co/ltg/norbert3-large_tsa) to identify sentiment targets in review text.

The extracted sentiment snippets are organised into a compact representation:

```text
Leserne likte: språk, karakterer, familiehistorie.
Leserne mislikte: forutsigbar handling og lav spenning.
```

This approach gives the summarisation model a focused representation of the most salient positive and negative aspects of reader feedback, rather than requiring it to process every raw review directly.

## Fine-Tuning

Two Norwegian instruction-tuned 7B language models were fine-tuned for abstractive summarisation:

* [`RuterNorway/Llama-2-7b-chat-norwegian`](https://huggingface.co/RuterNorway/Llama-2-7b-chat-norwegian)
* [`norallm/normistral-7b-warm`](https://huggingface.co/norallm/normistral-7b-warm)

Fine-tuning used the Norwegian Summarisation Benchmark Dataset (NorSumm) and combined supervised fine-tuning with QLoRA to reduce memory requirements while adapting the models for the task.

| Setting            | Value                          |
| ------------------ | ------------------------------ |
| Fine-tuning method | QLoRA + supervised fine-tuning |
| Quantisation       | 4-bit                          |
| LoRA rank          | 32                             |
| LoRA alpha         | 64                             |
| LoRA dropout       | 0.05                           |
| Learning rate      | 1e-5                           |
| Epochs             | 4                              |
| Optimiser          | `paged_adamw_8bit`             |
| Hardware           | Single NVIDIA A100 GPU         |

The resulting fine-tuned checkpoints are available on Hugging Face:

* [Add link to fine-tuned Llama checkpoint]
* [Add link to fine-tuned NorMistral checkpoint]

## Prompting and Generation

The models receive an instruction to transform the positive and negative sentiment snippets into a short, fluent Norwegian review summary.

The generation setup used:

* Beam search with `num_beams=4`
* Length penalty of `3.0`
* No-repeat n-gram size of `3`

The goal was to balance readability, consistency, and computational cost.

## Evaluation

Because the book-review dataset did not include gold-standard reference summaries, the generated summaries were evaluated manually.

The evaluation followed four dimensions from the SummEval framework:

| Dimension   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| Coherence   | Whether the summary is well structured and logically organised |
| Consistency | Whether the summary is faithful to the source material         |
| Fluency     | Whether the language is grammatically correct and readable     |
| Relevance   | Whether the summary captures the most important content        |

Three annotators first calibrated the rating criteria and achieved acceptable inter-annotator agreement before evaluating the remaining generated summaries.

### Main Results

Across 40 generated Norwegian book-review summaries, the fine-tuned Llama model performed better than NorMistral on coherence, consistency, and relevance. NorMistral performed slightly better on fluency.

| Dimension   | Fine-tuned Llama | Fine-tuned NorMistral |
| ----------- | ---------------: | --------------------: |
| Coherence   |             2.18 |                  2.05 |
| Consistency |             1.75 |                  1.43 |
| Fluency     |             2.10 |                  2.25 |
| Relevance   |             2.15 |                  1.40 |

These results suggest that the fine-tuned Llama model was the stronger candidate for this specific sentiment-aware Norwegian summarisation task.

## Example

### Input

```text
Leserne likte: språket, karakterene, familiehistorien og den gode fortellingen.
Leserne mislikte: handlingen var tidvis lite spennende og forutsigbar.
```

### Generated Summary

```text
Boken ble generelt godt mottatt av leserne, som særlig fremhevet språket,
karakterene og den sterke familiehistorien. Flere opplevde fortellingen som
engasjerende og velskrevet. Enkelte mente imidlertid at handlingen kunne være
forutsigbar og manglet spenning i enkelte partier.
```

## From Course Project to Reusable AI Component

The original work was developed in Google Colab as part of an NLP course project. This repository preserves the full research pipeline while restructuring it for clarity and reuse:

```text
Data preparation
        ↓
Sentiment tagging
        ↓
Fine-tuning
        ↓
Inference
        ↓
Evaluation
        ↓
Hugging Face model hosting
        ↓
Potential API or service integration
```

The same general pattern could support an internal AI service:

```text
Case notes, support tickets, email threads, or document excerpts
        ↓
Domain-specific preprocessing and retrieval
        ↓
Summarisation component
        ↓
Structured overview
        ↓
Human review
        ↓
Integration with an existing system
```

This repository focuses on the **model-development layer** of that architecture.

## Limitations and Responsible Use

This project is a research prototype and has important limitations:

* The models were fine-tuned on Norwegian news summarisation data and evaluated on book-review summarisation, not administrative cases.
* Generated summaries may hallucinate, omit important details, repeat content, or produce incomplete output.
* The source data did not include gold-standard book-review summaries, so the final task relied on human evaluation rather than automatic metrics alone.
* User-generated text may contain inaccurate, biased, or harmful language that can be reflected in generated summaries.
* The current implementation does not include authentication, authorisation, audit logging, monitoring, or a human approval interface.

A real deployment for internal documents would require:

* Domain-specific training or evaluation data
* Privacy and data-protection assessment
* Access control aligned with the source system
* Logging and monitoring
* Output validation
* Clear disclosure of AI-generated content
* Human review before consequential use

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_sentiment_tagging.ipynb
│   ├── 03_finetuning.ipynb
│   ├── 04_inference.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── sentiment_tagging.py
│   ├── prompting.py
│   ├── inference.py
│   └── evaluation.py
├── examples/
│   ├── sample_input.txt
│   └── sample_output.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproducibility

The repository documents the preprocessing, targeted sentiment analysis, QLoRA fine-tuning, generation, and evaluation workflow.

The underlying Bokelskere.no corpus is not included in this repository. It must be obtained separately through the National Library of Norway, and users are responsible for complying with the relevant terms of use.



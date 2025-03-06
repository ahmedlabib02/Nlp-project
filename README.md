NLP Project Milestone 1 Report – Data Analysis and Preprocessing
1. Introduction
Objective:
This milestone focuses on exploring, cleaning, and preprocessing the dataset to prepare it for downstream NLP tasks such as text classification and topic modeling. We analyze YouTube transcripts written in Arabic (with Egyptian dialect elements) and transform them into a consistent, noise-reduced representation suitable for machine learning.

Dataset Overview:

Size: 48 YouTube transcripts
Metadata: Each transcript comes with associated metadata (title, channel, category). One transcript is missing a category.
Language: The text includes both Modern Standard Arabic (MSA) and Egyptian dialect features.
2. Exploratory Data Analysis (EDA) – Raw Data
Before any cleaning, we performed an initial EDA to understand the structure and quality of the raw data.

2.1 Data Structure and Quality
Data Overview:
Using df.info() we confirmed the dataset contains 48 rows and 6 columns (including title, channel, category, transcript, raw_length, and diacritic_count).
Missing Values:
We found one missing value in the category column.
Duplicates:
We checked for duplicate transcripts to ensure data quality.
2.2 Descriptive Statistics
Transcript Length:
We computed the raw length (number of characters) of each transcript. The lengths ranged from 930 to 51,920 characters (mean ~21,778). This wide range signals variability that may require normalization or special handling.
Diacritic Count:
A diacritic count was also computed, providing insight into the consistency of diacritization across transcripts.
2.3 Metadata and Content
Category Distribution:
Visualizations of category counts highlighted potential imbalances.
Preliminary Word Frequency:
We performed a basic frequency analysis (after light cleaning) to identify the most common words, which helped inform the design of our custom stopword list.
Purpose of Raw EDA:
These initial steps established a baseline understanding of the dataset, its quality, and its variability, guiding our decisions for cleaning and normalization.

3. Preprocessing Pipeline
Our preprocessing pipeline consists of several steps designed to transform raw transcripts into a cleaner, more uniform format.

3.1 Cleaning and Normalization
Diacritic Removal:
Arabic diacritics were removed to unify different forms of the same word.
Normalization of Arabic Letters:
Variants of Alef (e.g., "أ", "إ", "آ") were normalized to "ا", and similar normalizations were applied to other letters.
Punctuation and Whitespace Cleaning:
Extraneous punctuation and inconsistent whitespace were removed.
3.2 Tokenization
Advanced Tokenization with Farasa:
We used Farasa for segmenting transcripts. Farasa helps in handling clitics and informal constructions typical in Egyptian dialect.
Post-Processing:
After tokenization, we removed segmentation markers (such as plus signs) to yield cleaner tokens.
3.3 Stopword Removal
Baseline and Custom Stopword List:
Starting with NLTK’s Arabic stopword list, we augmented it with additional common Egyptian function words (e.g., “ال”, “ان”, “عل”) identified from our raw frequency analysis.
Application:
The stopword removal step was applied to the tokenized data to filter out non-informative words.
3.4 Stemming and Lemmatization
Stemming with ISRIStemmer:
We applied the ISRIStemmer to reduce tokens to their roots, which reduced vocabulary size.
Lemmatization with CAMeL Analyzer:
We also lemmatized tokens using a CAMeL Analyzer (based on the “calima-msa-r13” model). Although this model is designed for MSA, it serves as a fallback when Egyptian-specific tools are unavailable.
Purpose of Preprocessing:
The objective was to produce a uniform representation that preserves semantic content while reducing noise and variability. This is essential for effective feature extraction and downstream modeling.

4. Exploratory Data Analysis (EDA) – Post-Preprocessing
After preprocessing, we conducted further analysis to assess the impact of our cleaning pipeline.

4.1 Vocabulary Analysis
Vocabulary Size and Frequency:
We computed the vocabulary size and used Counter to determine the most frequent tokens.
Observation: Some high-frequency tokens (e.g., “ال”) still appear, indicating further refinement of stopword removal might be needed.
4.2 Document Length Distribution
Token Count Per Document:
We calculated and visualized the number of tokens per document after processing.
Observation: Variability in document lengths can inform decisions on whether to normalize or segment long transcripts further.
4.3 Category-Specific Analysis
Top TF-IDF Terms Per Category:
We generated TF-IDF representations for each category and extracted the top terms.
Observation: In many categories, common function words still appear among the top terms, which suggests that stopword removal needs further tuning.
4.4 Visualizations
Word Cloud:
A word cloud was generated (using an Arabic-supportive font and proper reshaping) to visualize prominent tokens.
N-gram Analysis:
We also analyzed frequent bigrams/trigrams to capture multi-word expressions.
Purpose of Post-Preprocessing EDA:
This analysis confirmed that our preprocessing pipeline has substantially cleaned the data, though it also revealed areas for potential refinement—especially in stopword removal and morphological processing.

5. Analysis of Output and Limitations
Analysis of Output
TF-IDF Insights:
The top TF-IDF terms per document indicate that while some domain-specific words (like “حج”, “انتاجيه”, “رزق”) are highlighted, high-frequency function words (e.g., “ال”, “ان”) still appear. This suggests our stopword removal could be improved.
Morphological Processing Artifacts:
The stemming and lemmatization steps produced tokens that are sometimes truncated or include morphological artifacts (e.g., tokens with leftover clitic markers).
Vocabulary Reduction:
The reduction in vocabulary size post-processing helps reduce noise and improves feature consistency for modeling.
Limitations
Dialect vs. MSA:
Our use of MSA-based tools (e.g., “calima-msa-r13”) may not fully capture the nuances of Egyptian dialect, potentially affecting lemmatization quality.
Stopword Refinement:
Despite our efforts to customize the stopword list, some common function words persist in the final tokens.
Morphological Artifacts:
Aggressive stemming or segmentation can sometimes lead to truncated tokens that are less interpretable.
Small Dataset Size:
With only 48 transcripts, our analysis is based on a limited sample, which may affect the generalizability of our findings.
6. Conclusion and Next Steps
Summary:
We conducted a thorough EDA on both raw and preprocessed data, implemented a detailed preprocessing pipeline (cleaning, normalization, tokenization, stopword removal, and morphological processing), and analyzed the impact of these steps on vocabulary and document structure. Although our pipeline successfully reduced noise and standardized the data, certain limitations—especially related to dialectal nuances and persistent stopwords—remain.

Next Steps:

Feature Extraction:
Convert the processed text into numerical features (e.g., using TF-IDF or word embeddings) and build a baseline classification model.
Modeling and Evaluation:
Train a classifier (such as Logistic Regression or SVM) and evaluate its performance. Use error analysis to refine preprocessing steps.
Pipeline Refinement:
Experiment with further tuning of the stopword list and consider using more dialect-specific tools (if available) to improve morphological processing.
This report demonstrates our systematic approach to understanding, cleaning, and analyzing the dataset, providing a solid foundation for the subsequent milestones in the project.


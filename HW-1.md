# Global Objective

The main goal of the Capstone Project is to apply techniques and topics covered in the course to one of the industry-standard datasets. You will work in teams of two or three people.

Submissions will be divided into two parts - each part will be worth 50 points.

# Datasets

Approved datasets:

1. MovieLens Dataset [1M version] - [link](https://grouplens.org/datasets/movielens/1m/)
2. Book Crossing Dataset - [link](https://grouplens.org/datasets/book-crossing/)

Additionally, you can select any other open dataset that meets the following criteria:

- at least 10k users, 10k items, ≥ 500k interactions;
- interaction timestamps (or reliable ordering signal) available;
- either item metadata usable for content-based (text/tags/categories), or you must construct item features in a justified way.

Such a dataset should be approved at least two weeks before the homework deadline. To get the approval you need to submit:

- dataset link and description;
- basic statistics (users/items/interactions/time span)
- recommendation task formulation
- what metadata will you use for content-based approaches

# Submission #1: Classical Recommender Algorithms & Offline Evaluation

**Total Points:** 50

**Submission Structure:**

- link to the repository
- branch and commit hash for evaluation
- pdf report

## Objective

The goal of this submission is to build a solid classical recommender systems foundation, both theoretically and practically.

You are expected to understand, implement, and critically compare classical recommendation paradigms under a well-defined offline evaluation protocol.

This submission focuses on:

- similarity-based recommenders;
- matrix factorization;
- robust offline evaluation methodology;

The output should resemble a mini research prototype rather than a collection of disconnected notebooks.

## Components

### Repository Setup And Exploratory Data Analysis (5 points)

#### Objective

Set up a repository with a clear and structured layout to support iterative experimentation, results tracking, and code reusability. As a reference for the repository organization, you can use the attached screenshot, but you're not required to mimic it entirely.

Familiarize yourself with the dataset by conducting a comprehensive EDA to understand the dynamics of the selected dataset, identify patterns, and inform subsequent model development. EDA must go beyond descriptive statistics and directly inform modeling and evaluation choices.

#### Requirements

Repository must include:

- clear project structure (data / models / evaluation / experiments);
- README with setup and execution instructions;
- reproducible environment (requirements or environment file);

EDA must include:

- interaction sparsity analysis;
- user and item activity distributions;
- temporal dynamics (interaction volume over time);
- identification of at least two data pathologies (popularity skew, cold-start).

#### Deliverables

- GitHub repository
- README.md file with repository structure overview and setup instructions
- EDA notebook + written EDA summary with explicit modeling implications

### Offline Evaluation Strategy (10 points)

#### Objective

Define an offline evaluation methodology for assessing recommendation quality. The methodology will be reused across all models implemented in this submission.

#### Requirements

Evaluation methodology must define:

- data split strategy;
  - temporal split required (random split is not allowed);
  - mechanics of train/validation/test split;
- evaluation
  - rating prediction, classification, ranking with explicit justification for the choice
- metrics
  - at least one primary metric;
  - at least one secondary/diagnostic metric;

In your methodology description, you should explicitly discuss:

- what this evaluation setup captures;
- what it fails to capture;

#### Deliverables

- Evaluation code module (metrics implementation)
- Evaluation methodology design document (1-2 pages)

### Similarity-Based Recommenders (10 points)

#### Objective

Implement and analyze classical similarity-based recommendation approaches, with explicit attention to similarity choice and its implications.

#### Requirements

Implementation of content-based filtering:

- item representation must be explicitly defined;
- justification for the similarity function;

Collaborative filtering:

- choose between the user-user or the item-item approach;
- justification for the similarity function;

In justification for the similarity function choice, you need to compare at least two similarity functions.

#### Deliverables

- Model implementations
- Notebooks with experiments and evaluation
- Written comparison and interpretation of results (can be combined with matrix factorization models)

### Matrix Factorization (15 points)

#### Objective

Implement and evaluate two different matrix factorization techniques. Evaluate each model using an offline evaluation framework.

#### Requirements

Implementation of:

- FunkSVD
- Alternating Least Squares

In addition to implementations, you should discuss assumptions behind each method and differences in optimization behavior demonstrated via your experiments.

All MF models must be evaluated using the same offline protocol defined earlier.

#### Deliverables

- Model implementations
- Notebooks with experiments and evaluation
- Written comparison and interpretation of results (can be combined with matrix factorization models)

### Summary Report (5 points)

#### Objective

Analyze the results of all of the previous tasks and form an actionable summary and recommendations for the next steps.

#### Requirements

Provide a structured analysis covering:

- performance comparison across all implemented models;
- scenarios where each model fails;
- bias analysis (popularity, activity, etc.);
- which model would you deploy? how and why?;
- thoughts and considerations for the next steps;

This section is graded heavily on reasoning quality, not results.

#### Deliverables

- Final analysis document

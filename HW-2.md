# Submission #2

**Total Points:** 50

**Submission Structure:**

- link to the repository
- branch and commit hash for evaluation
- pdf report [optional in case of repository is self-explanatory with clear readme files]

## Objective

The goal of this submission is to move beyond point-wise prediction and classical collaborative filtering into ranking-oriented, hybrid, and online-aware recommender systems.

You are expected to:

- reason explicitly about ranking vs prediction;
- implement pairwise Learning-to-Rank methods with correct optimization logic;
- combine heterogeneous signals in hybrid recommenders;
- design and reason about online evaluation, including A/B testing and bandit algorithms.

## Components

### Ranking Heuristics & Graph-Based Signals (5 points)

#### Objective

Understand and evaluate non-learned ranking signals that are widely used in real systems as baselines, priors, or fallback mechanisms.

#### Requirements

Implement at least one ranking heuristic, such as:

- popularity- or recency-based ranking;
- graph-based propagation (e.g., item–item or user–item graph);
- PageRank / Personalized PageRank–style method.

You must discuss:

- what inductive bias the heuristic encodes;
- when it is competitive with learned models;
- when it fails systematically.

#### Deliverables

- Heuristic implementation
- Short written analysis (conceptual > empirical)

### Learning-to-Rank with Pairwise Optimization (10 points)

#### Objective

Move from heuristic ranking to explicitly optimized ranking models, focusing on pairwise Learning-to-Rank.

#### Requirements

*Model Implementation*

Implement Bayesian Personalized Ranking (BPR-OPT):

- correct pairwise loss;
- explicit negative sampling strategy;
- regularization and optimization choices clearly stated.

Core training logic must be implemented by you (no black-box trainers).

*Evaluation*

Compare BPR against at least one heuristic from Section 1 and use appropriate ranking metrics (e.g., Recall@K, NDCG@K, MAP@K).

You must analyze:

- convergence behavior;
- sensitivity to sampling;
- impact on head vs tail items.

#### Deliverables

- BPR implementation
- Experiments notebook
- Written comparison and interpretation

### Hybrid Recommender Systems (10 points)

#### Objective

Design hybrid recommenders that combine complementary signals.

#### Requirements

Implement at least one hybrid combining:

- collaborative signal;
- content-based or heuristic signal.

Possible strategies include:

- weighted blending;
- candidate generation + reranking;
- feature-level fusion.

You must justify:

- why the hybrid is structured this way;
- who benefits from it (which users/items).

#### Deliverables

- Hybrid model implementation
- Evaluation notebook
- Written analysis

### Classical Deep Learning for Recommendation (10 points)

#### Objective

Explore neural recommenders

#### Requirements

Implement and compare two classical deep learning–based recommender models. The chosen models should represent established neural approaches to recommendation (e.g., Neural Collaborative Filtering, Two-Tower models, autoencoder-based recommenders, Wide & Deep), but you are not limited to this list.

You must clearly justify your model choices and explain what modeling assumptions or representational advantages they introduce compared to non-neural baselines.

Constraints:

- architecture and objective must be explicit;
- same data split and metrics as other models;
- no "deep for the sake of deep".

You must discuss:

- representational differences vs MF/BPR;
- optimization and compute trade-offs;
- why performance improves or degrades.

#### Deliverables

- Model implementation
- Training & evaluation notebook
- Critical discussion

### Online Evaluation: A/B Testing & Bandits (10 points)

#### Objective

Demonstrate understanding of online evaluation and exploration, even in simulated form. Outline the methodology for A/B testing of recommender algorithms on your dataset. Make appropriate assumptions about the online system characteristics in place.

#### Requirements

*A/B Testing*

Design an A/B test:

- unit of randomization;
- primary and guardrail metrics;
- risks (novelty, position bias, feedback loops).

*Multi-Armed Bandits*

- Implement two bandit strategies (ε-greedy, UCB, or Thompson Sampling);
- simulate interactions using offline data;
- compare against each other and against static policy

You must explicitly explain:

- exploration–exploitation trade-offs;
- why offline metrics are insufficient.

#### Deliverables

- Simulation code
- Short methodological report

### Final System-Level Synthesis (5 points)

#### Objective

Reason like a production engineer, not a benchmark optimizer.

#### Requirements

Provide a structured reflection covering:

- offline vs online discrepancies;
- deployment choice and justification;
- iteration strategy post-deployment;
- key failure modes to monitor.

This section is graded primarily on reasoning quality.

# Ensemble Learning with Random Patches

This project was *completed as part of AIML232* at Te Herenga Waka — Victoria University of Wellington.

## Objective
Implement an ensemble algorithm using random subspaces and bagging (Random Patches) with decision trees as base learners. The model should support majority voting and weighted voting based on out-of-bag (OOB) accuracy.

## Dataset
- `electricity2.csv` - Contains the input features and target labels.

## Structure
- `ensemble_learning.ipynb` – Implements the Random Patches ensemble, trains multiple decision trees, calculates OOB accuracy, and demonstrates prediction with both majority and weighted voting.
- `electricity2.csv` – The dataset used for experiments.
- `requirements.txt` – Python dependencies.

## Methods
- Random Patches: Creates bootstrap samples of instances and subspaces of features for each ensemble member.
- Base Learner: `sklearn.tree.DecisionTreeClassifier`.
- Voting Schemes: Majority voting and OOB-weighted voting.
- OOB Metrics: Tracks the features used, OOB accuracy, and OOB sample size for each ensemble member.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
# Open `ensemble_learning.ipynb` and run all cells

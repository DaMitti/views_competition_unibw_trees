# CCEW Submission ViEWS Prediction Challenge 2023

## Overview

This repository contains the code of the [Center for Crisis Early Warning (CCEW)](https://www.unibw.de/ciss-en/ccew) submission to the [ViEWS Prediction Challenge 2023](https://viewsforecasting.org/research/prediction-challenge-2023/) titled *Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict*.

Further information on the competition:

Hegre et al. (Forthcoming). The 2023/24 ViEWS prediction competition. _Journal of Peace Research_, XXX.

## Introduction
Forecasting conflict, especially at the subnational level, involves high uncertainty in point predictions, which hampers practical application. Our contribution addresses this by integrating distribution-specific models into conflict prediction pipelines to estimate probability distributions and by incorporating regional modeling to account for local conflict dynamics.

We use a hurdle modeling strategy, combining binary classification for predicting the occurrence of fatalities with distributional regression for non-zero targets. We tackle the problem of zero-inflation by interpreting the classifier's probability as the share of zero-predictions in the final sample distributions, surpassing naive benchmarks. The framework we built for both steps (binary classification and regression) is rather flexible and auto-selects the best model based on tuning performance, so one can experiment with different base models quite easily. For our contribution to the competition we tested the following algorithms: 

For the binary classification step:
- [Random Forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://github.com/dmlc/xgboost)

For the regression step:
- [NGBoost](https://github.com/stanfordmlgroup/ngboost)
- [Distributional Random Forests](https://github.com/lorismichel/drf)
- [Quantile Regression Forests](https://github.com/zillow/quantile-forest)

The data used in this project was provided by the ViEWS team as part of the 2023/24 prediction challenge. It covers Africa and the Middle East, with monthly records available for each PRIO grid cell from 1990 onwards. The target variable is the number of fatalities from state-based armed conflict events, sourced from the Uppsala Conflict Data Program (UCDP). The dataset is highly zero-inflated, reflecting the rarity of conflict events, and includes a range of features relevant to predicting conflict dynamics.

More details about this work can be found in our technical report *Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict* in the repository root.

## Usage

> :warning: **This repo uses git-submodules!** Run `git submodule init` and  `git submodule update` after cloning or clone with `git clone --recurse-submodules`. The official [ViEWS Prediction Challenge repo](https://github.com/prio-data/prediction_competition_2023) is linked into this repo as a submodule and required for the evaluation scripts to run.

To download the data provided by the ViEWS team, run `./download_data.sh` if you are operating a Unix-like system or `./download_data.ps1` if you are on Windows. The download script also downloads the [CGAZ ADM0 GeoBoundaries](https://www.geoboundaries.org/globalDownloads.html) dataset used for creating the UN-region based clusters. You can then either run the whole (i) _training and prediction_ as well as the (ii) _evaluation_ pipelines directly on your machine using Python and a virtual environment or you can use our Docker image to run it (see below).

At the heart of the project is the _tuning and prediction pipeline_, which is defined and controlled by the `src/competition_pipeline.py` file. You can adjust any setting related to the initial training and the generation of prediction directly in `src/competition_pipeline.py`. After producing predictions based on the training runs, `src/evaluation_pipeline.py` evaluates the predictions against the metrics and benchmark models defined in the invitation to the [ViEWS Prediction Challenge 2023](https://viewsforecasting.org/research/prediction-challenge-2023/). As with the _tuning and prediction pipeline_, settings can be adjusted in the script directly. 

The `src/views_evaluation.ipynb` Jupyter notebook provides a few code snippets to gain additional insights, visualize our approach, and display the evaluation results of the generated predictions.


### Structure

The different components needed to run the pipeline are structured into estimators and utils.

- `src/estimators` includes sklearn-compatible wrappers for the Distributional Random Forest and the NGBoost implementations, as well as our hurdle class, which handles tuning, model selection and generation of raw predictions, and our global-local ensemble class.
- `src/utils` contains all workhorse functions called by the two pipelines to load data, handle conversions, create clusters, tune models, score models and handle evaluations.

The following outputs are stored in the repository:

- `src/tuning_trials_hdbscan` contains the hyperopt trials objects from hyperparameter tuning, which store tuning performance and are used to select the best models.
- `src/evaluation` contains metrics for our models and the five benchmarks.
- `src/unibw_trees*` directories contain our predictions for the respective model.
- `src/figures` contains the figures generated with the `views_evaluation.ipynb` notebooks for the technical report.

The required inputs for clustering and plotting are stored in `src/data`.


### Requirements

- Docker (if you want to run everything in Docker)
- Python 3.10 and R (if you want to run the code directly on your machine)

Running the _tuning and prediction pipeline_ took 1.5-2 weeks on 40 CPUs with 128GB memory.


### a) Vanilla Python

#### Unix-like systems

- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `(cd src && PYTHONPATH=../ python competition_pipeline.py)` or `./run_pipeline.sh`

#### Windows

The vanilla Python method is untested on Windows and will potentially fail. Use the Docker image instead (see below).

### b) Docker

You can either use the provided .sh/.ps1 script to start the docker container by adding the -d flag to `./run_pipeline.sh` (or by running `run_pipeline start` on Winbdows) or run the docker commands directly.

#### Unix-like systems

- `./run_pipeline.sh -d` or `./run_pipeline.sh -d predict` to start the container and run the pipeline. If you add the `-l` flag, the container still starts in the background but you will directly peek into the logs.
- `./run_pipeline.sh -d evaluate` runs the evaluation pipeline
- `./run_pipeline.sh logs` to peek into the logs of the running container
- `./run_pipeline.sh stop` to stop the running container

You can also do all of this manually by directly calling the required docker commands.

- `docker build -t ccew-tree .`
- `docker run -d -v .:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u competition_pipeline.py)"` or `docker run -d -v .:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u evaluation_pipeline.py)"`

#### Windows

- `.\run_pipeline.ps1 predict` to start the container and run the pipeline. If you add the `-l` flag, the container still starts in the background but you will directly peek into the logs.
- `.\run_pipeline.ps1 evaluate` to run the evaluation pipeline
- `.\run_pipeline.ps1 logs` to peek into the logs of the running container
- `.\run_pipeline.ps1 stop` to stop the running container

You can also do all of this manually by directly calling the required docker commands.

- `docker build -t ccew-tree .`
- `docker run -d -v ${PWD}:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u competition_pipeline.py)"` or `docker run -d -v .:/usr/src/app ccew-tree "(cd src && PYTHONPATH=../ python3.11 -u evaluation_pipeline.py)"`

## Reference
When you are using (parts of) our work, please cite the following technical report (currently only available upon request):
*Mittermaier, D., Bohne, T. & Hofer, M. (2024). Forests of UncertainT(r)ees: Using Tree-based Ensembles to Estimate Probability Distributions of Future Conflict.*

*Hegre, H. et al (Forthcoming), The 2023/24 VIEWS prediction competition, Journal of Peace Research, XXX.*

## Contributing
We welcome contributions to enhance the models and methodologies used in this study. Please submit pull requests or open issues for any suggestions or improvements.

## Contact
For any question, please contact one of the three original contributors of the ViEWS prediction challenge entry:
- [Tobias Bohne](mailto:tobias.bohne@unibw.de)
- [Martin Hofer](mailto:hofer.martin@pm.me)
- [Daniel Mittermaier](mailto:daniel.mittermaier@unibw.de)


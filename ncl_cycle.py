import glob
import functools
import pandas as pd
import numpy as np

from modAL import models

from al_for_fep import single_cycle_lib
from al_for_fep.utils import utils


TRAINING_KEY = 'Training'

class ALCycler(single_cycle_lib.MakitaCycle):

    def get_virtual_library(self):
        """Helper function to determine train/selection split."""
        feature_column = self._cycle_config.model_config.features.params['feature_column']
        target_column = self._cycle_config.model_config.targets.params['feature_column']

        virtual_lib = pd.read_csv(self._cycle_config.virtual_library)

        training_pool = []
        for fileglob in self._cycle_config.training_pool.split(','):
            for filename in glob.glob(fileglob):
                training_pool.append(pd.read_csv(filename))
        if training_pool:
            training_pool_ids = pd.concat(training_pool)
        else:
            training_pool_ids = pd.DataFrame({feature_column: []})

        selection_columns = self._cycle_config.selection_config.selection_columns

        for column in selection_columns:
            if column in virtual_lib.columns:
                continue

            if column == 'enamine_searched':
                virtual_lib[column] = False
            else:
                virtual_lib[column] = np.nan

        columns_to_keep = list(
            set(selection_columns + [feature_column, target_column]))
        virtual_lib = virtual_lib[columns_to_keep].drop_duplicates()

        virtual_lib[TRAINING_KEY] = virtual_lib[feature_column].isin(
            training_pool_ids[feature_column].values
        ) & ~virtual_lib[target_column].isna()

        return virtual_lib

    def run_cycle(self, virtual_library=None):

        model_config = self._cycle_config.model_config

        # training cases here are the S ie selected for oracle, ideally these would have the oracle predictions
        train_features, train_targets = self._get_train_features_and_targets(
            model_config, virtual_library[virtual_library[TRAINING_KEY]])

        library_features = self._get_selection_pool_features(
            model_config, virtual_library)

        selection_pool = virtual_library[~virtual_library[TRAINING_KEY]]
        selection_pool_features = self._get_selection_pool_features(
            model_config, selection_pool)

        estimator = utils.MODELS[model_config.model_type](
            model_config.hyperparameters, model_config.tuning_hyperparameters)

        if 'halfsample_log2_shards' in model_config:
            estimator = utils.HALF_SAMPLE_WRAPPER(
                subestimator=estimator.get_model(),
                shards_log2=model_config.halfsample_log2_shards,
                add_estimators=model_config.model_type in ['rf', 'gbm'])

        selection_config = self._cycle_config.selection_config
        query_strategy = functools.partial(
            utils.QUERY_STRATEGIES[selection_config.selection_type],
            n_instances=selection_config.num_elements,
            **selection_config.hyperparameters)

        target_multiplier = 1
        if selection_config.selection_type in ['thompson', 'EI', 'PI', 'UCB']:
            target_multiplier = -1

        train_targets = train_targets * target_multiplier

        if model_config.model_type in single_cycle_lib._BAYESIAN_MODELS:
            learner = models.BayesianOptimizer(
                estimator=estimator.get_model(),
                X_training=train_features,
                y_training=train_targets,
                query_strategy=query_strategy)
        else:
            learner = models.ActiveLearner(
                estimator=estimator.get_model(),
                X_training=train_features,
                y_training=train_targets,
                query_strategy=query_strategy)

        inference = learner.predict(library_features) * target_multiplier

        virtual_library['regression'] = inference.T.tolist()

        selection_idx, _ = learner.query(selection_pool_features)
        selection_columns = self._cycle_config.selection_config.selection_columns

        # selections, virtual_library
        return selection_pool.iloc[selection_idx][selection_columns], virtual_library
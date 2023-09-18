# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for running a single cycle of active learning."""
import time
import pandas as pd
import numpy as np
from pathlib import Path

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler


class AL:
    def __init__(self, config, initial_points=[]):
        previous_trainings = list(map(str, Path('generated').glob('cycle_*/selection.csv')))

        # specify training
        # config.training_pool = ','.join(["init.csv"] + previous_trainings)
        config.training_pool = ','.join(previous_trainings)
        print('Using trainings: ', config.training_pool)

        self.cycle = len(previous_trainings)
        self.cycler = ALCycler(config)
        self.virtual_library = self.cycler.get_virtual_library(initial_points)

    def report(self):
        # select only the ones that have been chosen before
        best_finds = self.virtual_library[self.virtual_library.cnnaffinity < -6]  # -6 is about 5% of the best cases
        print(f"IT: {self.cycle},Lib size: {len(self.virtual_library)}, "
              f"training size: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"cnnaffinity 0: {len(self.virtual_library[self.virtual_library.cnnaffinity == 0])}, "
              f"<-6 cnnaff: {len(best_finds)}")

    def get_next_best(self):
        start_time = time.time()
        chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)

        print(f"Found next best in: {time.time() - start_time}")
        self.cycle += 1
        return chosen_ones

    def set_answer(self, chosen_ones, smiles, result):
        # add this result
        self.virtual_library.loc[self.virtual_library.Smiles == smiles, 'cnnaffinity'] = result['cnnaffinity']
        # mark for future training
        self.virtual_library.loc[self.virtual_library.Smiles == smiles, ncl_cycle.TRAINING_KEY] = True

    def expand_chemical_space(self):
        pass
        # expand the virtual library
        # if len(virtual_library[virtual_library.Smiles == "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C"]) == 0:
        #     new_record = pd.DataFrame([{'Smiles': "CN(C(=O)c1cn(C)nc1-c1ccc(F)cc1F)c1nc2ccccc2n1C", ncl_cycle.TRAINING_KEY: False}])
        #     expanded_library = pd.concat([virtual_library_regression, new_record], ignore_index=True)
        #     virtual_library = expanded_library

    def csv_cycle_summary(self, chosen_ones):
        cycle_dir = Path(f"generated/cycle_{self.cycle:04d}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        self.virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=config.selection_config.selection_columns, index=False)
        self.report()


oracle = pd.read_csv("negative_oracle.csv")
def compute_fegrow(smiles):
    result = oracle[oracle.Smiles == smiles]
    return {'cnnaffinity': result.cnnaffinity.values[0]}


if __name__ == '__main__':
    config = get_gaussian_process_config()
    config.virtual_library = "chemical_space_500.csv"
    config.selection_config.num_elements = 20  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles"]
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    # initial datapoints
    random_starter = pd.read_csv(config.virtual_library).sample(20)
    for i, row in random_starter.iterrows():
        result = compute_fegrow(row.Smiles)
        random_starter.at[i, 'cnnaffinity'] = result['cnnaffinity']
    random_starter.to_csv('random_starter.csv', index=False)

    al = AL(config, random_starter)

    for i in range(8):
        chosen_ones = al.get_next_best()
        for i, row in chosen_ones.iterrows():
            result = compute_fegrow(row.Smiles)  # TODO no conformers? penalise
            al.set_answer(chosen_ones, row.Smiles, result)
            # update for stats
            chosen_ones.at[i, 'cnnaffinity'] = result['cnnaffinity']
        al.csv_cycle_summary(chosen_ones)

    print('hi')


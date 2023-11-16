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
import os.path
import time
import pandas as pd
from pathlib import Path

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler
from rdkit.SimDivFilters import rdSimDivPickers

from rdkit import DataStructs
from rdkit.DataStructs import cDataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors

cwd = os.getcwd()

class ActiveLearner:
    def __init__(self, config, initial_values=pd.DataFrame()):
        self.feature_column = config.model_config.targets.params.feature_column
        previous_trainings = list(map(str, Path('generated').glob('cycle_*/selection.csv')))
        config.training_pool = ','.join([config.training_pool] + previous_trainings)
        print(f'Using trainings: {config.training_pool}')
        
        self.cycle = len(previous_trainings)
        self.cycler = ALCycler(config)
        self.virtual_library = self.cycler.get_virtual_library()
        print('virtual library columns: ', self.virtual_library.columns)
        print('feature_column: ', self.feature_column)

    def report(self):
        best_finds = self.virtual_library[self.feature_column].mean() #self.virtual_library[self.virtual_library[self.feature_column]]  # NOT BEST JUST ALL
        print(f"IT: {self.cycle}, lib: {len(self.virtual_library)}, "
              f"training: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"{self.feature_column} no: {len(self.virtual_library[~self.virtual_library[self.feature_column].isna()])}, "
              f"mean {self.feature_column}: {best_finds}")

    def row_to_bitvect(self, row):
        bit_string = ''.join(str(bit) for bit in row)
        return cDataStructs.CreateFromBitString(bit_string)

    def diverse_sample(self, n_select, *args, **kwargs):
        rgroup_df = pd.read_csv('rgroups_clust_df.csv')
        # linker_df = pd.read_csv('linkers_clustered.csv') # Uncomment if linkers are to be used
        dfs = [rgroup_df]  # Add linker_df to the list if needed
        thresh = 0.6
        selected_molecules = []

        for df in dfs:
            clusters = df['cluster'].unique()
            picked_list = []
            cluster_dfs = []
            for cluster in clusters:
                cluster_df = df[df['cluster'] == cluster]
                fps = cluster_df['fp'].apply(self.row_to_bitvect).tolist()

                picker = rdSimDivPickers.LeaderPicker()
                picked_indices = list(picker.LazyBitVectorPick(fps, len(fps), thresh))
                fp1 = fps[picked_indices[0]]
                fp2 = fps[picked_indices[1]]

                similarity = DataStructs.FingerprintSimilarity(fp1, fp2)

                print(f"for cluster {cluster}: \n picked smiles : {cluster_df['Smiles'].iloc[picked_indices[0]]} and "
                      f"{cluster_df['Smiles'].iloc[picked_indices[1]]}, "
                      f"their tanimoto similarity is {similarity}")

                picked_list.append(picked_indices)
                clust_pick_df = cluster_df.iloc[picked_indices[:2]]
                cluster_dfs.append(clust_pick_df)
                print(f'for cluster {cluster} picked {len(picked_indices)} mols')
                selected_molecules.extend(clust_pick_df)


            # Concatenate all the DataFrames from each cluster into one
            pick_df = pd.concat(cluster_dfs, ignore_index=True) #TODO retrieve mols from chemical space with these rgroups by finding all smiles that were created from these rgroup IDs (ID is currently nan)


        return pick_df

#
    #
    #
            #get rgroup/linker ID for picked smiles
            #repeat 2 times, one for rgroups, one for linkers

        # df of n smiles, rgroup_ID, linker_ID
        # get ID for rgroups/linkers

        # list of  ids = [rgroup_ID, linker_ID]
        # get mols from chemical space that have rgroup_id = ids[0], linker_ID = ids[1]
        # return dataframe of id from total space, smiles, cnnaffinity, training (T/F)


        #return list of chosen smiles

    def get_next_best(self, random=True):
        not_null_rows = self.virtual_library[self.virtual_library[self.feature_column].notnull()]
        print(not_null_rows)
        if len(not_null_rows) == 0:
            print("No previous training data, randomising initial selection")
            true_rows = self.virtual_library[self.virtual_library.Training == True]
            if len(true_rows) != 0:
                raise AssertionError(f"Found rows with Training == True but feature column values are null, virtual library hasn't been updated from previous cycle. The rows are:\n{true_rows}")
            if random is True:
                starter = self.virtual_library.sample(self.cycler._cycle_config.selection_config.num_elements)
            else:
                print('Diverse set : True')
                starter = self.diverse_sample(self.virtual_library, self.cycler._cycle_config.selection_config.num_elements)
            return starter
        # AL
        start_time = time.time()
        print('AL on virtual library: ', self.virtual_library)
        chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)
        print(f"Found next best in: {time.time() - start_time}")
        self.cycle += 1
        return chosen_ones

    def set_answer(self, smiles, result):
        self.virtual_library.loc[self.virtual_library.Smiles == smiles, 
                                 [self.feature_column, ncl_cycle.TRAINING_KEY]] = result[self.feature_column], True  # Step 2

    def csv_cycle_summary(self, chosen_ones):
        cycle_dir = Path(f"{cwd}/generated/cycle_{self.cycle:04d}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        self.virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=self.cycler._cycle_config.selection_config.selection_columns, index=False)
        self.report()




if os.path.exists("negative_oracle.csv"):
    oracle = pd.read_csv("negative_oracle.csv")
def compute_fegrow(smiles):
    result = oracle[oracle.Smiles == smiles]
    return {'cnnaffinity': result.cnnaffinity.values[0]}


def expand_chemical_space(al):
    """
    For now, add up to a 100 of new random smiles as datapoints.
    """
    extras = oracle.sample(100).drop(columns=['cnnaffinity'], axis=1)
    not_yet_in = extras[~extras.Smiles.isin(al.virtual_library.Smiles.values)]
    not_yet_in = not_yet_in.assign(Training=False)   # fixme: this will break if we use a different keyword
    print(f'Adding {len(not_yet_in)} smiles out of 100')

    extended = pd.concat([al.virtual_library, not_yet_in], ignore_index=True)
    al.virtual_library = extended


if __name__ == '__main__':
    config = get_gaussian_process_config()
    config.virtual_library = "manual_init.csv"
    config.selection_config.num_elements = 30  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles"]
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048
    al = ActiveLearner(config)
    for i in range(5):
        chosen_ones = al.get_next_best(random=False)
        chosen_ones['Smiles']
        for i, row in chosen_ones.iterrows():
            result = compute_fegrow(row.Smiles)  # TODO no conformers? penalise
            al.set_answer(row.Smiles, result)
            # update for record keeping
            chosen_ones.at[i, 'cnnaffinity'] = result['cnnaffinity']
        al.csv_cycle_summary(chosen_ones)
        expand_chemical_space(al)
        cfg_json = config.to_json()

        with open(f'{i}_config.json', 'w') as fout:
            fout.write(cfg_json)

    print('hi')


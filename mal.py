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
import warnings
import subprocess

import pandas as pd
import numpy
import requests
from pathlib import Path

from rdkit import Chem
import dask

from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
import ncl_cycle
from ncl_cycle import ALCycler
from enamine import Enamine

class ActiveLearner:
    def __init__(self, config, client, initial_values=pd.DataFrame()):
        self.feature = config.model_config.targets.params.feature_column
        generated = Path('generated')

        previous_trainings = list(map(str, generated.glob('cycle_*/selection.csv')))
        if config.training_pool.strip() != '':
            previous_trainings += [config.training_pool]
        print(f'Detected {len(previous_trainings)} selection.csv which define the Training. ')
        config.training_pool = ','.join(previous_trainings)

        if previous_trainings:
            # use the latest full .csv which already has Training set
            # instead of the initial one
            config.virtual_library = str(generated / f"cycle_{len(previous_trainings):04d}" / "virtual_library_with_predictions.csv")

        self.cycle = len(previous_trainings)
        self.cycler = ALCycler(config)
        self.virtual_library = self.cycler.get_virtual_library()

        extra = ''
        if 'enamine_id' in self.virtual_library.columns:
            extra = f'Enamines: {len(self.virtual_library[self.virtual_library.enamine_id.notna()])}, '\
                    f'Enamines Training: {len(self.virtual_library[self.virtual_library.enamine_id.notna() & self.virtual_library.Training == True])}'
        print(f'Launching AL. '
              f'Cycle: {self.cycle}, '
              f'Features: {len(self.virtual_library[self.virtual_library[self.feature].notna()])}, '
              f'Training: {len(self.virtual_library[self.virtual_library.Training == True])}, ' +
              extra
              )

        self.enamine = Enamine()
        self.client = client

    def report(self):
        best_finds = self.virtual_library[self.feature].mean() #self.virtual_library[self.virtual_library[self.feature_column]]  # NOT BEST JUST ALL
        print(f"IT: {self.cycle}, lib: {len(self.virtual_library)}, "
              f"training: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"{self.feature} no: {len(self.virtual_library[~self.virtual_library[self.feature].isna()])}, "
              f"mean {self.feature}: {best_finds}")

    def get_next_best(self):
        # in the first iteration there is no data, pick random molecules
        not_null_rows = self.virtual_library[self.virtual_library.cnnaffinity.notnull()]
        if len(not_null_rows) == 0:
            # there is nothing dedicated to Training yet
            assert len(self.virtual_library[self.virtual_library.Training == True]) == 0

            random_starter = self.virtual_library.sample(self.cycler._cycle_config.selection_config.num_elements)
            return random_starter

        start_time = time.time()
        chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)
        print(f"Found next best {len(chosen_ones)} in: {time.time() - start_time:.1f}s")

        enamines = virtual_library_regression[virtual_library_regression.enamine_id.notna() &
                                              virtual_library_regression.cnnaffinity.isna()]
        if len(enamines) > 0:
            print(f"Adding on top {len(enamines)} Enamine molecules to be computed.")
        # select only the ones that have been chosen before
        best_finds = self.virtual_library[self.virtual_library[self.feature] < -6]  # -6 is about 5% of the best cases
        print(f"IT: {self.cycle}, lib: {len(self.virtual_library)}, "
              f"training: {len(self.virtual_library[self.virtual_library.Training])}, "
              f"feature no: {len(self.virtual_library[~self.virtual_library[self.feature].isna()])}, "
              f"<-6 feature: {len(best_finds)}")

    def get_next_best(self, force_random=False, add_enamines=0):
        self.cycle += 1

        # pick random molecules
        rows_not_yet_computed = self.virtual_library[~self.virtual_library[self.feature].notnull()]
        if len(rows_not_yet_computed) == len(self.virtual_library) or force_random:
            print("Selecting random molecules to study. ")
            chosen_ones = rows_not_yet_computed.sample(self.cycler._cycle_config.selection_config.num_elements)
        else:
            start_time = time.time()
            chosen_ones, virtual_library_regression = self.cycler.run_cycle(self.virtual_library)
            print(f"AL: generated next best {len(chosen_ones)} in: {time.time() - start_time:.1f}s")

        if add_enamines > 0:
            enamines = self.virtual_library[self.virtual_library.enamine_id.notna() &
                                                  self.virtual_library[self.feature].isna()]
            if len(enamines) > 0:
                print(f"Adding on top {len(enamines)} Enamine molecules to be computed.")
                chosen_ones = pd.concat([chosen_ones, enamines])

        return chosen_ones

    def set_feature_result(self, smiles, value):
        self.virtual_library.loc[self.virtual_library.Smiles == smiles,
                                 [self.feature, ncl_cycle.TRAINING_KEY]] = value, True

    @staticmethod
    @dask.delayed
    def _obabel_protonate(smi):
        return subprocess.run(['obabel', f'-:{smi}', '-osmi', '-p', '7', '-xh'],
                              capture_output=True).stdout.decode().strip()

    @staticmethod
    @dask.delayed
    def _scaffold_check(smih, scaffold):
        params = Chem.SmilesParserParams()
        params.removeHs = False

        mol = Chem.MolFromSmiles(smih, params=params)
        if mol is None:
            return False, None

        if mol.HasSubstructMatch(scaffold):
            return True, smih

        return False, None


    def add_enamine_molecules(self, scaffold, limit=300):
        """
        For the best scoring molecules that were grown,
        check if there are similar molecules in Enamine REAL database,
        if they are, add them to the dataset.

        @number: The maximum number of molecules that
            should be added from the Enamine REAL database.
        @similarity_cutoff: Molecules queried from the database have to be
            at least this similar to the original query.
        @scaffold: The scaffold molecule that has to be present in the search.
            If None, the requirement will be ignored.
        """

        # get the best performing molecules
        vl = self.virtual_library.sort_values(by='cnnaffinity')
        best_vl_for_searching = vl[:100]

        # nothing to search for yet
        if len(best_vl_for_searching[~best_vl_for_searching.cnnaffinity.isna()]) == 0:
            return

        if len(set(best_vl_for_searching.h)) > 1:
            raise NotImplementedError('Multiple growth vectors are used. ')

        # filter out previously queried molecules
        new_searches = best_vl_for_searching[best_vl_for_searching.enamine_searched == False]
        smiles_to_search = list(new_searches.Smiles)

        start = time.time()
        print('Querying Enamine REAL. ')
        try:
            results: pd.DataFrame = self.enamine.search_smiles(smiles_to_search, remove_duplicates=True)
        except requests.exceptions.HTTPError as HTTPError:
            print("Enamine API call failed. ", HTTPError)
            return
        print(f"Enamine returned with {len(results)} rows in {time.time() - start:.1f}s.")

        # prepare the scaffold for testing its presence
        # specifically, the hydrogen was replaced and has to be removed
        # for now we assume we only are growing one vector at a time - fixme
        scaffold_noh = Chem.EditableMol(scaffold)
        scaffold_noh.RemoveAtom(int(best_vl_for_searching.iloc[0].h))
        dask_scaffold = dask.delayed(scaffold_noh.GetMol())

        start = time.time()
        # protonate and check for scaffold
        delayed_protonations = [ActiveLearner._obabel_protonate(smi.rsplit(maxsplit=1)[0])
                            for smi in results.hitSmiles.values]
        jobs = self.client.compute([ActiveLearner._scaffold_check(smih, dask_scaffold)
                                             for smih in delayed_protonations])
        scaffold_test_results = [job.result() for job in jobs]
        scaffold_mask = [r[0] for r in scaffold_test_results]
        # smiles None means that the molecule did not have our scaffold
        protonated_smiles = [r[1] for r in scaffold_test_results if r[1] is not None]
        print(f"Dask obabel protonation + scaffold test finished in {time.time() - start:.2f}s.")
        print(f"Tested scaffold presence. Kept {sum(scaffold_mask)}/{len(scaffold_mask)}.")

        if len(scaffold_mask) > 0:
            similar = results[scaffold_mask]
            similar.hitSmiles = protonated_smiles
        else:
            similar = pd.DataFrame(columns=results.columns)

        # filter out Enamine molecules which were previously added
        new_enamines = similar[~similar.id.isin(vl.enamine_id)]

        # fixme: automate creating empty dataframes. Allow to configure default values initially.
        new_enamines_df = pd.DataFrame({'Smiles': new_enamines.hitSmiles,
                               self.feature: numpy.nan,
                               'h': vl.h[0], # fixme: for now assume that only one vector is used
                               'enamine_id': new_enamines.id,
                               'enamine_searched': False,
                               'Training': False })

        print("Adding: ", len(new_enamines_df))
        self.virtual_library = pd.concat([self.virtual_library, new_enamines_df], ignore_index=True)

    def csv_cycle_summary(self, chosen_ones):
        cycle_dir = Path(f"generated/cycle_{self.cycle:04d}")
        print(f"Next generation report saved: {cycle_dir}")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        self.virtual_library.to_csv(cycle_dir / 'virtual_library_with_predictions.csv', index=False)
        chosen_ones.to_csv(cycle_dir / "selection.csv", columns=self.cycler._cycle_config.selection_config.selection_columns, index=False)
        self.report()




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
    config.virtual_library = "chemical_space_smiles_500.csv"
    config.selection_config.num_elements = 30  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles"]
    config.model_config.targets.params.feature_column = 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    al = ActiveLearner(config)

    for i in range(5):
        chosen_ones = al.get_next_best()
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


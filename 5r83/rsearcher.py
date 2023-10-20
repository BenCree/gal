"""
FG_SCALE - if fewer tasks are in the queue than this, generate more. This also sets that many "proceses" in Dask.

 - Paths:
export FG_ANIMODEL_PATH=/path/to/animodel.pt
export FG_GNINA_PATH=/path/to/gnina
"""

import copy
import time
import random
import glob
import sys
import tempfile
import os
from pathlib import Path
import logging
import datetime
import threading
import queue
import cProfile

import dask
from dask.distributed import Client, performance_report
from rdkit import Chem
from rdkit.Chem import Descriptors
import openmm.app
import pandas as pd
import numpy as np

import helpers
from helpers import gen_intrns_dict, xstal_set, plip_score, sf1
import fegrow


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from mycluster import create_cluster
except ImportError:
    # set up a local cluster just in case
    def create_cluster():
        from dask.distributed import LocalCluster
        lc = LocalCluster(processes=True, threads_per_worker=1, n_workers=28)
        lc.adapt(maximum_cores=28)
        return lc


# Define the modified sf1 function to operate on a dictionary
def sf1(rmol_data_dict):
    # Assert that rmol_data_dict contains the necessary data
    required_keys = ['plip', 'interactions', 'cnn_ic50_norm', 'MW', 'QED', 'cnnaffinity']
    missing_keys = [key for key in required_keys if key not in rmol_data_dict]

    if missing_keys:
        raise ValueError(f"Missing necessary data: {', '.join(missing_keys)}")

    # Calculate the score
    sf1 = rmol_data_dict['QED'] * rmol_data_dict['plip'] * (1 + (0.5 * len(rmol_data_dict['interactions']))) * (
                10 ** (rmol_data_dict['cnnaffinity'] / rmol_data_dict['MW']))

    return sf1


@dask.delayed
def evaluate(scaffold, h, smiles, pdb_load):
    t_start = time.time()
    fegrow.RMol.set_gnina(os.environ['FG_GNINA_PATH'])
    with tempfile.TemporaryDirectory() as TMP:
        TMP = Path(TMP)
        os.chdir(TMP)
        print(f'TIME changed dir: {time.time() - t_start:.1f}s')

        # make a symbolic link to animodel
        # ani = Path(os.environ['FG_ANIMODEL_PATH'])
        # Path('animodel.pt').symlink_to(ani)
        # print(f'TIME linked animodel: {time.time() - t_start}')

        protein = str(TMP / 'protein_tmp.pdb')
        with open(protein, 'w') as PDB:
            PDB.write(pdb_load)

        params = Chem.SmilesParserParams()
        params.removeHs = False  # keep the hydrogens
        rmol = fegrow.RMol(Chem.MolFromSmiles(smiles, params=params))
        # remove the h
        scaffold = copy.deepcopy(scaffold)
        scaffold_m = Chem.EditableMol(scaffold)
        scaffold_m.RemoveAtom(int(h))
        scaffold = scaffold_m.GetMol()
        rmol._save_template(scaffold)
        print(f'TIME prepped rmol: {time.time() - t_start:.1f}s')

        rmol.generate_conformers(num_conf=50, minimum_conf_rms=0.4)
        print('Number of simple conformers: ', rmol.GetNumConformers())

        rmol.remove_clashing_confs(protein)
        print(f'TIME conformers done: {time.time() - t_start:.1f}s')

        rmol.optimise_in_receptor(
            receptor_file=protein,
            ligand_force_field="openff",
            use_ani=True,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name='CPU',
        )

        if rmol.GetNumConformers() == 0:
            raise Exception("No Conformers")

        print(f'TIME opt done: {time.time() - t_start:.1f}s')
        rmol.sort_conformers(energy_range=2) # kcal/mol
        affinities = rmol.gnina(receptor_file=protein)
        data = {
            "cnnaffinity": -affinities.CNNaffinity.values[0],
            "cnnaffinityIC50": affinities["CNNaffinity->IC50s"].values[0],
            "hydrogens": [atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomicNum() == 1]
        }



        tox = rmol.toxicity()
        tox = dict(zip(list(tox.columns), list(tox.values[0])))
        tox['MW'] = Descriptors.HeavyAtomMolWt(rmol)
        for k, v in tox.items():
            data[k] = v

        # compute all props
        print(f'Calculating PLIP')

        plip_itrns = rmol.plip_interactions(receptor_file=protein)
        plip_dict = gen_intrns_dict(plip_itrns)
        data['interactions'] = set(plip_dict.keys())
        data['plip'] += plip_score(xstal_set, rmol_data.interactions) * 50 # HYPERPARAM TO CHANGE
        data['cnn_ic50_norm'] = data['cnnaffinityIC50'] / data['MW']
        print('cnn: ', data['cnnaffinity'])
        print('cnnic50: ', data['cnnaffinityIC50'])
        print('mw: ', data['MW'])
        print('cnn norm: ', data['cnn_ic50_norm'] / data['MW'])
        data['sf1'] = sf1(data)
        print('sf1 = ', data['sf1'])
        print('rmol_data.interactions: ', data['interactions'])
        print('rmol_data.plip: ', data['plip'])
        print(f'Task: Completed the molecule generation in {time.time() - t_start} seconds. ')
        return rmol, rmol_data








def get_saving_queue():
    # start a saving queue (just for Rocket, which struggles with saving file speed)
    mol_saving_queue = queue.Queue()

    def worker_saver():
        while True:
            mol = mol_saving_queue.get()
            filename = Chem.MolToSmiles(Chem.RemoveHs(mol))

            if os.path.exists(f'structures/{filename}.sdf'):
                print('Overwriting the existing file')

            with Chem.SDWriter(f'structures/{filename}.sdf') as SD:
                SD.write(mol)
                print(f'Wrote {filename}')

            mol_saving_queue.task_done()

    threading.Thread(target=worker_saver, daemon=False).start()
    return mol_saving_queue


if __name__ == '__main__':
    import mal
    from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
    config = get_gaussian_process_config()
    initial_chemical_space = "manual_init.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 100  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h', 'plip', 'sf1', 'enamine_id']
    config.model_config.targets.params.feature_column = 'sf1' # 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    pdb_load = open('rec_final.pdb').read()
    scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]

    t_now = datetime.datetime.now()
    client = Client(create_cluster())
    print('Client', client)

    al = mal.ActiveLearner(config)
    jobs = {}
    next_selection = None
    mol_saving_queue = get_saving_queue()
    while True:
        print(f"{datetime.datetime.now() - t_now}: Queue: {len(jobs)} tasks ")

        for job, args in list(jobs.items()):
            if not job.done():
                continue

            # get back the original arguments
            smiles = jobs[job]
            del jobs[job]

            try:
                rmol, rmol_data = job.result()
                rmol_data['cycle'] = al.cycle  # TODO change to dict
                [rmol.SetProp(k, str(v)) for k, v in rmol_data.items()] #adding values to rmoldata dict
                mol_saving_queue.put(rmol)


                feature_column = config.model_config.targets.params.feature_column
                # Dynamic attribute access
                feature_value = rmol_data[feature_column]
                print('FEATURE COLUMN: ', feature_column)
                print('FEATURE VALUE: ', feature_value)



            except Exception as E:
                print('ERROR: Will be ignored. Continuing the main loop. Error: ', E)
                feature_value = 0


            al.virtual_library.loc[al.virtual_library.Smiles == Chem.MolToSmiles(rmol),
            [feature_column, 'Training']] = float(feature_value), True



        if len(jobs) == 0:
            print(f'Iteration finished. Next.')



            # save the results from the previous iteration
            if next_selection is not None:
                for i, row in next_selection.iterrows():
                    # bookkeeping
                    next_selection.loc[next_selection.Smiles == row.Smiles, [feature_column, 'Training']] = \
                    al.virtual_library[al.virtual_library.Smiles == row.Smiles][feature_column].values[0], True

                al.csv_cycle_summary(next_selection)

            # cProfile.run('next_selection = al.get_next_best()', filename='get_next_best.prof', sort=True)
            next_selection = al.get_next_best()

            # select 20 random molecules
            for_submission = [evaluate(scaffold, row.h, row.Smiles, pdb_load) for _, row in next_selection.iterrows()]
            for job, (_, row) in zip(client.compute(for_submission), next_selection.iterrows()):
                jobs[job] = row.Smiles

        time.sleep(5)

    mol_saving_queue.join()

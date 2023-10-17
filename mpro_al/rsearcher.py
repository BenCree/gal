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

# get hardware specific cluster
try:
    from mycluster import create_cluster
except ImportError:
    # set up a local cluster just in case
    def create_cluster():
        from dask.distributed import LocalCluster
        lc = LocalCluster(processes=True, threads_per_worker=1, n_workers=4)
        lc.adapt(maximum_cores=28)
        return lc

# preload the dataframes
rgroups = list(fegrow.RGroupGrid._load_molecules().Mol.values)
linkers = list(fegrow.RLinkerGrid._load_molecules().Mol.values)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NoConformers(Exception):
    pass

def score(scaffold, h, smiles, pdb_load):
    t_start = time.time()
    fegrow.RMol.set_gnina(os.environ['FG_GNINA_PATH'])
    with (tempfile.TemporaryDirectory() as TMP):
        TMP = Path(TMP)
        os.chdir(TMP)
        print(f'TIME changed dir: {time.time() - t_start}')

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
        print(f'TIME prepped rmol: {time.time() - t_start}')

        rmol_data = helpers.Data()

        rmol.generate_conformers(num_conf=20, minimum_conf_rms=0.4)
        print('Number of simple conformers: ', rmol.GetNumConformers())

        rmol.remove_clashing_confs(protein)
        print(f'TIME conformers done: {time.time() - t_start}')
        print('schedulerNumber of conformers after removing clashes: ', rmol.GetNumConformers())

        rmol.optimise_in_receptor(
            receptor_file=protein,
            ligand_force_field="openff",
            use_ani=False,
            sigma_scale_factor=0.8,
            relative_permittivity=4,
            water_model=None,
            platform_name='CPU',
        )


        #plip taminoto score on rmol_data.plip, new rmol_data, .plip_score
        # continue only if there are any conformers to be optimised
        if rmol.GetNumConformers() == 0:
            rmol_data.cnnaffinity = 0.1


        print(f'TIME opt done: {time.time() - t_start}')

        rmol.sort_conformers(energy_range=2) # kcal/mol
        affinities = rmol.gnina(receptor_file=protein)
        rmol_data.cnnaffinity = -affinities.CNNaffinity.values[0]
        #rmol_data.cnnaffinity = -Descriptors.HeavyAtomMolWt(rmol) / 100
        rmol_data.cnnaffinityIC50 = affinities["CNNaffinity->IC50s"].values[0]
        #rmol_data.hydrogens = [atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomicNum() == 1]




        tox = rmol.toxicity()
        tox = dict(zip(list(tox.columns), list(tox.values[0])))
        tox['MW'] = Descriptors.HeavyAtomMolWt(rmol)
        for k, v in tox.items():
            setattr(rmol_data, k, v)

        # compute all props
        print(f'Calculating PLIP')

        plip_itrns = rmol.plip_interactions(receptor_file=protein)
        plip_dict = gen_intrns_dict(plip_itrns)
        rmol_data.interactions = set(plip_dict.keys())
        rmol_data.plip += plip_score(xstal_set, rmol_data.interactions) * 10 # HYPERPARAM TO CHANGE
        rmol_data.cnn_ic50_norm = rmol_data.cnnaffinityIC50 / rmol_data.MW
        print('cnn: ', rmol_data.cnnaffinity)
        print('cnnic50: ', rmol_data.cnnaffinityIC50)
        print('cnnic50norm: ', rmol_data.cnn_ic50_norm)
        rmol_data.sf1 = sf1(rmol_data)
        print('sf1 = ', rmol_data.sf1)
        print('rmol_data.interactions: ', rmol_data.interactions)
        print('rmol_data.plip: ', rmol_data.plip)
        print(f'Task: Completed the molecule generation in {time.time() - t_start} seconds. ')
        return rmol, rmol_data


def expand_chemical_space(al):
    """
    Expand the chemical space. This is another selection problem.
    Select the best performing "small", as the starting point ...
    """
    if al.virtual_library[al.virtual_library.Training == True].empty:
        return

    params = Chem.SmilesParserParams()
    params.removeHs = False
    for i, row in al.virtual_library[al.virtual_library.Training == True].iterrows():
        mol = Chem.MolFromSmiles(row.Smiles, params=params)
        Chem.AllChem.EmbedMolecule(mol)
        hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        smiles = build_smiles(mol, hs, rgroups)
        new_smiles = smiles.assign(Training=False)  # fixme: this will break if we use a different keyword

        extended = pd.concat([al.virtual_library, new_smiles], ignore_index=True)
        al.virtual_library = extended


def build_smiles(core, hs, groups):
    """
    Build a list of smiles that can be generated in theory
    """
    start = time.time()
    smiless = []
    hhooks = []
    for h in hs:
        for group in groups:
            for linker in linkers:
                core_linker = fegrow.build_molecules(core, linker, [h])[0]
                new_mol = fegrow.build_molecules(core_linker, group)[0]
                smiles = Chem.MolToSmiles(new_mol)
                smiless.append(smiles)
                hhooks.append(h)
    print('Generated initial smiles in: ', time.time() - start)
    return pd.DataFrame({'Smiles': smiless, 'h': hhooks})


def get_saving_queue():
    # start a saving queue (just for Rocket, which struggles with saving file speed)
    mol_saving_queue = queue.Queue()

    def worker_saver():
        while True:
            mol = mol_saving_queue.get()
            start = time.time()
            helpers.save(mol)
            print(f'Saved molecule in {time.time() - start}')
            mol_saving_queue.task_done()

    threading.Thread(target=worker_saver, daemon=False).start()
    return mol_saving_queue


if __name__ == '__main__':
    import mal
    from al_for_fep.configs.simple_greedy_gaussian_process import get_config as get_gaussian_process_config
    config = get_gaussian_process_config()
    initial_chemical_space = "manual_init.csv"
    config.virtual_library = initial_chemical_space
    config.selection_config.num_elements = 2  # how many new to select
    config.selection_config.selection_columns = ["cnnaffinity", "Smiles", 'h', 'plip', 'sf1']
    config.model_config.targets.params.feature_column = 'sf1' # 'cnnaffinity'
    config.model_config.features.params.fingerprint_size = 2048

    t_now = datetime.datetime.now()
    client = Client(create_cluster())
    print('Client', client)

    mol_saving_queue = get_saving_queue()

    # load the initial molecule
    scaffold = Chem.SDMolSupplier('coreh.sdf', removeHs=False)[0]

    # if not os.path.exists(initial_chemical_space):
    #     smiles = build_smiles(scaffold, [6], rgroups)
    #     smiles.to_csv(initial_chemical_space, index=False)

    al = mal.ActiveLearner(config)
    print('Initialised config')
    futures = {}

    pdb_load = open('rec_final.pdb').read()

    next_selection = None

    while True:
        for future, args in list(futures.items()):
            if not future.done():
                continue

            # get back the original arguments
            scaffold, h, smiles, _ = futures[future]
            del futures[future]

            try:
                rmol, rmol_data = future.result()

                helpers.set_properties(rmol, rmol_data)
                feature_column = config.model_config.targets.params.feature_column
                # Dynamic attribute access
                if hasattr(rmol_data, feature_column):
                    feature_value = getattr(rmol_data, feature_column)
                    print('FEATURE COLUMN: ', feature_column)
                    print('FEATURE VALUE: ', feature_value)
                else:
                    print(f"Warning: {feature_column} not found in rmol_data")
                    feature_value = None  # or some default value, or raise an exception

                mol_saving_queue.put(rmol)
                al.virtual_library.loc[al.virtual_library.Smiles == Chem.MolToSmiles(rmol),
                                       [feature_column, 'Training']] = float(feature_value), True
            except Exception as E:
                print('ERROR: Will be ignored. Continuing the main loop. Error: ', E)
                continue

            print(f"{datetime.datetime.now() - t_now}: Queue: {len(futures)} tasks. ")

            if len(futures) == 0:
                print(f'Iteration finished. Next.')

                # expand_chemical_space(al)

                # save the results from the previous iteration
                if next_selection is not None:
                    for i, row in next_selection.iterrows():
                        # bookkeeping
                        next_selection.loc[next_selection.Smiles == row.Smiles, [feature_column, 'Training']] = \
                        al.virtual_library[al.virtual_library.Smiles == row.Smiles][feature_column].values[0], True

                    al.csv_cycle_summary(next_selection)

                next_selection = al.get_next_best()

                # select 20 random molecules
                for i, row in next_selection.iterrows():
                    args = [scaffold, row.h, row.Smiles, pdb_load]
                    futures[client.compute([score(*args), ])[0]] = args

            time.sleep(5)

        mol_saving_queue.join()

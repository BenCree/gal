import copy
import time
import tempfile
import os
from pathlib import Path
import logging
import datetime
import dask
from dask.distributed import Client, performance_report
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import traceback


import fegrow


try:
    from mycluster import create_cluster
except ImportError:
    def create_cluster():
        from dask.distributed import LocalCluster
        return LocalCluster()

@dask.delayed
def evaluate(scaffold, h, smiles, protein, gnina_path):
    t_start = time.time()

    fegrow.RMol.set_gnina(gnina_path)

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

    rmol.generate_conformers(num_conf=50, minimum_conf_rms=0.2)
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
        "hydrogens": [atom.GetIdx() for atom in rmol.GetAtoms() if atom.GetAtomicNum() == 1],
    }

    print(f'TIME Completed the molecule generation in {time.time() - t_start:.1f}s.')
    return rmol, data


if __name__ == '__main__':
    protein = 'rec_final.pdb'
    scaffold = Chem.SDMolSupplier('5R83_core.sdf', removeHs=False)[0]
    scaffold_dask = dask.delayed(scaffold)
    gnina_path = dask.delayed(os.environ['GNINA'])

    client = Client(create_cluster())
    print('Client', client)

    # load the Smiles for pcoessing
    dataset = pd.read_csv('manual_init_h6_rgroups_linkers100.csv')

    # filter out already done jobs
    scored_dir = Path('scored')
    scored_ids = {int(path.stem) for path in scored_dir.glob('*.sdf')}

    # filter out finished cases
    dataset_to_score = dataset[~dataset.index.isin(scored_ids)]
    print(f"Processing {len(dataset_to_score)} after removing {len(scored_ids)} already scored. ")

    # submit
    for_submission = [evaluate(scaffold_dask, 6, smiles, protein, gnina_path)
                      for smiles in dataset_to_score.Smiles.values]
    submitted = client.compute(for_submission)

    # map ids: jobs
    jobs = dict(zip(submitted, [i for i in dataset_to_score.index]))

    while len(jobs) > 0:
        for job, index in list(jobs.items()):
            if not job.done():
                continue

            # remove the job
            del jobs[job]

            try:
                rmol, rmol_data = job.result()
                # recover the properties (they are not passed with serialisation)
                [rmol.SetProp(k, str(v)) for k, v in rmol_data.items()]

                with Chem.SDWriter(str(scored_dir / (str(index) + '.sdf'))) as SD:
                    SD.write(rmol)
                    print(f'Scored and wrote {index}')

            except Exception as E:
                print(f'ERROR when scoring a molecule {index}. Assigning a penalty. Error: ', E)
                traceback.print_exc()

        print(f'{datetime.datetime.now()}: Left {len(jobs)} to process. ')
        time.sleep(5)
    print('Quitting all finished')


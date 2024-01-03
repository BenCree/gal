"""
This is a script to be used after FEgrow was used to build the chemical space (dock and sccore).

Extract from .sdf files the scores into a pandas dataframe.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

cs = pd.read_csv('manual_init_h6_rgroups_linkers100.csv', usecols=["Smiles"])

cs["cnnaffinity"] = np.nan
cnnaffinity_col_index = cs.columns.get_loc('cnnaffinity')

for sdf in Path(".").glob("scored/*sdf"):
    sdf_id = int(sdf.stem)
    mol = Chem.SDMolSupplier(str(sdf))[0]
    sdf_Kd = float(mol.GetProp('cnnaffinity'))

    cs.iloc[sdf_id, cnnaffinity_col_index ] = sdf_Kd

cs.dropna(inplace=True)
cs.to_csv("cs_scored.csv")

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

"""Library for data handling in the Makita concentrated pipeline."""

import ast
import functools
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors


def parse_feature_smiles_morgan_fingerprint(
    feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
  """Parses SMILES strings into morgan fingerprints.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    fingerprint_radius: integer radius to use for Morgan fingerprint
      calculation.
    fingerprint_size: integer number of bits to use in the Morgan fingerprint.

  Returns:
    Array of Morgan fingerprints of the molecules represented by the given
    SMILES strings.
  """

  start = time.time()
  fingerprint_fn = functools.partial(
      AllChem.GetMorganFingerprintAsBitVect,
      radius=fingerprint_radius,
      nBits=fingerprint_size)

  from_scratch =  np.array([
      np.array(fingerprint_fn(Chem.MolFromSmiles(smiles)))
      for smiles in feature_dataframe[feature_column]
  ])
  print(f"Computed Fingerprints in {time.time() - start} for {len(feature_dataframe)} rows")
  return from_scratch

  # import fingerprints_db
  # fingerprints = []
  # fp_skeleton = ExplicitBitVect(2048)
  # for id, row in feature_dataframe.iterrows():
  #     smiles = row[feature_column]
  #     fp_base64 = fingerprints_db.get(id)
  #     if fp_base64 is None:
  #         fp = fingerprint_fn(Chem.MolFromSmiles(smiles))
  #         fp_base64 = fp.ToBase64()
  #         fingerprints_db.add(id, fp_base64)
  #         fingerprints.append(np.array(fp))
  #     else:
  #         fp_skeleton.FromBase64(fp_base64)
  #         fingerprints.append(np.array(fp_skeleton))
  # print(f"Computed Fingerprints in {time.time() - start} s")
  #
  # return np.array(fingerprints)


def parse_feature_smiles_rdkit_properties(
    feature_dataframe, feature_column, *args,
    **kwargs):
  """Computes RDKit Descriptor values for input features.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    *args:
    **kwargs: Extra arguments not used by this function. These need to be
      included for this parser to satisfy the generic parser interface.

  Returns:
    Array of rdkit descriptor values for the molecules described in the feature
      dataframe.
  """
  del args, kwargs  # Unused.
  start = time.time()
  descriptors = [name for name, _ in Chem.Descriptors.descList]
  calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
  import pandas as pd
  import numpy as np

  # Assuming your NumPy array is called 'data'
  data = np.array([
      np.array(calculator.CalcDescriptors(Chem.MolFromSmiles(smiles)))
      for smiles in feature_dataframe[feature_column]
  ])
  df = pd.DataFrame(data, columns=descriptors)

  positive_df = df.loc[:, (df >= 0).all()]

  positive_data = positive_df.to_numpy()

  # Specify the filename where you want to save the CSV file
  output_file = '/home/c0065492/code/gal/cs50k/output.csv'

  # Save the DataFrame as a CSV file
  positive_df.to_csv(output_file, index=False, header=False) 
  print(f"Computed Descriptors in {time.time() - start} for {len(feature_dataframe)} rows")
  return positive_data

def parse_feature_smiles_morgan_fingerprint_with_descriptors(
    feature_dataframe, feature_column,
    fingerprint_radius, fingerprint_size):
  """Generates input vectors with molecules' fingerprints and descriptor values.

  Args:
    feature_dataframe: pandas DataFrame with a column of SMILES data.
    feature_column: string column name of feature_series that holds SMILES data.
    fingerprint_radius: integer radius to use for Morgan fingerprint
      calculation.
    fingerprint_size: integer number of bits to use in the Morgan fingerprint.

  Returns:
    Two dimensional array holding a concatenation of the Morgan fingerprint and
    the rdkit descriptor values for each input SMILES.
  """
  start = time.time()
  fingerprint_features = parse_feature_smiles_morgan_fingerprint(
      feature_dataframe, feature_column, fingerprint_radius, fingerprint_size)
  descriptor_values = parse_feature_smiles_rdkit_properties(
      feature_dataframe, feature_column)

  print(f"Computed Descriptors in {time.time() - start} for {len(feature_dataframe)} rows")
  return np.concatenate([fingerprint_features, descriptor_values], axis=1)


def parse_feature_vectors(feature_dataframe,
                          feature_column):
  """Converts string feature vectors into numpy arrays for modeling.

  Args:
    feature_dataframe: pandas dataframe of shape (N, *) with a column containing
      full length feature vectors as strings.
    feature_column: string name of the column of feature_dataframe holding the
      feature vectors.

  Returns:
    A numpy array of the full input matrix of shape (N, len(feature vectors)).
  """
  return np.array([
      np.array(ast.literal_eval(samp))
      for samp in feature_dataframe[feature_column].to_numpy()
  ])


def parse_feature_numbers(feature_dataframe,
                          feature_column):
  """Parses a series of floats or float-like values.

  Anything that can be parsed as a float (e.g. a string) is valid here.

  Args:
    feature_dataframe: pandas series of shape (n_samples, *) containing a column
      holding float like values.
    feature_column: string column name holding float like features.

  Returns:
    A numpy array of the input series.
  """
  return np.array(list(map(float, feature_dataframe[feature_column])))

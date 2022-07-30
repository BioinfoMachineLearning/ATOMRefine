# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and utilities for the structure module."""

import functools
from typing import Dict
from amber import residue_constants
from amber import all_atom
from amber import utils_model
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

def find_structural_violations(
    batch: Dict[str, jnp.ndarray],
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    config: ml_collections.ConfigDict
    ):
  """Computes several checks for structural violations."""

  # Compute between residue backbone violations of bonds and angles.
  connection_violations = all_atom.between_residue_bond_loss(
      pred_atom_positions=atom14_pred_positions,
      pred_atom_mask=batch['atom14_atom_exists'].astype(jnp.float32),
      residue_index=batch['residue_index'].astype(jnp.float32),
      aatype=batch['aatype'],
      tolerance_factor_soft=config.violation_tolerance_factor,
      tolerance_factor_hard=config.violation_tolerance_factor)

  # Compute the Van der Waals radius for every atom
  # (the first letter of the atom name is the element type).
  # Shape: (N, 14).
  atomtype_radius = jnp.array([
      residue_constants.van_der_waals_radius[name[0]]
      for name in residue_constants.atom_types
  ])
  atom14_atom_radius = batch['atom14_atom_exists'] * utils_model.batched_gather(
      atomtype_radius, batch['residx_atom14_to_atom37'])

  # Compute the between residue clash loss.
  between_residue_clashes = all_atom.between_residue_clash_loss(
      atom14_pred_positions=atom14_pred_positions,
      atom14_atom_exists=batch['atom14_atom_exists'],
      atom14_atom_radius=atom14_atom_radius,
      residue_index=batch['residue_index'],
      overlap_tolerance_soft=config.clash_overlap_tolerance,
      overlap_tolerance_hard=config.clash_overlap_tolerance)

  # Compute all within-residue violations (clashes,
  # bond length and angle violations).
  restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
      overlap_tolerance=config.clash_overlap_tolerance,
      bond_length_tolerance_factor=config.violation_tolerance_factor)
  atom14_dists_lower_bound = utils_model.batched_gather(
      restype_atom14_bounds['lower_bound'], batch['aatype'])
  atom14_dists_upper_bound = utils_model.batched_gather(
      restype_atom14_bounds['upper_bound'], batch['aatype'])
  within_residue_violations = all_atom.within_residue_violations(
      atom14_pred_positions=atom14_pred_positions,
      atom14_atom_exists=batch['atom14_atom_exists'],
      atom14_dists_lower_bound=atom14_dists_lower_bound,
      atom14_dists_upper_bound=atom14_dists_upper_bound,
      tighten_bounds_for_loss=0.0)

  # Combine them to a single per-residue violation mask (used later for LDDT).
  per_residue_violations_mask = jnp.max(jnp.stack([
      connection_violations['per_residue_violation_mask'],
      jnp.max(between_residue_clashes['per_atom_clash_mask'], axis=-1),
      jnp.max(within_residue_violations['per_atom_violations'],
              axis=-1)]), axis=0)

  return {
      'between_residues': {
          'bonds_c_n_loss_mean':
              connection_violations['c_n_loss_mean'],  # ()
          'angles_ca_c_n_loss_mean':
              connection_violations['ca_c_n_loss_mean'],  # ()
          'angles_c_n_ca_loss_mean':
              connection_violations['c_n_ca_loss_mean'],  # ()
          'connections_per_residue_loss_sum':
              connection_violations['per_residue_loss_sum'],  # (N)
          'connections_per_residue_violation_mask':
              connection_violations['per_residue_violation_mask'],  # (N)
          'clashes_mean_loss':
              between_residue_clashes['mean_loss'],  # ()
          'clashes_per_atom_loss_sum':
              between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
          'clashes_per_atom_clash_mask':
              between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
      },
      'within_residues': {
          'per_atom_loss_sum':
              within_residue_violations['per_atom_loss_sum'],  # (N, 14)
          'per_atom_violations':
              within_residue_violations['per_atom_violations'],  # (N, 14),
      },
      'total_per_residue_violations_mask':
          per_residue_violations_mask,  # (N)
  }


def compute_violation_metrics(
    batch: Dict[str, jnp.ndarray],
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    violations: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
  """Compute several metrics to assess the structural violations."""

  ret = {}
  extreme_ca_ca_violations = all_atom.extreme_ca_ca_distance_violations(
      pred_atom_positions=atom14_pred_positions,
      pred_atom_mask=batch['atom14_atom_exists'].astype(jnp.float32),
      residue_index=batch['residue_index'].astype(jnp.float32))
  ret['violations_extreme_ca_ca_distance'] = extreme_ca_ca_violations
  ret['violations_between_residue_bond'] = utils_model.mask_mean(
      mask=batch['seq_mask'],
      value=violations['between_residues'][
          'connections_per_residue_violation_mask'])
  ret['violations_between_residue_clash'] = utils_model.mask_mean(
      mask=batch['seq_mask'],
      value=jnp.max(
          violations['between_residues']['clashes_per_atom_clash_mask'],
          axis=-1))
  ret['violations_within_residue'] = utils_model.mask_mean(
      mask=batch['seq_mask'],
      value=jnp.max(
          violations['within_residues']['per_atom_violations'], axis=-1))
  ret['violations_per_residue'] = utils_model.mask_mean(
      mask=batch['seq_mask'],
      value=violations['total_per_residue_violations_mask'])
  return ret

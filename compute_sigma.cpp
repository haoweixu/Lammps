/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Haowei Xu (MIT)
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "compute_sigma.h"
#include "fix_store.h"
#include "atom.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "pair.h"
#include "pair_poly_lj_cut.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

ComputeSigma::ComputeSigma(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg) 
{
    
  peratom_flag = 1;
  size_peratom_cols = 0;
  
}


ComputeSigma::~ComputeSigma()
{
}

void ComputeSigma::init()
{
}

void ComputeSigma::compute_peratom()
{
  int nlocal = atom->nlocal;
  memory->destroy(vector_atom);
  memory->create(vector_atom, nlocal, "Sigma:vector_atom");
  
	tagint *tag = atom->tag;
  tagint id;
	PairPolyLJCut *pair = (PairPolyLJCut *) force->pair;
  double **sigma = pair->sigma;
  for (int i=0; i<nlocal; i++) {
    id = tag[i]; // note that sigma has length ntype+1
    vector_atom[i] = sigma[id][id];
  }
  
}

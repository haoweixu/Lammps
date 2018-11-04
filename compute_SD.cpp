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
#include "compute_SD.h"
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
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

ComputeSD::ComputeSD(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg) 
{
  if (narg < 3) error->all(FLERR,"Illegal compute SD command. Compute 1 all SD");
    
  peratom_flag = 1;
  size_peratom_cols = 0;
  
//  int *tag = atom->tag;
  
  int n = strlen(id) + strlen("_COMPUTE_STORE") + 1;
  id_fix = new char[n];
  strcpy(id_fix,id);
  strcat(id_fix,"_COMPUTE_STORE");

  char **newarg = new char*[6];
  newarg[0] = id_fix;
  newarg[1] = group->names[igroup];
  newarg[2] = (char *) "STORE";
  newarg[3] = (char *) "peratom";
  newarg[4] = (char *) "1";
  newarg[5] = (char *) "3";
  modify->add_fix(6,newarg);
  fix = (FixStore *) modify->fix[modify->nfix-1];
  delete [] newarg;

  // calculate xu,yu,zu for fix store array
  // skip if reset from restart file

  if (fix->restart_reset) fix->restart_reset = 0;
  else {
    double **xoriginal = fix->astore;

    double **x = atom->x;
    int *mask = atom->mask;
    imageint *image = atom->image;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) domain->unmap(x[i],image[i],xoriginal[i]);
      else xoriginal[i][0] = xoriginal[i][1] = xoriginal[i][2] = 0.0;
  }
}


ComputeSD::~ComputeSD()
{
  if (modify->nfix) modify->delete_fix(id_fix);
  delete [] id_fix;
}

void ComputeSD::init()
{
  // set fix which stores reference atom coords
  int ifix = modify->find_fix(id_fix);
  if (ifix < 0) error->all(FLERR,"Could not find compute msd fix ID");
  fix = (FixStore *) modify->fix[ifix];  
  
}

void ComputeSD::compute_peratom()
{
  int nlocal = atom->nlocal;
  memory->destroy(vector_atom);
  memory->create(vector_atom, nlocal, "SD:vector_atom");
  double **x = atom->x;  
  tagint *image = atom->image;
  int *mask = atom->mask;
	double unwrap[3];
  double delx, dely, delz;
  double **xoriginal = fix->astore;
  
  for (int i=0; i<nlocal; i++){
    if (mask[i]&groupbit) {
      domain->unmap(x[i], image[i], unwrap);
      delx = unwrap[0] - xoriginal[i][0];
      dely = unwrap[1] - xoriginal[i][1];
      delz = unwrap[2] - xoriginal[i][2];
      vector_atom[i] = delx*delx + dely*dely + delz*delz;
    }
    else
      vector_atom[i] = 0;
  }
}

void ComputeSD::set_arrays(int i) {
	double **xoriginal = fix->astore;
  double **x = atom->x;
  xoriginal[i][0] = x[i][0];
  xoriginal[i][1] = x[i][1];
  xoriginal[i][2] = x[i][2];  
}

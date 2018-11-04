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

#include <string.h>
#include <stdlib.h>
#include "fix_adjust_timestep.h"
#include "atom.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixAdjustTimeStep::FixAdjustTimeStep(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal fix addforce command.");

  global_freq = 1;
  
  kt = atof(arg[3]);
  Et = atof(arg[4]);
  inc = atof(arg[5]);
  
}

/* ---------------------------------------------------------------------- */

FixAdjustTimeStep::~FixAdjustTimeStep()
{
  
}

/* ---------------------------------------------------------------------- */

int FixAdjustTimeStep::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAdjustTimeStep::init()
{
  
}

/* ---------------------------------------------------------------------- */

void FixAdjustTimeStep::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdjustTimeStep::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAdjustTimeStep::post_force(int vflag)
{
  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
    
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
 
  double max_v = 0.0, this_v, this_f;
  double max_fv = 0.0, this_fv;
  
  double max_v_glb, max_fv_glb;
  
  for (int i=0; i<nlocal; i++) {
    this_v = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
    this_f = f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2];
    this_fv = this_v * this_f;
    
    max_v = this_v > max_v ? this_v : max_v;
    max_fv = this_fv > max_fv ? this_fv : max_fv;
  }
 
  MPI_Allreduce(&max_v, &max_v_glb, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&max_fv, &max_fv_glb, 1, MPI_DOUBLE, MPI_MAX, world);
  
  double old_dt = inc*update->dt;
  double kt_v = kt / sqrt(max_v_glb);
  double Et_fv = Et / sqrt(max_fv_glb);
  
  old_dt = old_dt < kt_v ? old_dt : kt_v;
  old_dt = old_dt < Et_fv ? old_dt : Et_fv;
  
  update->dt = old_dt;
  if (comm->me == 0)
	  cout << "reset time step to " << update->dt << "\n"; 
}

/* ---------------------------------------------------------------------- */

void FixAdjustTimeStep::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixAdjustTimeStep::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

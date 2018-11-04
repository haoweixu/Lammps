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
   Contributing author:  Haowei Xu (MIT)
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "compute_phase_seperation_global.h"
#include "compute_phase_seperation.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"
#include <iostream>
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

/* ---------------------------------------------------------------------- */

ComputePhaseSeperationGlobal::ComputePhaseSeperationGlobal(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal compute phase_sep_global command: compute 1 all phase_sep_global phase_sep_id return_type");

  int n = strlen(arg[3]) + 1;
  ps_id = new char[n];
  strcpy(ps_id, arg[3]);
  
	n = strlen(arg[4]) + 1;
  return_type = new char[n];
  strcpy(return_type, arg[4]);
    
  scalar_flag = 1;
}

/* ---------------------------------------------------------------------- */

ComputePhaseSeperationGlobal::~ComputePhaseSeperationGlobal()
{
  delete [] ps_id;
}

/* ---------------------------------------------------------------------- */

void ComputePhaseSeperationGlobal::init()
{
  int icompute = modify->find_compute(ps_id);
  if (icompute < 0) 
    error->all(FLERR, "compute phase_sep does not exist");
  ps = (ComputePhaseSeperation *)modify->compute[icompute];
}

/* ---------------------------------------------------------------------- */

double ComputePhaseSeperationGlobal::compute_scalar()
{
  if (!ps->invoked_flag)
    ps->compute_peratom();
	
	if (strcmp(return_type, "global_ps") == 0)  
 		scalar = ps->global_ps;
  if (strcmp(return_type, "global_ps_num") == 0)
		scalar = (double)ps->global_ps_num;

  return scalar;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */
double ComputePhaseSeperationGlobal::memory_usage()
{
  double bytes = 0;
  return bytes;
}

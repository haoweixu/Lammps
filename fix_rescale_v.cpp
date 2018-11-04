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
#include "fix_rescale_v.h"
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

FixRescaleV::FixRescaleV(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal fix rescale_v command. Syntax: fix fixID groupID rescale_v freq scale");

  global_freq = 1;
  
  freq = force->inumeric(FLERR, arg[3]);
  scale = force->numeric(FLERR, arg[4]); 
}

/* ---------------------------------------------------------------------- */

FixRescaleV::~FixRescaleV()
{
  
}

/* ---------------------------------------------------------------------- */

int FixRescaleV::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRescaleV::init()
{
}

/* ---------------------------------------------------------------------- */

void FixRescaleV::setup(int vflag)
{
}

/* ---------------------------------------------------------------------- */

void FixRescaleV::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixRescaleV::post_force(int vflag)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  if (update->ntimestep%freq != 0) 
		return;
	
	if (comm->me==0)
		cout << v[1][0] << "\n";

  for (int i=0; i<nlocal; ++i) {
    v[i][0] *= scale;
    v[i][1] *= scale;
    v[i][2] *= scale;
  }
	if (comm->me==0)
		cout << v[1][0] << "\n";
}

/* ---------------------------------------------------------------------- */

void FixRescaleV::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixRescaleV::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

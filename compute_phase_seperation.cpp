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

ComputePhaseSeperation::ComputePhaseSeperation(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal compute phase_sep command: compute 1 all phase_sep");

  rcut = -1;
  if (force->pair != NULL)
    rcut = force->pair->cutforce;
  
  nevery = 1;
  thres = -1;
  
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nevery") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR, "Illegal compute phase_sep command");
      else 
        nevery = force->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "rcut") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR, "Illegal compute phase_sep command");
      else 
        rcut = force->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "thres") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR, "Illegal compute phase_sep command");
      else 
        thres = force->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    }
  }
 
//	cout << "rcut " << rcut << endl;
 
  if (rcut < -1)
    error->all(FLERR, "r_cut should be positive."); 
    
  peratom_flag = 1;
  size_peratom_cols = 0;
}

/* ---------------------------------------------------------------------- */

ComputePhaseSeperation::~ComputePhaseSeperation()
{
  delete [] itype_frac;
}

/* ---------------------------------------------------------------------- */

void ComputePhaseSeperation::init()
{
  int ntypes = atom->ntypes;
  int nlocal = atom->nlocal;
  double natoms = (double)atom->natoms;
  int *type = atom->type;
  itype_frac = new double [ntypes+1];
  int *n_itype_local = new int [ntypes+1];
  int *n_itype_global = new int [ntypes+1];
  
	for (int i=0; i<=ntypes; i++)
		n_itype_local[i] = 0;

  // note that type index starts from 1
  for (int i=0; i<nlocal; ++i)
    n_itype_local[ type[i] ] ++;
  
  MPI_Allreduce(n_itype_local, n_itype_global, ntypes+1, MPI_INT, MPI_SUM, world);
  
  for (int i=1; i<=ntypes; ++i)
    itype_frac[i] = (double)n_itype_global[i] / natoms;
  
//  if (comm->me == 0)
//    for (int i=1; i<= ntypes; ++i)
//      cout << "fraction of type " << i << " is " << itype_frac[i] << endl;
  
  // need an occasional full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
 
  if (force->pair!=NULL && rcut>force->pair->cutforce) {
    neighbor->requests[irequest]->cut = 1;
    neighbor->requests[irequest]->cutoff = rcut;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePhaseSeperation::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePhaseSeperation::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (invoked_peratom%nevery != 0)
    return;

  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  double cutsq = rcut*rcut;
  int *ilist,*jlist,*numneigh,**firstneigh;

//	cout << "cutsq " << cutsq << endl;

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int nlocal=atom->nlocal, nmax=atom->nmax, natoms=atom->natoms;
  double **x = atom->x;
  int *mask = atom->mask;
  int ntypes = atom->ntypes;
  int *type = atom->type;
  int *neigh_itype = new int [ntypes+1];
  int neigh_total;
  double phase_seperation, this_itype_frac, local_ps=0;
  int local_ps_num=0;
  
  memory->destroy(vector_atom);
  memory->create(vector_atom, nmax, "phase_sep:vector_atom");

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      for (int itype=0; itype<=ntypes; itype++)
        neigh_itype[itype] = 0;
      neigh_total = 0;

      // loop over list of all neighbors within force cutoff
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq) {
          neigh_total ++;
          neigh_itype[ type[j] ] ++;
        }
      }
      
      phase_seperation = 0;
			
      for (int itype=1; itype<=ntypes; itype++) {
        this_itype_frac = (double)neigh_itype[itype]/(double)neigh_total - itype_frac[itype];
//				cout << "this_itype_frac" << this_itype_frac << endl;

//				cout << "neigh_itype " << itype << '\t' << neigh_itype[itype] << endl;
        phase_seperation += this_itype_frac*this_itype_frac;
      }
      vector_atom[i] = phase_seperation;
      if (vector_atom[i] > thres)
        local_ps_num ++;
        
    }
    local_ps += vector_atom[i];
  }
  
  MPI_Allreduce(&local_ps, &global_ps, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&local_ps_num, &global_ps_num, 1, MPI_INT, MPI_SUM, world);
  global_ps /= natoms;
  
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */
double ComputePhaseSeperation::memory_usage()
{
  double bytes = 0;
  return bytes;
}

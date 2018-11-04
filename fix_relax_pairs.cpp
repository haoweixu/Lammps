/*-------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*---------------------------------------------------------------------------
   Contributing Author: Haowei Xu (MIT)
   
   Some pair potentials, like Buckingham potential, become infinitely attractive
   when two atoms are too close to each other, which leads to the so called
   "fusion" problem.
   
   This Fix class fixes this problem by simply checking the distance of each pair
   before every MD step and separate them if they are closer than a prescribed 
   distance, which is usually the maximum of the potential barrier.

	 This method does not help at all...

----------------------------------------------------------------------------*/

#include <math.h>
#include <cmath>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "fix_relax_pairs.h"
#include "atom.h"
#include "domain.h"
#include "update.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "kspace.h"
#include "modify.h"
#include "compute.h"
#include "error.h"
#include "math_extra.h"
#include "irregular.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

FixPairRelax::FixPairRelax(LAMMPS *lmp, int narg, char **arg) :
list(NULL), 
Fix(lmp, narg, arg) 
{ 
  int i, j;
  double disttmp;
 
//	force_reneighbor = 1;	
//	restart_pbc = 1;

	 ntypes = atom->ntypes;
//	if (comm->me == 0)	
//		cout << ntypes << endl;
  int nn = ntypes*(ntypes+1)/2;
  if (narg < nn+4) error->all(FLERR, "Illegal fix pair/relax command. Syntax: Fix ID all pair/relax epsilon min_dist_array.");
  
  epsilon = 1 + atof(arg[3]);
 
  dist_after_mv = new double *[ntypes];
  min_dist_sqr = new double *[ntypes];
  
  int index = 4;

	for (i=0; i<ntypes; i++) {
    dist_after_mv[i] = new double [ntypes];
    min_dist_sqr[i] = new double [ntypes];
	}

  for (i=0; i<ntypes; i++){ 
    for (j=i; j<ntypes; j++) {
      disttmp = atof(arg[index]);
      dist_after_mv[i][j] = epsilon*disttmp;
			dist_after_mv[j][i] = epsilon*disttmp;
      min_dist_sqr[i][j] = disttmp*disttmp;
			min_dist_sqr[j][i] = disttmp*disttmp;

      index += 1;
    }
	}
   
  mv_count = 0;
  if (comm->me == 0)
		cout << "constructor ends" << endl;   
}

FixPairRelax::~FixPairRelax()
{
  for(int i=0; i<ntypes; i++) {
    delete [] dist_after_mv[i];
    delete [] min_dist_sqr[i];
  }
}

void FixPairRelax::init() {
  // need a half neighbor list
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
	neighbor->requests[irequest]->occasional = 1;
	if (comm->me == 0)
		cout << "init_init" << endl;

}

void FixPairRelax::init_list(int id, NeighList *ptr)
{
  list = ptr;
	if (comm->me == 0)
		cout << "init_list" << endl;
}

int FixPairRelax::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= MIN_PRE_FORCE;
	return mask;
}

/*void FixPairRelax::setup(int vflag)
{
//  pre_force(vflag);
 	if (comm->me == 0)
		cout << "pair/relax setup" << endl;
}*/

void FixPairRelax::min_setup(int vflag)
{
//  pre_force(vflag);
}

void FixPairRelax::pre_force(int vflag)
{

	if (comm->me == 0)
		cout << "enter post_integrate" << endl;
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double rsq, r, xi, yi, zi, xj, yj, zj, distijsq, ratio;
  double delx, dely, delz, midx, midy, midz;
  int *ilist, *jlist, *numneigh, **firstneigh;
  bigint natoms;
  
  int mv_flag = 0;
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;

 	inum = atom->nlocal;
	if (list=NULL)
		neighbor->build_one(list, 1);
	list = neighbor->lists[0];
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
 
  for (ii=0; ii<inum; ii++){
    i = ilist[ii];
   	xi = x[i][0];
    yi = x[i][1];
    zi = x[i][2];
    itype = type[i];
 	  jlist = firstneigh[i];
	  jnum = numneigh[i];
//		if (comm->me == 0)
//			cout << "enter jj itype " << itype << " jnum" << jnum << endl;
    for (jj=0; jj<jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
			j = jj;
      jtype = type[j];

      xj = x[j][0];
      yj = x[j][1];
      zj = x[j][2];
      
      delx = xi - xj;
      dely = yi - yj;
      delz = zi - zj;
      rsq = delx*delx + dely*dely + delz*delz;
      r = sqrt(rsq);
      
      midx = (xi + xj) / 2.0;
      midy = (yi + yj) / 2.0;
      midz = (zi + zj) / 2.0;

      if (rsq <= min_dist_sqr[itype-1][jtype-1]) {
        mv_count += 1;
        mv_flag = 1;
        ratio = dist_after_mv[itype-1][jtype-1] / r;
        
        // new position 
        x[i][0] = delx*ratio/2.0 + midx;
        x[i][1] = dely*ratio/2.0 + midy;
        x[i][2] = delz*ratio/2.0 + midz;
       
				if (j < nlocal) {
					x[j][0] = -delx*ratio/2.0 + midx;
       		x[j][1] = -dely*ratio/2.0 + midy;
        	x[j][2] = -delz*ratio/2.0 + midz;
				}
 
        if (comm->me == 0)
          cout << "pair/relax called for " << mv_count << " times at " <<update->ntimestep << endl;
			} 
    } 
  }

  
  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance
	
tagint *image = atom->image;
  for (i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

//	if (comm->me == 0) 
//		cout << "after_remap" << endl;
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->reset_box();
  Irregular *irregular = new Irregular(lmp);
  irregular->migrate_atoms();
  delete irregular;
  if (domain->triclinic) domain->lamda2x(atom->nlocal);

  // check if any atoms were lost

  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (natoms != atom->natoms && comm->me == 0) {
    char str[128];
    sprintf(str,"Lost atoms via DA: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }
}

void FixPairRelax::min_pre_force(int vflag){
  pre_force(vflag);
}

double FixPairRelax::memory_usage() {
  return 0.0;
}


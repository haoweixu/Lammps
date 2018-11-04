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
   Contributing author: Yunwei Mao (XJTU) Haowei Xu (MIT)
   
   DA movements that lead to too short distance between atoms are rejected
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_DA_check.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "input.h"
#include "irregular.h"
#include "variable.h"
#include "random_mars.h"
#include "random_park.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

#define V_MAX 1.0
#define SMALL -1e20
#define LARGE  1e20
#define INVOKED_PERATOM 8
/* ---------------------------------------------------------------------- */

DA_check::DA_check(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 9) error->all(FLERR,"Illegal fix DA_check command. Fix 3 all DA_check id_pe NumALL xratio yratio zratio dist_thres seed");
  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int n = strlen(arg[3]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[3]);
  NUM = ATOBIGINT(arg[4]);
  // NUM determines how many atoms should be moved in one step.
  xratio = atof(arg[5]);
  yratio = atof(arg[6]);
  zratio = atof(arg[7]);
  dist_thres = atof(arg[8]);
  dist_thres = dist_thres*dist_thres;
  seed = atoi(arg[9]);
  
  nmax = 0;

  if (NUM < 0.0){
	if (comm->me == 0)
           error->warning(FLERR,"Set Num=0");
	NUM=0;
   }
  if (xratio<0.0||yratio<0.0||zratio<0.0){
        if (comm->me == 0)
           error->warning(FLERR,"Set ratio=0.0");
        xratio = 0;
	yratio = 0;
	zratio = 0;
  }
  if (seed<0.0){
        if (comm->me == 0)
           error->warning(FLERR,"Set seed=0");
        seed=0;   
  }
  array_buf=NULL;
}

/* ---------------------------------------------------------------------- */

DA_check::~DA_check()
{
  delete []id_pe;
  memory->destroy(array_buf);
}

/* ---------------------------------------------------------------------- */

int DA_check::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DA_check::init()
{
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA_check does not exist");
  pe = modify->compute[icompute];
	int irequest = neighbor->request(this, instance_me);
	neighbor->requests[irequest]->pair = 0;
	neighbor->requests[irequest]->fix = 1;
	neighbor->requests[irequest]->occasional = 1; 
}

/* ---------------------------------------------------------------------- */

void DA_check::setup(int vflag)
{
  post_force(vflag);
}

void DA_check::min_setup(int vflag)
{
  post_force(vflag);
}

void DA_check::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void DA_check::post_force(int vflag)
{
  int i, count;
  double ddx, ddy, ddz, xold, yold, zold, delx, dely, delz, dist_ij;
  
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;


  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;
  
  // create a full array & buf if has not been created

  if (array_buf == NULL) memory->create(array_buf,natoms,"DA_check:array_buf");

  int *recvcounts,*displs;

  // gather the tag & property DA_checkta info of all the atoms on each proc

  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);
  // nlocal on each proc is gathered and distributed to all procs, stored in recvcounts. 
  
  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
  
  delete [] recvcounts;
  delete [] displs;

  sort(array_buf,0,natoms-1);

  PENUM = array_buf[natoms-NUM-1];

  neighbor->build_one(list, 1);
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *jlist;
  int j, jnum, ii, jj, check;
  
  //DA_check.
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;

  RanPark *random = new RanPark(lmp,1);
  double dx = domain->xprd*xratio;
  double dy = domain->yprd*yratio;
  double dz = domain->zprd*zratio;
  //if (comm->me == 0 && screen) fprintf(screen,"%f%f%f\n", dx, dy, dz);
  
  for (ii=0; ii<inum; ii++) {
    i = ilist[ii];
    if ( ! (mask[i] & groupbit) ) continue;
    
    if (PE[i] > PENUM) {
      xold = x[i][0];
      yold = x[i][1];
      zold = x[i][2];
      
      count = 1;
      jlist = firstneigh[i];
      jnum = numneigh[i];
      
      while (true) {
        check = 1;
        random->reset(count*seed, x[i]);
        ddx = dx *(random->uniform()-0.5);
        ddy = dy *(random->uniform()-0.5);
        ddz = dz *(random->uniform()-0.5);
        x[i][0] = xold + ddx;
        x[i][1] = yold + ddy;
        x[i][2] = zold + ddz;
        
        // check distance with neighbors. In most cases, the distance moved
        // is not large. So we don't need to build a new neighbor list
        for (jj=0; jj<jnum; jj++){
          j = jlist[jj];
          j &= NEIGHMASK;
          if (!(mask[j] & groupbit)) continue;
          delx = x[i][0] - x[j][0];
          dely = x[i][1] - x[j][1];
          delz = x[i][2] - x[j][2];
          dist_ij = delx*delx + dely*dely + delz*delz;
          if (dist_ij < dist_thres){
//						if (comm->me == 0)
//							cout << i << "\t" << j << "\t" << dist_ij << "\t" << dist_thres << endl;
            check = 0;
            break;
          } 
        }
        
        count += 1;
				if (count >= 20){
					cout << "leaving DA from processor " << comm->me << endl;
					break;
				}       
 
        if (check == 1)
          break;
      }
    }
  }
  
  delete random;

  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance
  
  // remap() and pbc() are both methods in class Domain. see domain.cpp

  tagint *image = atom->image;
  for (i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->reset_box();
  Irregular *irregular = new Irregular(lmp);
  irregular->migrate_atoms();
  // important for atom exchange between procs. ----Haowei Xu----
  
  delete irregular;
  if (domain->triclinic) domain->lamda2x(atom->nlocal);

  // check if any atoms were lost

  bigint nblocal = atom->nlocal;
  MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  if (natoms != atom->natoms && comm->me == 0) {
    char str[128];
    sprintf(str,"Lost atoms via DA_check: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }

}

void DA_check::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DA_check::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DA_check::sort(double * a_one, bigint low, bigint high)
{
  bigint i, j;
  double t;
  if(low<high)
  {
    i=low;j=high; t = a_one[low];
    while (i<j)
    {
      while(i<j&&a_one[j]>t)
        j--;
      if (i<j)
      {
          a_one[i]=a_one[j];
          i++;
      }
      while(i<j&&a_one[i]<=t)
          i++;
      if(i<j)
      {
          a_one[j]=a_one[i];
          j--;
      }
    }
    a_one[i]=t;
    sort(a_one,low, i-1);
    sort(a_one, i+1, high);
  }
}

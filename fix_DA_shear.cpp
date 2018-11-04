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
   Contributing author: Yunwei Mao (XJTU)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_DA_shear.h"
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
#include "input.h"
#include "irregular.h"
#include "variable.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
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

DA_shear::DA_shear(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int ntypes = atom->ntypes; 
  
  if ( narg < 9 ) error->all(FLERR,"Illegal fix DA_shear command. Fix 3 all DA_shear id_pe Num1 xratio yratio zratio seed");
  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int count = 3;
  int n = strlen(arg[count]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[count]);
  count += 1;
  
  NUM = atoi(arg[count]);
  count += 1;
    
  dx = domain->xprd*atof(arg[count]);
  count += 1;
  
  dy = domain->yprd*atof(arg[count]);
  count += 1;
  
  dz = domain->zprd*atof(arg[count]);
  count += 1;
  
  seed = atoi(arg[count]);
  count += 1;
	  
  if (seed<0.0){
    if (comm->me == 0)
      error->warning(FLERR,"Set seed=0");
    seed=0;   
  }


}

/* ---------------------------------------------------------------------- */

DA_shear::~DA_shear()
{
  delete [] id_pe;
}


void DA_shear::init_list(int id, NeighList *ptr)
{
  list = ptr;
}


/* ---------------------------------------------------------------------- */

int DA_shear::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DA_shear::init()
{
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA_shear does not exist");
  pe = modify->compute[icompute];
  int irequest = neighbor->request(this, instance_me);
	neighbor->requests[irequest]->pair = 0;
	neighbor->requests[irequest]->fix = 1;
	neighbor->requests[irequest]->occasional = 1;   
}

/* ---------------------------------------------------------------------- */

void DA_shear::setup(int vflag)
{
  post_force(vflag);
}

void DA_shear::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void DA_shear::post_force(int vflag)
{
  int i, itype;
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
  int *type = atom->type;
  int ntypes = atom->ntypes;
  double **x = atom->x;
  int *mask = atom->mask;
  int *tag = atom->tag;

  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;

	double *array_buf = new double [natoms];  

  int *recvcounts,*displs;
  
  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }

  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 

  sort(array_buf,0,natoms-1);  
  PENUM = array_buf[natoms-NUM-1];
    
  neighbor->build_one(list, 1);
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *jlist;
  int j, jnum, ii, jj, count_local;
  
  count_local = 0;

  for (ii=0; ii<inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    if (PE[i]>PENUM) {
      for (jj=0; jj<jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;
        count_local ++;
        centers.push_back(i);
        neighs.push_back(tag[j]);
      }
		}
  }
  
  vector<int>::iterator c_iter, n_iter;
  c_iter = centers.begin(); n_iter = neighs.begin();
  
  double * send_buff = new double [4*count_local+1];
  int cur_center, cur_neigh;
  for (i=0; i<count_local; i++) {
    cur_center = *c_iter;
    cur_neigh = *n_iter;
    send_buff[4*i]    = (double) cur_neigh;
    send_buff[4*i+1]  = x[cur_center][0];
    send_buff[4*i+2]  = x[cur_center][1];
    send_buff[4*i+3]  = x[cur_center][2];
    
		c_iter ++;
    n_iter ++;
  }
  
  MPI_Allgather(&count_local, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  int count_global = 0;
  displs[0] = 0;
  for (int iproc = 0; iproc<nprocs; iproc++) {
    count_global += recvcounts[iproc];
    recvcounts[iproc] *= 4;
  }
  for (int iproc = 1; iproc<nprocs; iproc++)
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  
  double *recv_buff = new double [4*count_global];
  MPI_Allgatherv(send_buff,4*count_local,MPI_DOUBLE,recv_buff,recvcounts,displs,MPI_DOUBLE,world); 
 
  int * mv_tag = new int [count_global];
	double ** center_pos = new double *[count_global];
  for (i=0; i<count_global; i++) {
    mv_tag[i] = (int) recv_buff[4*i];
    center_pos[i] = new double [3];
    center_pos[i][0] = recv_buff[4*i+1];
    center_pos[i][1] = recv_buff[4*i+2];
    center_pos[i][2] = recv_buff[4*i+3];
  }
  
  RanPark *random = new RanPark(lmp,1);
  
  int itag, index;
	double xcenter, ycenter, zcenter;  
  for (i = 0; i < nlocal; i++) {
 	  itag = tag[i];
    index = find_index(itag, mv_tag, count_global);
    if (index < 0)
      continue;
    xcenter = center_pos[index][0];
    ycenter = center_pos[index][1];
    zcenter = center_pos[index][2];
  
    // center position is not used currently
    // still a translational move;
    random->reset(seed,x[i]);
    double ddx = dx *(random->uniform()-0.5);
    double ddy = dy *(random->uniform()-0.5);
    double ddz = dz *(random->uniform()-0.5);
    x[i][0] += ddx;
    x[i][1] += ddy;
    x[i][2] += ddz;	

  }

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
    sprintf(str,"Lost atoms via DA_shear: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }
  
  delete random;
  delete [] recvcounts;
	delete [] array_buf;
  delete [] displs;
  delete [] mv_tag;
  delete [] send_buff;
  delete [] recv_buff;
  for (i=0; i<count_global; i++)
    delete [] center_pos[i];
}

void DA_shear::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DA_shear::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DA_shear::sort(double * a_one, bigint low, bigint high)
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

int DA_shear::find_index(int tag, int *array, int array_size)
{
  for (int i=0; i<array_size; i++) {
    if (tag == array[i])
      return i;
  }
  return -1;
}

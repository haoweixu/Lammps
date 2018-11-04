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
#include "fix_DA_type.h"
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

DA_type::DA_type(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  int ntypes = atom->ntypes; 
  
  if ( narg < (5+4*ntypes) ) error->all(FLERR,"Illegal fix DA_type command. Fix 3 all DA_type id_pe Num1 xratio1 yratio1 zratio1 Num2 xratio2 yratio2 zratio2... seed");
  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int count = 3;
  int n = strlen(arg[count]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[count]);
  count += 1;
  
  NUM = new int[ntypes];
  dx = new double[ntypes];
  dy = new double[ntypes];
  dz = new double[ntypes];
  
  for(int i=0; i<ntypes; i++) {
    NUM[i] = atoi(arg[count]);
    count += 1;
    
    dx[i] = domain->xprd*atof(arg[count]);
    count += 1;
  
    dy[i] = domain->yprd*atof(arg[count]);
    count += 1;
  
    dz[i] = domain->zprd*atof(arg[count]);
    count += 1;
    
    if (comm->me == 0)
      cout << "ratio" << "\t" << dx[i] << "\t" << dy[i] << "\t" << dz[i] << endl;
    
  }  
  
  seed = atoi(arg[count]);
  count += 1;
	  
  if (seed<0.0){
    if (comm->me == 0)
      error->warning(FLERR,"Set seed=0");
    seed=0;   
  }

  PENUM = new double [ntypes];

}

/* ---------------------------------------------------------------------- */

DA_type::~DA_type()
{
  delete [] id_pe;
  delete [] PENUM;
  delete [] NUM;
  delete [] dx;
  delete [] dy;
  delete [] dz;
}

/* ---------------------------------------------------------------------- */

int DA_type::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DA_type::init()
{
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA_type does not exist");
  pe = modify->compute[icompute]; 
}

/* ---------------------------------------------------------------------- */

void DA_type::setup(int vflag)
{
  post_force(vflag);
}

void DA_type::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void DA_type::post_force(int vflag)
{
  int i, itype;
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
  int *type = atom->type;
  int ntypes = atom->ntypes;

  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;
  
  int *ntype_local = new int [ntypes];
  double **petype_local = new double* [ntypes];
  for(i=0; i<ntypes; i++){
    ntype_local[i] = 0;
    petype_local[i] = new double [nlocal];
  }
    
  for(i=0; i<nlocal; i++) {
    itype = type[i] - 1;
    petype_local[itype][ntype_local[itype]] = PE[i];
    ntype_local[itype] += 1;
  }

  // gather the tag & property data info of all the atoms on each proc
  
  for(i=0; i<ntypes; i++) {
    int *recvcounts = new int[nprocs];
    int *displs = new int[nprocs];
    
    MPI_Allgather(&ntype_local[i],1,MPI_INT,recvcounts,1,MPI_INT,world);
    displs[0] = 0;
    
    int ntype_global = 0;
    for (int iproc = 1; iproc < nprocs; iproc++) {
      displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
      ntype_global += recvcounts[iproc-1];
    }
    
    ntype_global += recvcounts[nprocs-1]; 
    double *array_buf = new double[ntype_global];
    
    MPI_Allgatherv(petype_local[i],ntype_local[i],MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
    
    sort(array_buf,0,ntype_global-1);

    PENUM[i] = array_buf[ntype_global-NUM[i]-1];
     
    delete [] recvcounts;
    delete [] displs;
    delete [] array_buf;
  }
  
  delete [] ntype_local;
  for(i=0; i<ntypes; i++)
    delete [] petype_local[i];
    
  //DA_type.
  double **x = atom->x;
  int *mask = atom->mask;
	int *mol = atom->molecule;
	int *tag = atom->tag;
  int imol, jmol;

	RanPark *random = new RanPark(lmp,1);

  for (i = 0; i < nlocal; i++) {
		itype = type[i] - 1;
		if ( PE[i]<=PENUM[itype] | !mask[i] | !groupbit )
			continue;

   	random->reset(seed,x[i]);
    double ddx = dx[itype] *(random->uniform()-0.5);
    double ddy = dy[itype] *(random->uniform()-0.5);
    double ddz = dz[itype] *(random->uniform()-0.5);
		imol = mol[i];
	
		if (itype == 0) {
			x[i][0] += ddx;
			x[i][1] += ddy;
			x[i][2] += ddz;
		//	cout << " moved\n";
		}
		
		else {
		for (int j=0; j < nlocal; j++) {
			jmol = mol[j];
			if (imol != jmol | itype == 0)
				continue;
			x[j][0] += ddx;
			x[j][1] += ddy;
			x[j][2] += ddz;
		//		cout << tag[i] << "\t" << tag[j] << "\t" << jmol << "\t" << imol << " moved\n";
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
    sprintf(str,"Lost atoms via DA_type: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }

}

void DA_type::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DA_type::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DA_type::sort(double * a_one, bigint low, bigint high)
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

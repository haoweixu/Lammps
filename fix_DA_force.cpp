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
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_DA_force.h"
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
#include<cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

#define V_MAX 1.0
#define SMALL -1e20
#define LARGE  1e20
#define INVOKED_PERATOM 8
/* ---------------------------------------------------------------------- */

DA_Force::DA_Force(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix DA command. Fix 3 all DA id_pe NumALL xratio seed.");
  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int n = strlen(arg[3]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[3]);
  NUM = ATOBIGINT(arg[4]);
  xratio = atof(arg[5]);
//  yratio = atof(arg[6]);
//  zratio = atof(arg[7]);
//  maxenergy = atof(arg[8]);
//  minenergy = atof(arg[9]);
  seed = atoi(arg[6]);

  nmax = 0;

  if (NUM < 0.0){
	if (comm->me == 0)
           error->warning(FLERR,"Set Num=0");
	NUM=0;
   }
  if (xratio<0.0) {
        if (comm->me == 0)
           error->warning(FLERR,"Set ratio=0.0");
        xratio = 0;
///	yratio = 0;
//	zratio = 0;
  }
  if (seed<0.0){
        if (comm->me == 0)
           error->warning(FLERR,"Set seed=0");
        seed=0;   
  }
  array_buf=NULL;
}

/* ---------------------------------------------------------------------- */

DA_Force::~DA_Force()
{
  delete []id_pe;
  memory->destroy(array_buf);
}

/* ---------------------------------------------------------------------- */

int DA_Force::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DA_Force::init()
{

  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA does not exist");
  pe = modify->compute[icompute]; 
}

/* ---------------------------------------------------------------------- */

void DA_Force::setup(int vflag)
{
  post_force(vflag);
}

void DA_Force::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void DA_Force::post_force(int vflag)
{
  int i;
  // double EngCur = 0;
  // double factor = 1.0;

  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;


  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;

  // create a full array & buf if has not been created

  if (array_buf == NULL) memory->create(array_buf,natoms,"DA_Force:array_buf");

  int *recvcounts,*displs;

  // gather the tag & property data info of all the atoms on each proc

  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
  delete [] recvcounts;
  delete [] displs;

  sort(array_buf,0,natoms-1);
  /*if (comm->me == 0 && screen){
   for(int j=0;j<natoms;j++)
	fprintf(screen,"%f\n", array_buf[j]);
  }*/
  //calculate the avarage energy of atoms in the system
  // for(i = 0;i< natoms; i++)
  // EngCur = EngCur + array_buf[i];
  // EngCur = EngCur/natoms;

  // calculate the corresponding factor that used
  // factor = (EngCur-minenergy)/(maxenergy-minenergy);
  // if (factor>1.0) factor=1.0;
  // if (factor<0.0) factor=0.0;
  // if (comm->me == 0 && screen)
  //       fprintf(screen, "factor is reset to be %f\n", factor);

  //NUM = NUM*factor;
  PENUM = array_buf[natoms-NUM-1];
  //if (comm->me == 0 && screen) fprintf(screen,"PENUM = %f\n", PENUM);

  //DA.
  double **x = atom->x;
	double **f = atom->f;
	double fx, fy, fz, f_total;
  int *mask = atom->mask;
  int *type = atom->type;
	double dl = domain->xprd*xratio;

  RanPark *random = new RanPark(lmp,1);
  // double dx = domain->xprd*xratio;
  // double dy = domain->yprd*yratio;
  // double dz = domain->zprd*zratio;
  //if (comm->me == 0 && screen) fprintf(screen,"%f%f%f\n", dx, dy, dz);
	
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (PE[i]>PENUM){
				fx = f[i][0];
				fy = f[i][1];
				fz = f[i][2];
				f_total = sqrt(fx*fx + fy*fy + fz*fz);

        random->reset(seed,x[i]);
				dl = dl * random->uniform();

        x[i][0] += dl * (fx / f_total);
        x[i][1] += dl * (fy / f_total);
        x[i][2] += dl * (fz / f_total);
      }
    }
  }
  delete random;

  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  tagint *image = atom->image;
  for (i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

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

void DA_Force::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DA_Force::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DA_Force::sort(double * a_one, bigint low, bigint high)
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

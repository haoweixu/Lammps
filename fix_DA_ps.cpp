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
   Contributing author: Yunwei Mao (XJTU), Haowei Xu (MIT)
------------------------------------------------------------------------- */
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_DA_ps.h"
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
/* ---------------------------------------------------------------------- */

DA_ps::DA_ps(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), array_buf(NULL)
{
  if (narg < 11) error->all(FLERR,"Illegal fix DA_ps command. Fix 3 all DA_ps id_pe id_ps NumALL xratio yratio zratio ps_tol seed.");
  
  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);
  
  // default values
	e_tol = 1e-6;
	f_tol = 1e-6;
	max_iter = 1000;
	max_eval = 1000;
  
  int count = 3;
  int n = strlen(arg[count]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[count++]);
  
  n = strlen(arg[count]) + 1;
  id_ps = new char[n];
  strcpy(id_ps,arg[count++]);
  
  NUM = ATOBIGINT(arg[count++]);
  
  xratio = atof(arg[count++]);
  yratio = atof(arg[count++]);
  zratio = atof(arg[count++]);
  
  ps_tol = atof(arg[count++]);
  seed = atoi(arg[count++]);

  atom->add_callback(0);
  
  int nlocal = atom->nlocal;
  int nmax = atom->nmax;
  double **x = atom->x;
	
	memory->create(xold, nmax, 3, "fix_DA_ps_ps: xold");
  memory->create(moved, nmax, "fix_DA_ps_ps: moved");
  for (int i=0; i<nlocal; ++i) {
    xold[i][0] = x[i][0];
    xold[i][1] = x[i][1];
    xold[i][2] = x[i][2];
    moved[i] = 0;
  }
  
	int icompute = modify->find_compute(id_pe);
  if (icompute < 0) 
    error->all(FLERR,"compute pe ID for fix DA_ps does not exist");
  pe = modify->compute[icompute];
  
  icompute = modify->find_compute(id_ps);
  if (icompute < 0) 
    error->all(FLERR,"compute ps ID for fix DA_ps does not exist");
  ps = (ComputePhaseSeperation *)modify->compute[icompute];
}

/* ---------------------------------------------------------------------- */

DA_ps::~DA_ps()
{
  delete [] id_pe;
  delete [] id_ps;
  memory->destroy(array_buf);
  memory->destroy(xold);
  memory->destroy(moved);
  atom->delete_callback(id, 0);
}

/* ---------------------------------------------------------------------- */

int DA_ps::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DA_ps::init()
{
}

/* ---------------------------------------------------------------------- */

void DA_ps::setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void DA_ps::post_force(int vflag)
{
  int i;
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;

  if (!pe->invoked_flag)
    pe->compute_peratom();
  
  if (!ps->invoked_flag)
    ps->compute_peratom();
 
  double ps_old = ps->global_ps_num, ps_new = 0;
  double *PE = pe->vector_atom;

  // create a full array & buf if has not been created
  if (array_buf == NULL) memory->create(array_buf,natoms,"DA_ps:array_buf");

  int *recvcounts,*displs;

  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) 
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  
  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
  delete [] recvcounts;
  delete [] displs;

  sort(array_buf,0,natoms-1);
  PENUM = array_buf[natoms-NUM-1];

  Input::CommandCreatorMap *cmd_map = input->command_map;
  Input::CommandCreator min_creator = (*cmd_map)["minimize"];
  Input::CommandCreator run_creator = (*cmd_map)["run"];
  
  char **min_arg = new char *[4];
  int min_narg = 4;
  min_arg[0] = new char[32]; sprintf(min_arg[0], "%f", e_tol);
	min_arg[1] = new char[32]; sprintf(min_arg[1], "%f", f_tol);
	min_arg[2] = new char[32]; sprintf(min_arg[2], "%d", max_iter);
	min_arg[3] = new char[32]; sprintf(min_arg[3], "%d", max_eval);
 
  char **run_arg = new char *[1];
  int run_narg = 1;
  run_arg[0] = new char[32]; sprintf(run_arg[0], "%d", 0);
  
  //DA_ps.
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
	imageint *image = atom->image;

  RanPark *random = new RanPark(lmp,1);
  double dx = domain->xprd*xratio;
  double dy = domain->yprd*yratio;
  double dz = domain->zprd*zratio;
	double ddx, ddy, ddz;
	
	int mv_count_lcl = 0, mv_count_glb = 0;
  int accept = 0;
  int trial = 1;
 
  while (!accept) {
    // first DA_ps trial
		if (trial == 1) {
			for (i = 0; i < nlocal; i++) {
				if (mask[i] & PE[i]>PENUM){
					random->reset(trial*seed, x[i]);
					ddx = dx * (random->uniform()-0.5);
					ddy = dy * (random->uniform()-0.5);
					ddz = dz * (random->uniform()-0.5);
					x[i][0] += ddx;
					x[i][1] += ddy;
					x[i][2] += ddz;

					moved[i] = 1;
				}
			}
		}
		// non-initial DA_ps trials
		else {
			nlocal = atom->nlocal;
			x = atom->x;
			mv_count_lcl = 0;
			mv_count_glb = 0;
      if (comm->me == 0)
        cout << "ENTER " << trial << " TRIAL" << endl;
			for (i = 0; i < nlocal; i++) {
				// go back to original position
				x[i][0] = xold[i][0];
				x[i][1] = xold[i][1];
				x[i][2] = xold[i][2];
				
        if (moved[i] != 1)
          continue;

    		mv_count_lcl ++;

				random->reset(trial*seed, x[i]);
				ddx = 2 * dx * (random->uniform()-0.5) * random->uniform();
				ddy = 2 * dy * (random->uniform()-0.5) * random->uniform();
				ddz = 2 * dz * (random->uniform()-0.5) * random->uniform();
				x[i][0] += ddx;
				x[i][1] += ddy;
				x[i][2] += ddz;
			}
  		MPI_Allreduce(&mv_count_lcl,&mv_count_glb,1,MPI_INT,MPI_SUM,world);
			if (comm->me == 0)
				cout << "MOVED PARTICLES " << mv_count_glb << endl; 
		}
   
//		cout << "reach line 259 from proc " << comm->me << endl;
//    restore_pbc();
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
    	sprintf(str,"Lost atoms via DA_ps: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    	error->warning(FLERR,str);
 	 	}
//		cout << "reach line 261 from proc " << comm->me << endl;
    min_creator(lmp, min_narg, min_arg); 

		// Metropolis, decide whether to accept this DA_ps step or not.
  	if (!ps->invoked_flag)
    	ps->compute_peratom();
		ps_new = ps->global_ps_num; 
    if (comm->me == 0) {
			cout << "ps_new: " << ps_new << " ps_old: " << ps_old << endl;
      if (ps_new < ps_old) {
        cout << "Case A: DA_ps movement accepted after " << trial << " trials" << endl;
        accept = 1;
      }
      else {
        random->reset(trial*seed, (double *)&ps_new);
        double Ranf = random->uniform();
        cout << "Ranf\t" << Ranf << endl;
        if ( Ranf < exp(-(ps_new - ps_old)/ps_tol) ) {
          cout << "Case B: DA_ps movement accepted after " << trial << " trials" << endl;
          accept = 1;
        }
      } 
    }
    
    trial ++; 
		if (trial>10 && accept==0) {
			nlocal = atom->nlocal;
			x = atom->x;
			for (i = 0; i < nlocal; i++) {
				// go back to original position
				x[i][0] = xold[i][0];
				x[i][1] = xold[i][1];
				x[i][2] = xold[i][2];
			}
			accept = 1;
			if (comm->me == 0)
				cout << "Leaving DA after 10 unsuccessful trials" << endl;
		}
    MPI_Bcast(&accept, 1, MPI_INT, 0, world);
  }
  
  delete random;
  
//	cout << "reach line 304 from proc " << comm->me << endl;
// restore_pbc();
	nlocal = atom->nlocal;
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
    sprintf(str,"Lost atoms via DA_ps: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }
  min_creator(lmp, min_narg, min_arg); 
//	cout << "reach line 306 from proc " << comm->me << endl;
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DA_ps::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DA_ps::grow_arrays(int nmax)
{
  memory->grow(xold, nmax, 3, "fix_DA_ps_ps: xold");
  memory->grow(moved, nmax, "fix_DA_ps_ps: moved");
}

void DA_ps::copy_arrays(int i, int j, int delflag) {
  memcpy(this->xold[j], this->xold[i], 3*sizeof(double));
	moved[j] = moved[i];
}

int DA_ps::pack_exchange(int i, double *buf)
{
  buf[0] = xold[i][0];
  buf[1] = xold[i][1];
  buf[2] = xold[i][2];
  buf[3] = moved[i];
  return 4;
}

int DA_ps::unpack_exchange(int nlocal, double *buf)
{
  xold[nlocal][0] = buf[0];
  xold[nlocal][1] = buf[1];
  xold[nlocal][2] = buf[2];
  moved[nlocal] = buf[3];
  return 4;
}


void DA_ps::set_arrays(int i) {
	memset(xold[i], 0, sizeof(double)*3);
	moved[i] = 0;
}


void DA_ps::sort(double * a_one, bigint low, bigint high)
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

// move atoms back inside simulation box and to new processors
// use remap() instead of pbc() in case atoms moved a long distance
// use irregular() in case atoms moved a long distance
void DA_ps::restore_pbc() {
	int nlocal = atom->nlocal;
	double **x = atom->x;
	int natoms = atom->natoms;
  imageint *image = atom->image;
  for (int i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);
	
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
    sprintf(str,"Lost atoms via DA_ps: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }
}


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
   Contributing author: Haowei Xu (MIT)
   
   This pair_file calculates the Thomas-Fermi kinetic energy for a non-interacting fermi system, and the pseudo-force from this kinetic energy term.
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_tf_kinetic.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;
/* ---------------------------------------------------------------------- */

PairTFKinetic::PairTFKinetic(LAMMPS *lmp) : Pair(lmp)
{
  alpha = 0.0122829 * 1e4; // MeV * fm^2
	settings_flag = 0;
//	cout << alpha << "\n";
}

/* ---------------------------------------------------------------------- */

PairTFKinetic::~PairTFKinetic()
{
  if (allocated) {
    memory->destroy(rho_local);
    memory->destroy(rho_global);
    memory->destroy(exp_r_R);
    delete grid_x;
    delete grid_y;
    delete grid_z;
  }
}

/* ---------------------------------------------------------------------- */

void PairTFKinetic::compute(int eflag, int vflag)
{
//	cout << "enter compute";
  double xmax=0, xmin=0, ymax=0, ymin=0, zmax=0, zmin=0;
  double xmax_global, xmin_global, ymax_global, ymin_global, zmax_global, zmin_global;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int ntype = atom->ntypes;
  int nlocal = atom->nlocal;
  
  for (int i=0; i<nlocal; ++i) {
    xmax = x[i][0]>xmax ? x[i][0] : xmax;
    xmin = x[i][0]<xmin ? x[i][0] : xmin;
    ymax = x[i][1]>ymax ? x[i][1] : ymax;
    ymin = x[i][1]<ymin ? x[i][1] : ymin;
    zmax = x[i][2]>zmax ? x[i][2] : zmax;
    zmin = x[i][2]<zmin ? x[i][2] : zmin;
  }
  MPI_Allreduce(&xmax, &xmax_global, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&xmin, &xmin_global, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&ymax, &ymax_global, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&ymin, &ymin_global, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&zmax, &zmax_global, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&zmin, &zmin_global, 1, MPI_DOUBLE, MPI_MIN, world);
  
  xmax_global += 3*sigma;
  xmin_global -= 3*sigma;
  ymax_global += 3*sigma;
  ymin_global -= 3*sigma;
  zmax_global += 3*sigma;
  zmin_global -= 3*sigma;
  
  double x_itv, y_itv, z_itv;
  x_itv = (xmax_global-xmin_global)/(nx-1);
  y_itv = (ymax_global-ymin_global)/(ny-1);
  z_itv = (zmax_global-zmin_global)/(nz-1);
  
  int nyz = ny*nz;
  for (int i=0; i<ngrids; ++i) {
    grid_x[i] = xmin_global + floor(i/nyz) * x_itv;
    grid_y[i] = ymin_global + floor( (i%nyz)/nz ) * y_itv;
    grid_z[i] = zmin_global + ((i%nyz) % nz) * z_itv;
	
//		if (comm->me==0)
//			cout << grid_x[i]<<"\t"<<grid_y[i]<<"\t"<<grid_z[i]<<"\n";
  }

  
  for (int i=0; i<ntype; ++i)
    for (int j=0; j<nlocal; ++j)
      rho_local[i][j] = 0;
  
  memory->create(exp_r_R, ngrids, nlocal, "pair:exp_r_R");
  
  int itype;
  double delx, dely, delz;
  double igridx, igridy, igridz;
  for (int ir=0; ir<ngrids; ++ir) {
    igridx = grid_x[ir];
    igridy = grid_y[ir];
    igridz = grid_z[ir];
    for (int iR=0; iR<nlocal; ++iR) {
      itype = type[iR]-1;
      delx = x[iR][0] - igridx;
      dely = x[iR][1] - igridy;
      delz = x[iR][2] - igridz;
      exp_r_R[ir][iR] = exp(- (delx*delx+dely*dely+delz*delz)/sigma22 );
      rho_local[itype][ir] += exp_r_R[ir][iR];
    }
  }
  
  for (int i=0; i<ntype; ++i)
    for(int j=0; j<ngrids; ++j) 
      rho_local[i][j] *= beta;
  
  for (int i=0; i<ntype; ++i) {
    MPI_Allreduce(rho_local[i], rho_global[i], ngrids, MPI_DOUBLE, MPI_SUM, world);
  }
  
  // by now, rho(r) and exp(-(r-R)^2/2simga^2) are calculated. Next, calculate force 
  // on each atom.
  
  double dV = x_itv * y_itv * z_itv;
  double int3x, int3y, int3z, prefactor;
  double coeff = (5.0*alpha*beta*dV)/(3.0*sigma22);
  for (int iR=0; iR<nlocal; ++iR) {
    int3x=0; int3y=0; int3z=0;
    itype = type[iR]-1;
    for (int ir=0; ir<ngrids; ++ir) {
      delx = x[iR][0] - grid_x[ir];
      dely = x[iR][1] - grid_y[ir];
      delz = x[iR][2] - grid_z[ir];
      
      prefactor = pow(rho_global[itype][ir], 2.0/3.0) * exp_r_R[ir][iR] ;
      int3x += prefactor * delx;
      int3y += prefactor * dely;
      int3z += prefactor * delz;
    }
    
    int3x *= coeff;
    int3y *= coeff;
    int3z *= coeff;
   
		if (comm->me==0 && iR==1)
			cout << "TF force" << int3x << "\t" << int3y << "\t" << int3z << "\n";
 
    f[iR][0] += int3x;
    f[iR][1] += int3y;
    f[iR][2] += int3z;
  }
  
  double ke2pe;
	for (int itype=0; itype<ntype; ++itype) {
		ke2pe = 0.0;
  	for (int ir=0; ir<ngrids; ++ir)
    	ke2pe += pow(rho_global[itype][ir], 5.0/3.0); 
  	ke2pe *= (alpha*dV);
  	if (eflag && eflag_global) eng_vdwl += ke2pe;
	}	
 	if (comm->me==0) {
		cout << "TF kinetic energy " << ke2pe << " \n"; 
		cout << "eng_vdwl " << eng_vdwl << " \n"; 
	}
  
//  for (int ir=0; ir<ngrids; ++ir)
//    delete exp_r_R[ir];
  memory->destroy(exp_r_R);
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTFKinetic::allocate()
{
  if (settings_flag != 1)
    error->all(FLERR,"allocate called before settings");
  
  allocated = 1;
  int ntypes = atom->ntypes;
	int n = atom->ntypes;
  int natoms = atom->natoms;
  
  ngrids = nx*ny*nz;
 	memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag");
	for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

	memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(rho_local, ntypes, ngrids, "pair:rho_local");
  memory->create(rho_global, ntypes, ngrids, "pair:rho_global");
  memory->create(exp_r_R, ngrids, natoms, "pair:exp_r_R");
  
  grid_x = new double [ngrids];
  grid_y = new double [ngrids];
  grid_z = new double [ngrids];

//	cout << ngrids << "\n";
	
//	cout << "allocate called\n";
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTFKinetic::settings(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Illegal pair_style command");

  double pi = 3.1415926535897;

  sigma = force->numeric(FLERR,arg[0]);
  sigma22 = 2 * sigma * sigma; // 2 * sigma^2
  beta = pow(sigma22*pi, -1.50);
  
  nx = force->numeric(FLERR,arg[1]);
  ny = force->numeric(FLERR,arg[2]);
  nz = force->numeric(FLERR,arg[3]);
//	nx = ny = nz = 10; 
 
  settings_flag = 1;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTFKinetic::coeff(int narg, char **arg)
{
  if (narg != 2) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

	for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }
	
//	cout << "coeff called \n";
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTFKinetic::init_style()
{
	neighbor->request(this,instance_me);	
//	cout << "init_style called\n";	
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTFKinetic::init_one(int i, int j)
{
//	cout << "init_one called\n";
	return 1.0e-5;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTFKinetic::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  fwrite(&sigma, sizeof(double), 1, fp);
  fwrite(&nx, sizeof(int), 1, fp);
  fwrite(&ny, sizeof(int), 1, fp);
  fwrite(&nz, sizeof(int), 1, fp);
  
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTFKinetic::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int me = comm->me;
  if (me == 0) {
    fread(&sigma, sizeof(double), 1, fp);
    fread(&nx, sizeof(int), 1, fp);
    fread(&ny, sizeof(int), 1, fp);
    fread(&nz, sizeof(int), 1, fp);    
  }
  MPI_Bcast(&sigma,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&nx,1,MPI_INT,0,world);
  MPI_Bcast(&ny,1,MPI_INT,0,world);
  MPI_Bcast(&nz,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTFKinetic::write_restart_settings(FILE *fp)
{
  fwrite(&beta,sizeof(double),1,fp);
  fwrite(&sigma22,sizeof(double),1,fp);
  fwrite(&ngrids,sizeof(int),1,fp);

}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTFKinetic::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&beta,sizeof(double),1,fp);
    fread(&sigma22,sizeof(double),1,fp);
    fread(&ngrids,sizeof(int),1,fp);

  }
  MPI_Bcast(&beta,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&sigma22,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&ngrids,1,MPI_INT,0,world);

}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairTFKinetic::write_data(FILE *fp)
{
  fprintf(fp,"%g %d %d %d\n", sigma, nx, ny, nz);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairTFKinetic::write_data_all(FILE *fp)
{
}

/* ---------------------------------------------------------------------- */

double PairTFKinetic::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
	return 0;
}

/* ---------------------------------------------------------------------- */

void *PairTFKinetic::extract(const char *str, int &dim)
{
}

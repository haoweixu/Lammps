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
   Contributing authors: Haowei Xu (MIT)
------------------------------------------------------------------------- */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_TBSMA.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

PairTBSMA::PairTBSMA(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  
  rho = NULL;
  F = NULL;
  A = NULL;
  p = NULL;
  q = NULL;
  xi = NULL;
  lambda = NULL;
  r0 = NULL;
  P = NULL;
  Q = NULL;
  lx2 = NULL;

	nmax = 0;

  comm_forward = 1;
  comm_reverse = 1;
}

PairTBSMA::~PairTBSMA()
{
  if (copymode) return;

  memory->destroy(rho);
  memory->destroy(F);


  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(A);
    memory->destroy(p);
    memory->destroy(q);
    memory->destroy(xi);
    memory->destroy(lambda);
    memory->destroy(r0);
    memory->destroy(P);
    memory->destroy(Q);
    memory->destroy(lx2);
  }
}

void PairTBSMA::compute(int eflag, int vflag)
{
  int i,j,ii,jj,m,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  int *ilist,*jlist,*numneigh,**firstneigh;
	double r,rsq,exp_p,exp_q;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(F);
    nmax = atom->nmax;
    memory->create(rho,nmax,"pair:rho");
    memory->create(F,nmax,"pair:F");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zero out density

  if (newton_pair) {
    for (i = 0; i < nall; i++) rho[i] = 0.0;
  } else for (i = 0; i < nlocal; i++) rho[i] = 0.0;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        exp_q = exp( -2 * q[itype][jtype] * (r/r0[itype][jtype] - 1) );
        rho[i] += lx2[itype][jtype] * exp_q;
        
        if (newton_pair || j < nlocal) {
          rho[j] += lx2[itype][jtype] * exp_q;
        }
      }
    }
  } 

  // communicate and sum densities

  if (newton_pair) comm->reverse_comm_pair(this);


  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    F[i] = sqrt(rho[i]);
    if (eflag) {
      if (eflag_global) eng_vdwl -= F[i];
      if (eflag_atom) eatom[i] -= F[i];
    }
  }

  // communicate derivative of embedding function
  comm->forward_comm_pair(this);

  // compute forces on each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
           
      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        // pair potential
        exp_p = exp( -p[itype][jtype] * ( r/r0[itype][jtype] - 1) );
        evdwl = A[itype][jtype] * exp_p; 
        fpair = P[itype][jtype] * exp_p / r;
        
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
        
        // eam potential, note that the force on atom i comes from three parts:
        // 1) the pair potential; 
        // 2) derivative of eam potential of atom i
        // 3) derivative of eam potential of atom j;

        exp_q = exp( -2 * q[itype][jtype] * (r/r0[itype][jtype] - 1) );
        
        fpair = - 0.5 * (1/F[i] + 1/F[j]) * Q[itype][jtype] * exp_q / r;
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
        
        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTBSMA::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut,n+1,n+1,"pair:cut");
  
  memory->create(A,n+1,n+1,"pair:A");
  memory->create(p,n+1,n+1,"pair:p");
  memory->create(q,n+1,n+1,"pair:q");
  memory->create(xi,n+1,n+1,"pair:xi");
  memory->create(lambda,n+1,n+1,"pair:lambda");
  memory->create(r0,n+1,n+1,"pair:r0");
  memory->create(P,n+1,n+1,"pair:P");
  memory->create(Q,n+1,n+1,"pair:Q");
  memory->create(lx2,n+1,n+1,"pair:lx2");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTBSMA::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  // cut_global, lambda_aa, lambda_ab
  
  cut_global = force->numeric(FLERR,arg[0]);
  lambda_aa = force->numeric(FLERR,arg[1]);
  lambda_ab = force->numeric(FLERR,arg[2]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTBSMA::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  // pair_coeff * * A p q xi r0 [cut]
  
  double A_ = force->numeric(FLERR,arg[2]);
  double p_ = force->numeric(FLERR,arg[3]);
  double q_ = force->numeric(FLERR,arg[4]);
  double xi_ = force->numeric(FLERR,arg[5]);
  double r0_ = force->numeric(FLERR,arg[6]);

  double cut_one = cut_global;
  if (narg == 8) cut_one = force->numeric(FLERR,arg[7]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      A[i][j] = A_;
      p[i][j] = p_;
      q[i][j] = q_;
      xi[i][j] = xi_;
      r0[i][j] = r0_;
      
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTBSMA::init_style()
{
  neighbor->request(this,instance_me);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTBSMA::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    A[i][j] = A[j][i] = mix_energy(A[i][i],A[j][j],r0[i][i],r0[j][j]);
    p[i][j] = p[j][i] = mix_distance(p[i][i],p[j][j]);
    q[i][j] = q[j][i] = mix_distance(q[i][i],q[j][j]);
    r0[i][j] = r0[j][i] = mix_distance(r0[i][i],r0[j][j]);
    xi[i][j] = xi[j][i] = mix_energy(xi[i][i],xi[j][j],r0[i][i],r0[j][j]);
    cut[i][j] = cut[j][i] = mix_distance(cut[i][i],cut[j][j]);
  }
  
  if (i==j) 
    lambda[i][j] = lambda_aa;
  else
    lambda[i][j] = lambda[j][i] = lambda_ab;
  
  P[i][j] = P[j][i] = A[i][j] * p[i][j] / r0[i][j];
  double lx = lambda[i][j] * xi[i][j];
  lx2[i][j] = lx2[j][i] = lx*lx;
  Q[i][j] = Q[j][i] = 2.0 * lx2[i][j] * q[i][j] / r0[i][j];
  
  return cut[i][j];
}


/* ---------------------------------------------------------------------- */

int PairTBSMA::pack_forward_comm(int n, int *list, double *buf,
                               int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = F[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairTBSMA::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) F[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

int PairTBSMA::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = rho[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairTBSMA::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

double PairTBSMA::memory_usage()
{
  double bytes = 0;
  return bytes;
}

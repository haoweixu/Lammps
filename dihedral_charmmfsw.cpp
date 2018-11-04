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
   Contributing author: Paul Crozier (SNL)
     The force-shifted sections were provided by Robert Meissner 
     and Lucio Colombi Ciacchi of Bremen University, Bremen, Germany,
     with additional assistance from Robert A. Latour, Clemson University 
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "dihedral_charmmfsw.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "pair.h"
#include "update.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define TOLERANCE 0.05

/* ---------------------------------------------------------------------- */

DihedralCharmmfsw::DihedralCharmmfsw(LAMMPS *lmp) : Dihedral(lmp)
{
  weightflag = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

DihedralCharmmfsw::~DihedralCharmmfsw()
{
  if (allocated && !copymode) {
    memory->destroy(setflag);
    memory->destroy(k);
    memory->destroy(multiplicity);
    memory->destroy(shift);
    memory->destroy(cos_shift);
    memory->destroy(sin_shift);
    memory->destroy(weight);
  }
}

/* ---------------------------------------------------------------------- */

void DihedralCharmmfsw::compute(int eflag, int vflag)
{
  int i1,i2,i3,i4,i,m,n,type;
  double vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z,vb2xm,vb2ym,vb2zm;
  double edihedral,f1[3],f2[3],f3[3],f4[3];
  double ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg,rginv,ra2inv,rb2inv,rabinv;
  double df,df1,ddf1,fg,hg,fga,hgb,gaa,gbb;
  double dtfx,dtfy,dtfz,dtgx,dtgy,dtgz,dthx,dthy,dthz;
  double c,s,p,sx2,sy2,sz2;
  int itype,jtype;
  double delx,dely,delz,rsq,r2inv,r6inv,r;
  double forcecoul,forcelj,fpair,ecoul,evdwl;

  edihedral = evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  // insure pair->ev_tally() will use 1-4 virial contribution

  if (weightflag && vflag_global == 2)
    force->pair->vflag_either = force->pair->vflag_global = 1;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *atomtype = atom->type;
  int **dihedrallist = neighbor->dihedrallist;
  int ndihedrallist = neighbor->ndihedrallist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;
  double qqrd2e = force->qqrd2e;

  for (n = 0; n < ndihedrallist; n++) {
    i1 = dihedrallist[n][0];
    i2 = dihedrallist[n][1];
    i3 = dihedrallist[n][2];
    i4 = dihedrallist[n][3];
    type = dihedrallist[n][4];

    // 1st bond

    vb1x = x[i1][0] - x[i2][0];
    vb1y = x[i1][1] - x[i2][1];
    vb1z = x[i1][2] - x[i2][2];

    // 2nd bond

    vb2x = x[i3][0] - x[i2][0];
    vb2y = x[i3][1] - x[i2][1];
    vb2z = x[i3][2] - x[i2][2];

    vb2xm = -vb2x;
    vb2ym = -vb2y;
    vb2zm = -vb2z;

    // 3rd bond

    vb3x = x[i4][0] - x[i3][0];
    vb3y = x[i4][1] - x[i3][1];
    vb3z = x[i4][2] - x[i3][2];

    ax = vb1y*vb2zm - vb1z*vb2ym;
    ay = vb1z*vb2xm - vb1x*vb2zm;
    az = vb1x*vb2ym - vb1y*vb2xm;
    bx = vb3y*vb2zm - vb3z*vb2ym;
    by = vb3z*vb2xm - vb3x*vb2zm;
    bz = vb3x*vb2ym - vb3y*vb2xm;

    rasq = ax*ax + ay*ay + az*az;
    rbsq = bx*bx + by*by + bz*bz;
    rgsq = vb2xm*vb2xm + vb2ym*vb2ym + vb2zm*vb2zm;
    rg = sqrt(rgsq);

    rginv = ra2inv = rb2inv = 0.0;
    if (rg > 0) rginv = 1.0/rg;
    if (rasq > 0) ra2inv = 1.0/rasq;
    if (rbsq > 0) rb2inv = 1.0/rbsq;
    rabinv = sqrt(ra2inv*rb2inv);

    c = (ax*bx + ay*by + az*bz)*rabinv;
    s = rg*rabinv*(ax*vb3x + ay*vb3y + az*vb3z);

    // error check

    if (c > 1.0 + TOLERANCE || c < (-1.0 - TOLERANCE)) {
      int me;
      MPI_Comm_rank(world,&me);
      if (screen) {
        char str[128];
        sprintf(str,"Dihedral problem: %d " BIGINT_FORMAT " "
                TAGINT_FORMAT " " TAGINT_FORMAT " "
                TAGINT_FORMAT " " TAGINT_FORMAT,
                me,update->ntimestep,
                atom->tag[i1],atom->tag[i2],atom->tag[i3],atom->tag[i4]);
        error->warning(FLERR,str,0);
        fprintf(screen,"  1st atom: %d %g %g %g\n",
                me,x[i1][0],x[i1][1],x[i1][2]);
        fprintf(screen,"  2nd atom: %d %g %g %g\n",
                me,x[i2][0],x[i2][1],x[i2][2]);
        fprintf(screen,"  3rd atom: %d %g %g %g\n",
                me,x[i3][0],x[i3][1],x[i3][2]);
        fprintf(screen,"  4th atom: %d %g %g %g\n",
                me,x[i4][0],x[i4][1],x[i4][2]);
      }
    }

    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;

    m = multiplicity[type];
    p = 1.0;
    ddf1 = df1 = 0.0;

    for (i = 0; i < m; i++) {
      ddf1 = p*c - df1*s;
      df1 = p*s + df1*c;
      p = ddf1;
    }

    p = p*cos_shift[type] + df1*sin_shift[type];
    df1 = df1*cos_shift[type] - ddf1*sin_shift[type];
    df1 *= -m;
    p += 1.0;

    if (m == 0) {
      p = 1.0 + cos_shift[type];
      df1 = 0.0;
    }

    if (eflag) edihedral = k[type] * p;

    fg = vb1x*vb2xm + vb1y*vb2ym + vb1z*vb2zm;
    hg = vb3x*vb2xm + vb3y*vb2ym + vb3z*vb2zm;
    fga = fg*ra2inv*rginv;
    hgb = hg*rb2inv*rginv;
    gaa = -ra2inv*rg;
    gbb = rb2inv*rg;

    dtfx = gaa*ax;
    dtfy = gaa*ay;
    dtfz = gaa*az;
    dtgx = fga*ax - hgb*bx;
    dtgy = fga*ay - hgb*by;
    dtgz = fga*az - hgb*bz;
    dthx = gbb*bx;
    dthy = gbb*by;
    dthz = gbb*bz;

    df = -k[type] * df1;

    sx2 = df*dtgx;
    sy2 = df*dtgy;
    sz2 = df*dtgz;

    f1[0] = df*dtfx;
    f1[1] = df*dtfy;
    f1[2] = df*dtfz;

    f2[0] = sx2 - f1[0];
    f2[1] = sy2 - f1[1];
    f2[2] = sz2 - f1[2];

    f4[0] = df*dthx;
    f4[1] = df*dthy;
    f4[2] = df*dthz;

    f3[0] = -sx2 - f4[0];
    f3[1] = -sy2 - f4[1];
    f3[2] = -sz2 - f4[2];

    // apply force to each of 4 atoms

    if (newton_bond || i1 < nlocal) {
      f[i1][0] += f1[0];
      f[i1][1] += f1[1];
      f[i1][2] += f1[2];
    }

    if (newton_bond || i2 < nlocal) {
      f[i2][0] += f2[0];
      f[i2][1] += f2[1];
      f[i2][2] += f2[2];
    }

    if (newton_bond || i3 < nlocal) {
      f[i3][0] += f3[0];
      f[i3][1] += f3[1];
      f[i3][2] += f3[2];
    }

    if (newton_bond || i4 < nlocal) {
      f[i4][0] += f4[0];
      f[i4][1] += f4[1];
      f[i4][2] += f4[2];
    }

    if (evflag)
      ev_tally(i1,i2,i3,i4,nlocal,newton_bond,edihedral,f1,f3,f4,
               vb1x,vb1y,vb1z,vb2x,vb2y,vb2z,vb3x,vb3y,vb3z);

    // 1-4 LJ and Coulomb interactions
    // tally energy/virial in pair, using newton_bond as newton flag

    if (weight[type] > 0.0) {
      itype = atomtype[i1];
      jtype = atomtype[i4];

      delx = x[i1][0] - x[i4][0];
      dely = x[i1][1] - x[i4][1];
      delz = x[i1][2] - x[i4][2];
      rsq = delx*delx + dely*dely + delz*delz;
      r2inv = 1.0/rsq;
      r6inv = r2inv*r2inv*r2inv;

      // modifying coul and LJ force and energies to apply 
      //   force_shift and force_switch as in CHARMM pairwise
      // LJ interactions between 1-4 atoms should usually be
      //   for r < cut_inner, so switching not applied

      r = sqrt(rsq);
      if (implicit) forcecoul = qqrd2e * q[i1]*q[i4]*r2inv;
      else if (dihedflag) forcecoul = qqrd2e * q[i1]*q[i4]*sqrt(r2inv);
      else forcecoul = qqrd2e * q[i1]*q[i4]*(sqrt(r2inv) - 
                                             r*cut_coulinv14*cut_coulinv14);
      forcelj = r6inv * (lj14_1[itype][jtype]*r6inv - lj14_2[itype][jtype]);
      fpair = weight[type] * (forcelj+forcecoul)*r2inv;

      if (eflag) {
        if (dihedflag) ecoul = weight[type] * forcecoul;
        else ecoul = weight[type] * qqrd2e * q[i1]*q[i4] *
               (sqrt(r2inv) + r*cut_coulinv14*cut_coulinv14 - 
                2.0*cut_coulinv14);
        evdwl14_12 = r6inv*lj14_3[itype][jtype]*r6inv - 
          lj14_3[itype][jtype]*cut_lj_inner6inv*cut_lj6inv;
        evdwl14_6 = -lj14_4[itype][jtype]*r6inv + 
          lj14_4[itype][jtype]*cut_lj_inner3inv*cut_lj3inv;
        evdwl = evdwl14_12 + evdwl14_6;
        evdwl *= weight[type];
      }

      if (newton_bond || i1 < nlocal) {
        f[i1][0] += delx*fpair;
        f[i1][1] += dely*fpair;
        f[i1][2] += delz*fpair;
      }
      if (newton_bond || i4 < nlocal) {
        f[i4][0] -= delx*fpair;
        f[i4][1] -= dely*fpair;
        f[i4][2] -= delz*fpair;
      }

      if (evflag) force->pair->ev_tally(i1,i4,nlocal,newton_bond,
                                        evdwl,ecoul,fpair,delx,dely,delz);
    }
  }
}

/* ---------------------------------------------------------------------- */

void DihedralCharmmfsw::allocate()
{
  allocated = 1;
  int n = atom->ndihedraltypes;

  memory->create(k,n+1,"dihedral:k");
  memory->create(multiplicity,n+1,"dihedral:k");
  memory->create(shift,n+1,"dihedral:shift");
  memory->create(cos_shift,n+1,"dihedral:cos_shift");
  memory->create(sin_shift,n+1,"dihedral:sin_shift");
  memory->create(weight,n+1,"dihedral:weight");

  memory->create(setflag,n+1,"dihedral:setflag");
  for (int i = 1; i <= n; i++) setflag[i] = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one type
------------------------------------------------------------------------- */

void DihedralCharmmfsw::coeff(int narg, char **arg)
{
  if (narg != 5) error->all(FLERR,"Incorrect args for dihedral coefficients");
  if (!allocated) allocate();

  int ilo,ihi;
  force->bounds(FLERR,arg[0],atom->ndihedraltypes,ilo,ihi);

  // require integer values of shift for backwards compatibility
  // arbitrary phase angle shift could be allowed, but would break
  //   backwards compatibility and is probably not needed

  double k_one = force->numeric(FLERR,arg[1]);
  int multiplicity_one = force->inumeric(FLERR,arg[2]);
  int shift_one = force->inumeric(FLERR,arg[3]);
  double weight_one = force->numeric(FLERR,arg[4]);

  if (multiplicity_one < 0)
    error->all(FLERR,"Incorrect multiplicity arg for dihedral coefficients");
  if (weight_one < 0.0 || weight_one > 1.0)
    error->all(FLERR,"Incorrect weight arg for dihedral coefficients");
  if (weight_one > 0.0) weightflag=1;

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    k[i] = k_one;
    shift[i] = shift_one;
    cos_shift[i] = cos(MY_PI*shift_one/180.0);
    sin_shift[i] = sin(MY_PI*shift_one/180.0);
    multiplicity[i] = multiplicity_one;
    weight[i] = weight_one;
    setflag[i] = 1;
    count++;
  }

  if (count == 0) error->all(FLERR,"Incorrect args for dihedral coefficients");
}

/* ----------------------------------------------------------------------
   error check and initialize all values needed for force computation
------------------------------------------------------------------------- */

void DihedralCharmmfsw::init_style()
{
  if (strstr(update->integrate_style,"respa")) {
    Respa *r = (Respa *) update->integrate;
    if (r->level_pair >= 0 && (r->level_pair != r->level_dihedral))
      error->all(FLERR,"Dihedral style charmmfsw must be set to same"
        " r-RESPA level as 'pair'");
    if (r->level_outer >= 0 && (r->level_outer != r->level_dihedral))
      error->all(FLERR,"Dihedral style charmmfsw must be set to same"
        " r-RESPA level as 'outer'");
  }

  // insure use of CHARMM pair_style if any weight factors are non-zero
  // set local ptrs to LJ 14 arrays setup by Pair
  // also verify that the correct 1-4 scaling is set

  if (weightflag) {

    if ((force->special_lj[3] != 0.0) || (force->special_coul[3] != 0.0))
      error->all(FLERR,"Must use 'special_bonds charmm' with"
                 " dihedral style charmm for use with CHARMM pair styles");

    int itmp;
    if (force->pair == NULL)
      error->all(FLERR,"Dihedral charmmfsw is incompatible with Pair style");
    lj14_1 = (double **) force->pair->extract("lj14_1",itmp);
    lj14_2 = (double **) force->pair->extract("lj14_2",itmp);
    lj14_3 = (double **) force->pair->extract("lj14_3",itmp);
    lj14_4 = (double **) force->pair->extract("lj14_4",itmp);
    int *ptr = (int *) force->pair->extract("implicit",itmp);
    if (!lj14_1 || !lj14_2 || !lj14_3 || !lj14_4 || !ptr)
      error->all(FLERR,"Dihedral charmmfsw is incompatible with Pair style");
    implicit = *ptr;
  }

  // constants for applying force switch (LJ) and force_shift (coul)
  // to 1/4 dihedral atoms to match CHARMM pairwise interactions

  int itmp;
  int *p_dihedflag = (int *) force->pair->extract("dihedflag",itmp);
  double *p_cutljinner = (double *) force->pair->extract("cut_lj_inner",itmp);
  double *p_cutlj = (double *) force->pair->extract("cut_lj",itmp);
  double *p_cutcoul = (double *) force->pair->extract("cut_coul",itmp);
  
  if (p_cutcoul == NULL || p_cutljinner == NULL || 
      p_cutlj == NULL || p_dihedflag == NULL)
    error->all(FLERR,"Dihedral charmmfsw is incompatible with Pair style");
  
  dihedflag = *p_dihedflag;
  cut_coul14 = *p_cutcoul;
  cut_lj_inner14 = *p_cutljinner;
  cut_lj14 = *p_cutlj;

  cut_coulinv14 = 1/cut_coul14;
  cut_lj_inner3inv = (1/cut_lj_inner14) * (1/cut_lj_inner14) * 
    (1/cut_lj_inner14);
  cut_lj_inner6inv = cut_lj_inner3inv * cut_lj_inner3inv;
  cut_lj3inv = (1/cut_lj14) * (1/cut_lj14) * (1/cut_lj14);
  cut_lj6inv = cut_lj3inv * cut_lj3inv;
}

/* ----------------------------------------------------------------------
   proc 0 writes out coeffs to restart file
------------------------------------------------------------------------- */

void DihedralCharmmfsw::write_restart(FILE *fp)
{
  fwrite(&k[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&multiplicity[1],sizeof(int),atom->ndihedraltypes,fp);
  fwrite(&shift[1],sizeof(int),atom->ndihedraltypes,fp);
  fwrite(&weight[1],sizeof(double),atom->ndihedraltypes,fp);
  fwrite(&weightflag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads coeffs from restart file, bcasts them
------------------------------------------------------------------------- */

void DihedralCharmmfsw::read_restart(FILE *fp)
{
  allocate();

  if (comm->me == 0) {
    fread(&k[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&multiplicity[1],sizeof(int),atom->ndihedraltypes,fp);
    fread(&shift[1],sizeof(int),atom->ndihedraltypes,fp);
    fread(&weight[1],sizeof(double),atom->ndihedraltypes,fp);
    fread(&weightflag,sizeof(int),1,fp);
  }
  MPI_Bcast(&k[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&multiplicity[1],atom->ndihedraltypes,MPI_INT,0,world);
  MPI_Bcast(&shift[1],atom->ndihedraltypes,MPI_INT,0,world);
  MPI_Bcast(&weight[1],atom->ndihedraltypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&weightflag,1,MPI_INT,0,world);

  for (int i = 1; i <= atom->ndihedraltypes; i++) {
    setflag[i] = 1;
    cos_shift[i] = cos(MY_PI*shift[i]/180.0);
    sin_shift[i] = sin(MY_PI*shift[i]/180.0);
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void DihedralCharmmfsw::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ndihedraltypes; i++)
    fprintf(fp,"%d %g %d %d %g\n",i,k[i],multiplicity[i],shift[i],weight[i]);
}


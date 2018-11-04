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

#include <string.h>
#include <stdlib.h>
#include "fix_elec_stopping.h"
#include "atom.h"
#include "atom_masks.h"
#include "accelerator_kokkos.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

enum{NONE,CONSTANT,EQUAL,ATOM};

/*-------------------------------------------------------------
Add force on the projectile atoms from electron stopping.
At high energy:     Bethe-Bloch
At low energy:      Lindhard-Scharff 
-------------------------------------------------------------*/

/* ---------------------------------------------------------------------- */

FixElecStopping::FixElecStopping(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  xstr(NULL), ystr(NULL), zstr(NULL), estr(NULL), idregion(NULL), sforce(NULL)

{
  if (narg < 6) error->all(FLERR,"Illegal fix addforce command, 
  syntax: fix ID group-ID elec/stopping type Z1(incident particle) Z2 number densiy hl_threshold");
  count = 3;
  inc_type = atoi(arg[count]);
  count += 1;

  Z1 = atof(arg[count]);
  count += 1;

  Z2 = atof(arg[count]);
  count += 1;

  num_density = atof(arg[count]); // atom/m3;
  count += 1;

  hl_thres = atof(arg[count]);
  count += 1;

  mass_e = 5.4461e-4; // mass of electron in metal unit;
  e_Gaussian = 1.5189e-14;
  e_SI = 1.602e-19;
  k = 11.5; // eV
  Ibar = k*Z2; // in eV
}

/* ---------------------------------------------------------------------- */

FixElecStopping::~FixElecStopping()
{
  
}

/* ---------------------------------------------------------------------- */

int FixElecStopping::setmask()
{
  datamask_read = datamask_modify = 0;

  int mask = 0;
  mask |= POST_FORCE;
//  mask |= THERMO_ENERGY;
//  mask |= POST_FORCE_RESPA;
//  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixElecStopping::init()
{
  // check variables

}

/* ---------------------------------------------------------------------- */

void FixElecStopping::post_force(int vflag)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  int *type = atom->type;
  int *mass = atom->mass;
  int nlocal = atom->nlocal;

  double mass2eV;
  double Joule2eV = 1.0/(1.602e-19);
  double v2SI = 100.0; // A/ps to m/s;
  double Joulem2eVA = Joule2eV * 1e-10; // force, Joule/m to eV/A;
  double pi = 3.14159265359;
  double c0 = 299792458 * 1e-2; // speed of light in A/ps;


  double gamma_e;
  double k_LS; //  Lindhard-Scharff k-prime
  double Ei, Ei_SI;
  double dEdx;
  double beta;

  int itype;
  for (int i=1; i<nlocal; ++i) {
    itype = type[i]
    if (itype != inc_type)
      continue;
    imass = mass[itype];
    vsq = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
    vnorm = sqrt(vsq);

    Ei = 0.5 * imass * vsq * v2SI * v2SI * Joule2eV; // in eV
    Ei_SI = Ei / Joule2eV;
    
    if (Ei > hl_thres) { // Bethe-Bloch 
      gamma_e = 4*imass*mass_e / pow(imass+mass_e, 2.0)
      beta = vnorm / c0;

      B = Z2 * log(gamma_e * Ei / Ibar);
      dEdx = -2 * pi * num_density * Z1*Z1 * imass * pow(e_Gaussian, 4.0) * B / (mass_e * Ei_SI); // in SI, Joule/m;
      dEdx = dEdx * Joulem2eVA; // in eV/A;
      
      f[i][0] += dEdx * v[i][0] / vnorm;
      f[i][1] += dEdx * v[i][1] / vnorm;
      f[i][2] += dEdx * v[i][2] / vnorm;
    }

    else if (Ei < hl_thres) { // Lindhard-Scharff
      k_LS = 3.83*pow(Z1,7.0/6.0)*Z2 / 
      (sqrt(imass) * pow( pow(Z1,2.0/3.0)+pow(Z2,2.0/3.0), 3.0/2.0 ) );
      dEdx = -num_density * k_LS * sqrt(Ei/1000); // in 1e-19 eV/m;
      dEdx = dEdx * 1e-29; // in eV/A;

      f[i][0] += dEdx * v[i][0] / vnorm;
      f[i][1] += dEdx * v[i][1] / vnorm;
      f[i][2] += dEdx * v[i][2] / vnorm;
    }
  }
  
}

/* ---------------------------------------------------------------------- */

void FixElecStopping::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixElecStopping::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixElecStopping::compute_scalar()
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[0];
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixElecStopping::compute_vector(int n)
{
  // only sum across procs one time

  if (force_flag == 0) {
    MPI_Allreduce(foriginal,foriginal_all,4,MPI_DOUBLE,MPI_SUM,world);
    force_flag = 1;
  }
  return foriginal_all[n+1];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixElecStopping::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom*4 * sizeof(double);
  return bytes;
}

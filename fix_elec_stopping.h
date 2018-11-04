/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(elec/stopping, FixElecStopping)

#else // register the fix here

#ifndef LMP_FIX_ELECSTOPPING_H
#define LMP_FIX_ELECSTOPPING_H

#include "fix.h"

namespace LAMMPS_NS {

class FixElecStopping : public Fix {
 public:
  FixElecStopping(class LAMMPS *, int, char **);
  ~FixElecStopping();
  int setmask();
  void init();
  void setup(int);
//  void min_setup(int);
  void post_force(int);
//  void post_force_respa(int, int, int);
//  void min_post_force(int);
//  double compute_scalar();
//  double compute_vector(int);
  double memory_usage();

 private:
  double Z1, Z2, num_density, hl_thres;
  int inc_type;

  double mass_e; // mass of electron;
  double e_Gaussion; // charge of electron, in Gauss unit, which is e^/sqrt(4*pi*epsilon0) is SI unit;
  double e_SI; // charge of electron in Si unit
  double k; // ironization energy constant, \bar{I} = K*Z2;
  double Ibar;

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix addforce does not exist

Self-explanatory.

E: Variable name for fix addforce does not exist

Self-explanatory.

E: Variable for fix addforce is invalid style

Self-explanatory.

E: Cannot use variable energy with constant force in fix addforce

This is because for constant force, LAMMPS can compute the change
in energy directly.

E: Must use variable energy with fix addforce

Must define an energy variable when applying a dynamic
force during minimization.

*/

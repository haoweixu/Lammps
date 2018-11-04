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

FixStyle(ISF,FixISF)

#else // register the fix here

#ifndef LMP_FIX_ISF_H
#define LMP_FIX_ISF_H

#include "fix.h"
#include <vector>
using namespace std;

namespace LAMMPS_NS {

class FixISF : public Fix {
 public:
  FixISF(class LAMMPS *, int, char **);
  ~FixISF();
  int setmask();
  void setup(int);
  void min_setup(int);
  void post_force(int);
  void min_post_force(int);
  double memory_usage();

 private:
  double **xold, **xnew;
  vector <int> timestep;
  vector <double> isf_r;
  vector <double> isf_i;
  int *type_global, TYPE;
  double q;
  
  int me, nprocs;
  
  int check_timestep(int);
  
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

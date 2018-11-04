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

#ifdef COMPUTE_CLASS

ComputeStyle(SD, ComputeSD)

#else // register the fix here

#ifndef LMP_COMPUTE_SD_H
#define LMP_COMPUTE_SD_H

#include "compute.h"
#include <vector>
using namespace std;

namespace LAMMPS_NS {

class ComputeSD : public Compute {
 public:
  ComputeSD(class LAMMPS *, int, char **);
  ~ComputeSD();
  virtual void compute_peratom();
  void set_arrays(int);
 	void init(); 

 private:
  char *id_fix;
  class FixStore *fix;
  int TYPE;
  
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

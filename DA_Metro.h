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

#ifdef COMMAND_CLASS

CommandStyle(DA_Metro,DA_Metro)

#else

#ifndef LMP_DA_METRO_H
#define LMP_DA_METRO_H

#include "pointers.h"

namespace LAMMPS_NS {

class DA_Metro : public Pointers {
 public:
  DA_Metro(class LAMMPS *);
  void command(int, char **);
  void sort(double *, bigint, bigint);

 protected:
  bigint NUM;
  double xratio;
  double yratio;
  double zratio;
  int seed;
  int maxstep;
  double mindis;
 
	double e_tol;
	double f_tol;
	int max_iter;
  int max_eval;

	double sigma;
	
  int me;
  int nprocs;
  int nmax;
  
  char *id_pe;

  class Compute *pe;

  double *array_buf;
  double PENUM;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix langevin period must be > 0.0

The time window for temperature relaxation must be > 0

E: Fix langevin omega requires atom style sphere

Self-explanatory.

E: Fix langevin angmom requires atom style ellipsoid

Self-explanatory.

E: Variable name for fix langevin does not exist

Self-explanatory.

E: Variable for fix langevin is invalid style

It must be an equal-style variable.

E: Fix langevin omega requires extended particles

One of the particles has radius 0.0.

E: Fix langevin angmom requires extended particles

This fix option cannot be used with point paritlces.

E: Fix langevin variable returned negative temperature

Self-explanatory.

E: Cannot zero Langevin force of 0 atoms

The group has zero atoms, so you cannot request its force
be zeroed.

E: Could not find fix_modify temperature ID

The compute ID for computing temperature does not exist.

E: Fix_modify temperature ID does not compute temperature

The compute ID assigned to the fix must compute temperature.

W: Group for fix_modify temp != fix group

The fix_modify command is specifying a temperature computation that
computes a temperature on a different group of atoms than the fix
itself operates on.  This is probably not what you want to do.

*/

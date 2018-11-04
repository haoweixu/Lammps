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

ComputeStyle(phase_sep, ComputePhaseSeperation)

#else

#ifndef LMP_COMPUTE_PHASE_SEPERATION_H
#define LMP_COMPUTE_PHASE_SEPERATION_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputePhaseSeperation : public Compute {
 public:
  ComputePhaseSeperation(class LAMMPS *, int, char **);
  ~ComputePhaseSeperation();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  double memory_usage();
  
  double global_ps;
  int global_ps_num;

 private:
  double rcut;
  double thres;
  class NeighList *list;
  int nevery;
  double *itype_frac;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute orientorder/atom requires a pair style be defined

Self-explanatory.

E: Compute orientorder/atom cutoff is longer than pairwise cutoff

Cannot compute order parameter beyond cutoff.

W: More than one compute orientorder/atom

It is not efficient to use compute orientorder/atom more than once.

*/

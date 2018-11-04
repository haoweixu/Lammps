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

#ifdef PAIR_CLASS

PairStyle(tbsma,PairTBSMA)

#else

#ifndef LMP_PAIR_TBSMA_H
#define LMP_PAIR_TBSMA_H

#include <stdio.h>
#include "pair.h"

namespace LAMMPS_NS {


class PairTBSMA : public Pair {
 public:
 
  PairTBSMA(class LAMMPS *);
  virtual ~PairTBSMA();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
//  void init_list(int, class NeighList *)
//  double single(int, int, int, int, double, double, double, double &);
//  virtual void *extract(const char *, int &);

  virtual int pack_forward_comm(int, int *, double *, int, int *);
  virtual void unpack_forward_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();

 protected:
 
  virtual void allocate();

	int nmax;
  double lambda_aa, lambda_ab;
  double cut_global, **cut;
  double **A, **p, **q, **xi, **lambda, **r0;
  double **P, **Q, **lx2;
  double *rho, *F;
  
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Cannot open EAM potential file %s

The specified EAM potential file cannot be opened.  Check that the
path and name are correct.

*/

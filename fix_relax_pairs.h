/*-------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*---------------------------------------------------------------------------
   Contributing Author: Haowei Xu (MIT)
   
   Some pair potentials, like Buckingham potential, become infinitely attractive
   when two atoms are too close to each other, which leads to the so called
   "fusion" problem.
   
   This Fix class fixes this problem by simply checking the distance of each pair
   before every MD step and separate them if they are closer than a prescribed 
   distance, which is usually the maximum of the potential barrier.
----------------------------------------------------------------------------*/


#ifdef FIX_CLASS

FixStyle(pair/relax, FixPairRelax)

#else

#ifndef LMP_FIX_PAIR_RELAX_H
#define LMP_FIX_PAIR_RELAX_H

#include "fix.h"

namespace LAMMPS_NS {
  
class FixPairRelax : public Fix {
  public:
    FixPairRelax(class LAMMPS *, int, char **);
    ~FixPairRelax();
    int setmask();
    void init();
//    void setup(int);
    void min_setup(int);
    void pre_force(int);
    void min_pre_force(int);
    double memory_usage();
    void init_list(int, class NeighList *);
  
  protected:
    int ntypes;
    double **dist_after_mv, **min_dist_sqr;
    double epsilon;
    class NeighList *list;
    int mv_count;
};
  
}







#endif
#endif

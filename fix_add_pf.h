/*--------------------------------Haowei Xu-------------------------------------
11/30/2017
-------------------------------------------------------------------------------*/

#ifdef FIX_CLASS

FixStyle(add_pf, FixAddPF)

#else

#ifndef LMP_FIX_ADD_PF_H
#define LMP_FIX_ADD_PF_H

#include "fix.h"

namespace LAMMPS_NS {
	
class FixAddPF : public Fix {
	public:
	
	FixAddPF(class LAMMPS *, int, char **);
	~FixAddPF();
	int setmask();
	void init();
	void setup(int);
	void min_setup(int);
	void post_force(int);
//	void post_force_respa(int, int, int);
	void min_post_force(int);
	double compute_scalar();
	void copy_arrays(int, int, int);
	void grow_arrays(int);
	int pack_exchange(int, double *);
	int unpack_exchange(int, double *);
	void set_arrays(int);
//	double compute_vector();
	double memory_usage();
	
	private:
	
	double W;
	double sigma;
	double **x_origin;
	int length_x_origin;
	double dist_all;
//	int dist_all_flag;
	double W_exp;
};	
	
}

#endif
#endif

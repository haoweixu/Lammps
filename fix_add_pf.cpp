/*------------------------------------Haowei Xu------------------------------------------
11/30/2017
thermo_energy should be set to one to enalbe adding potential energy from pf (or generally,
from all fixes that contributes to the totoal pe).

12/05/2017
some notes on MPI:
	1. x_origin should be a local vector, which each proc has its own ones. Of 
	course x_origin can also be a global vector, and be broadcasted to all procs. 
	But this would be a waste of memory.
	
	2. some functions, grow_arrays, pack_exchange, unpack_exchange, etc. should be 
	added to enable exchange x_origin between procs when one atom moves from one 
	proc to another. Add_callback() should be called in the constructor, while 
	delete_callback shall be called in the destructor.
	
	3. about ghost atoms. It seems that these ghost atoms facilitate the information
	exchange between procs. Thus ghost atoms should also have there x_origin (use nmax
	rather than nlocal in some cases). But note that ghost atoms should not enter the 
	penalty function. Also, when adding extra forces, use nlocal is enough. There may 
	be some routines that copies atom class properties like forces.
	
	4. nmax = nlocal + nghost. Id runs from 0 to nmax-1, where 0 to nlocal-1 represent 
	real atoms, while remaining ones are ghost atoms.
	
	5. some functions in memory class, like memory->create, memory->destroy.
-----------------------------------------------------------------------------------------*/

#include <string.h>
#include <stdlib.h>
#include <cmath>
#include "fix_addforce.h"
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
#include "fix_add_pf.h"
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

FixAddPF::FixAddPF(LAMMPS *lmp, int narg, char **arg) : 
	Fix(lmp, narg, arg) 
{
	// fix ID all add_pf W sigma
	if (narg < 5) error->all(FLERR, "Illegal fix add_pf command");
	
	dynamic_group_allow = 1;
	scalar_flag = 1;
//	vector_flag = 1;
	
	extscalar = 1; // take care of this. may affect how the potential is added to total potential;
	extvector = 1;
	
	thermo_energy = 1;
//	nevery = 1;
	
	W = force->numeric(FLERR, arg[3]);
//	std::cout << "W is " << W;
	sigma = force->numeric(FLERR, arg[4]);
	
	dist_all = 0;
	
	int nmax = atom->nmax;
	double **x = atom->x;
	int *mask = atom->mask;
	imageint *image = atom->image;
	double unwrap[3];
	memory->create(x_origin, nmax, 3, "addpf:x_origin");
//	x_origin = new double*[nlocal];
	for (int i=0; i<nmax; ++i) {
//		x_origin[i] = new double[3];
//		memory->create(x_origin[i], 3, "addpf:x_origin");
		domain->unmap(x[i], image[i], unwrap);
		for (int j=0; j<3; ++j)
			x_origin[i][j] = unwrap[j];
	}
	atom->add_callback(0);	
}

FixAddPF::~FixAddPF() {
//	for(int i=0; i<length_x_origin; ++i)
//		memory->destroy(x_origin[i]);
	memory->destroy(x_origin);
	atom->delete_callback(id, 0);
}

void FixAddPF::init() {
	
}

int FixAddPF::setmask() {
	datamask_read = datamask_modify = 0; // take care of this, don't know what this means;
	
	int mask = 0;
	mask |= POST_FORCE;
	mask |= THERMO_ENERGY;
	mask |= MIN_POST_FORCE;
//	mask |= POST_FORCE_RESPA;
	return mask;
}

void FixAddPF::setup(int vflag) {
	if (strstr(update->integrate_style,"verlet"))
		post_force(vflag);
}

void FixAddPF::min_setup(int vflag) {
	post_force(vflag);
}

void FixAddPF::post_force(int vflag) {
	
	double **x = atom->x;
	double **f = atom->f;
	int *mask = atom->mask;
	imageint *image = atom->image;
	int nlocal = atom->nlocal;
	length_x_origin = atom->nlocal;
	double unwrap[nlocal][3];
	double dist_local = 0, dist_1d;
	double two_sigma_sqr = 2*sigma*sigma;
	
	// distance between current configuration and the origin of the pf
	for (int i=0; i<nlocal; ++i) {
		domain->unmap(x[i],image[i],unwrap[i]);
		for (int j=0; j<3; ++j) {
			dist_1d = unwrap[i][j] - x_origin[i][j];
			dist_local += dist_1d * dist_1d;
		}
	}
	
	dist_all = 0;
//	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&dist_local, &dist_all, 1, MPI_DOUBLE, MPI_SUM, world);	
	W_exp = W * exp( -dist_all / (two_sigma_sqr) );
	for (int i=0; i<nlocal; ++i) {
		f[i][0] += (unwrap[i][0] - x_origin[i][0]) * W_exp / two_sigma_sqr;
		f[i][1] += (unwrap[i][1] - x_origin[i][1]) * W_exp / two_sigma_sqr;
		f[i][2] += (unwrap[i][2] - x_origin[i][2]) * W_exp / two_sigma_sqr;
	}
//	std::cout << "add_pf_called\n";		
}

void FixAddPF::min_post_force(int vflag) {
	post_force(vflag);
}

double FixAddPF::compute_scalar() {
	return W_exp;
}

void FixAddPF::grow_arrays(int nmax) {
//	std::cout << "GROW_ARRAYS CALLED";
/*	x_origin = new double *[nmax];
	for (int i=0; i<nmax; ++i) {
		memory->grow(x_origin[i], 3, "addpf:x_origin");
	}
*/
	memory->grow(x_origin, nmax, 3, "addpf:x_origin");
}

void FixAddPF::copy_arrays(int i, int j, int delflag) {
//	std::cout << "COPY_ARRAYS CALLED";
	x_origin[j][0] = x_origin[i][0];
	x_origin[j][1] = x_origin[i][1];
	x_origin[j][2] = x_origin[i][2];
}


int FixAddPF::pack_exchange(int i, double *buf) {
//	std::cout << "PACK_EXCHANGE CALLED";
	buf[0] = x_origin[i][0];
	buf[1] = x_origin[i][1];
	buf[2] = x_origin[i][2];
	return 3;
}

int FixAddPF::unpack_exchange(int nlocal, double *buf) {
//	std::cout << "UNPACK_EXCHANGE CALLED";
	x_origin[nlocal][0] = buf[0];
	x_origin[nlocal][1] = buf[1];
	x_origin[nlocal][2] = buf[2];
	return 3;
//	std::cout << "UNPACK_EXCHANGE END";
}

double FixAddPF::memory_usage() {
	int nmax = atom->nmax;
	double bytes = 0.0;
	bytes += nmax * 3 * sizeof(double);
	return bytes;
}

void FixAddPF::set_arrays(int i) {
	memset(x_origin[i], 0, sizeof(double)*3);
} 


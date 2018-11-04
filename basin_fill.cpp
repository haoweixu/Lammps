/*------------------------------------Haowei Xu--------------------------------------
12/01/2017
------------------------------------------------------------------------------------*/

#include <stdlib.h>
#include <string.h>
#include "run.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "fix_minimize.h"
#include "compute.h"
#include "neighbor.h"
#include "pair.h"
#include "domain.h"
#include "update.h"
#include "force.h"
#include "integrate.h"
#include "modify.h"
#include "output.h"
#include "finish.h"
#include "input.h"
#include "timer.h"
#include "error.h"
#include "basin_fill.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace LAMMPS_NS;

/*------------------------------------------------------------------------------------
syntax:

basin_fill  N  etol  ftol  keyword values

N = number of local minima you want to find
etol: stopping tolerance for energy from added penalty functions, unitless (phi_pf / Phi_0)
ftol: stopping tolerance for force that comes from genuine potential (without pf), force units

keyword = min_tol  or pf or run_step (both are optional) 

min_tol: tolerance for minimization.
value = min_etol, min_ftol, maxiter, maxeval (see LAMMPS manual for more details on minimize command ) 
default values:
min_etol = 1e-4
min_ftol = 1e-6
maxiter = 100
maxeval = 1000

pf: parameters for added pf
value = W, sigma, amplitude and variance for the gaussian
default values:
W = 1
sigma = 0.1

run_step: how many steps to run after a pf is added
value = n_step
default value:
n_step = 10
-------------------------------------------------------------------------------------*/

Basin_Fill::Basin_Fill(LAMMPS *lmp) : Pointers(lmp), n_minima(0) {
	E_fly = new vector<double>;
}

void Basin_Fill::command(int narg, char **arg) {
	
	int eflag = 1, vflag = 1;	

	if (narg < 3) error->all(FLERR, "Illegal basin_fill command");
	
	n_minima = force->bnumeric(FLERR, arg[0]);

//	cout << "NUMBER OF MINIMA" << n_minima << endl;	

	double fill_etol = force->numeric(FLERR, arg[1]);
	double fill_ftol = force->numeric(FLERR, arg[2]);

//	cout << "LINE 68" << endl;

	
	// default values 
	char * min_arg[4];
	min_arg[0] = new char[20]; sprintf(min_arg[0], "%f", 1e-4);
	min_arg[1] = new char[20]; sprintf(min_arg[1], "%f", 1e-6);
	min_arg[2] = new char[20]; sprintf(min_arg[2], "%d", 100);
	min_arg[3] = new char[20]; sprintf(min_arg[3], "%d", 1000);
	int min_narg = 4;
	
	double W = 1, sigma = 0.1;


	char * run_arg[1];
	run_arg[0] = new char[10]; sprintf(run_arg[0], "%d", 10);

//	cout << "LINE 81" << endl;
	
	// parse optional args
	int iarg = 1;
	while (iarg < narg) {
		if( strcmp(arg[iarg], "min_tol") == 0 ) {
			if (iarg+4 > narg) error->all(FLERR, "Illegal basin_fill command");
			for(int j=0; j<4; ++j)
				strcpy(min_arg[j], arg[iarg+j+1]);
			iarg += 4;
		}
		if( strcmp(arg[iarg], "pf") == 0 ) {
			if (iarg+2 > narg) error->all(FLERR, "Illegal basin_fill command");
			W = force->numeric(FLERR, arg[iarg+1]);
			sigma = force->numeric(FLERR, arg[iarg+2]);
			iarg += 2;
		}
		if( strcmp(arg[iarg], "run_step") == 0 ) {
			strcpy(run_arg[0], arg[iarg+1]);
			iarg += 1;
		}
		iarg += 1;
	}

//	cout << "LINE 101" << endl;
	
	char * addpf_arg[5];
	addpf_arg[1] = new char[20]; sprintf(addpf_arg[1], "all");
	addpf_arg[2] = new char[20]; sprintf(addpf_arg[2], "add_pf");
	addpf_arg[3] = new char[20]; sprintf(addpf_arg[3], "%f", W);
	addpf_arg[4] = new char[20]; sprintf(addpf_arg[4], "%f", sigma);
	int addpf_narg = 5;

//	cout << "LINE 110" << endl;
	
	Input::CommandCreatorMap * cmd_map = input->command_map;
	Input::CommandCreator min_creator = (*cmd_map)["minimize"];
	Input::CommandCreator run_creator = (*cmd_map)["run"];
	
	// start to find minima
	double cri_f, cri_e;
	int n_pf = 0; // total penalty functions added
	E_min = new double[n_minima]; // record the energy at minima
	double E_total, E_pf;
	double e_check, f_check;
	
	min_creator(lmp, min_narg, min_arg);
	
	int id = modify->find_compute("thermo_pe");
	class Compute *total_pe_compute = modify->compute[id]; // the compute instance that computes thermo_pe.
	
	E_min[0] = total_pe_compute->compute_scalar();
	
	for(int i_minima=1; i_minima < n_minima; ++i_minima){
		cri_e = 0;
		cri_f = 0;
		
		while ( !cri_f || !cri_e ){
			cri_e = 0;
			cri_f = 0;
			
			addpf_arg[0] = new char[30]; sprintf(addpf_arg[0], "%d", n_pf);
			modify->add_fix(addpf_narg, addpf_arg); // add a new pf
			n_pf += 1;
			
			run_creator(lmp, 1, run_arg);
			
			min_creator(lmp, min_narg, min_arg); // minimize
			
			E_total = total_pe_compute->compute_scalar(); // compute total energy
			E_pf = modify->thermo_energy(); // compute energy from pf
			

			clear_force(); // now clear force

			if(modify->n_pre_force) {
				modify->pre_force(vflag);
			}
			
			force->pair->compute(eflag, vflag); // compute force without pf
			
			if(modify->n_pre_reverse)
				modify->pre_reverse(eflag, vflag);

			if(force->newton) 
				comm->reverse_comm();

			// now check whether a new minimum is found
			e_check = abs(E_pf) / abs(E_total - E_pf);

			E_fly->push_back(E_total - E_pf);
 
			if ( e_check < fill_etol ) cri_e = 1;
			
			f_check = f_sqr();
			
			cout << "iter  " << n_pf << "\t" <<  "F_sqr_total  " << f_check << "\t" << "E_pair " << E_total-E_pf << "\t" << "E_pf  " << E_pf << endl;
	
			if ( f_check < (fill_ftol*fill_ftol) ) cri_f = 1;
		}
		
		E_min[i_minima] = E_total - E_pf;
	}
	
	final_staff();
		
}


void Basin_Fill::clear_force()
{
  size_t nbytes;

//  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
//      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
//     if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
//      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
//      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes ) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
//        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
//        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}

double Basin_Fill::f_sqr()
{
	int i,n;
	double *fatom;
	int nvec = 3*atom->nlocal;
	int nlocal = atom->nlocal;
	int nglobal;
	double *fvec; 
	if(nvec) fvec = atom->f[0];

	double local_norm2_sqr = 0.0;
	for (i = 0; i < nvec; i++) local_norm2_sqr += fvec[i]*fvec[i];
	double norm2_sqr = 0.0;
	MPI_Allreduce(&local_norm2_sqr,&norm2_sqr,1,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(&nlocal, &nglobal, 1, MPI_INT, MPI_SUM, world);
	
	cout << "NGLOBAL  " << nglobal <<endl;

	return norm2_sqr/nglobal;
}

void Basin_Fill::final_staff() {
	ofstream myfile;
	myfile.open("Basin_Fill");
	for (int i=0; i<n_minima; ++i)
		myfile << i << "\t" << E_min[i] << "\n";
	myfile.close();
	
	myfile.open("E_fly");
	vector<double>::const_iterator i;
	int j = 0;
	for (i=E_fly->begin(); i != E_fly->end() ; ++i)
		myfile << j << "\t" << *i << "\n";
		++j;
	myfile.close();
}





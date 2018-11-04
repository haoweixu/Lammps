/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Yunwei Mao (XJTU) 	Haowei XU (MIT)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "DA_Metro.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "respa.h"
#include "comm.h"
#include "input.h"
#include "irregular.h"
#include "variable.h"
#include "random_mars.h"
#include "random_park.h"
#include "memory.h"
#include "error.h"
#include "group.h"
#include<iostream>
#include<fstream>
#include<cmath>

using namespace LAMMPS_NS;
using namespace std;

#define V_MAX 1.0
#define SMALL -1e20
#define LARGE  1e20
#define INVOKED_PERATOM 8
/* ---------------------------------------------------------------------- */
DA_Metro::DA_Metro(LAMMPS *lmp) : Pointers(lmp) {}


void DA_Metro::command(int narg, char **arg) {
  if (narg < 7) error->all(FLERR,"Illegal DA_Metro command. DA_Metro pe/atom_id NumALL xratio yratio zratio seed sigma.");
   
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int n = strlen(arg[0]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[0]);
  NUM = ATOBIGINT(arg[1]);
  xratio = atof(arg[2]);
  yratio = atof(arg[3]);
  zratio = atof(arg[4]);
  seed = atoi(arg[5]);
	sigma = atof(arg[6]);
	
	// default values
	e_tol = 1e-6;
	f_tol = 1e-6;
	max_iter = 500;
	max_eval = 1000;

	// parse optional args
	int iarg = 1;
	while (iarg < narg) {
		if( strcmp(arg[iarg], "min_tol") == 0 ) {
			if (iarg+4 > narg) error->all(FLERR, "Illegal fix_DA_Metro command");
			e_tol = atof(arg[iarg+1]);
			f_tol = atof(arg[iarg+2]);
			max_iter = atof(arg[iarg+3]);
			max_eval = atof(arg[iarg+4]);
			iarg += 4;
		}
		iarg += 1;
	}

  nmax = 0;

  if (NUM < 0.0){
	   if (comm->me == 0)
       error->warning(FLERR,"Set Num=0");
	   NUM=0;
  }
  if (xratio<0.0||yratio<0.0||zratio<0.0){
        if (comm->me == 0)
           error->warning(FLERR,"Set ratio=0.0");
        xratio = 0;
	      yratio = 0;
	      zratio = 0;
  }
  if (seed<0.0){
        if (comm->me == 0)
           error->warning(FLERR,"Set seed=0");
        seed=0;   
  }
  array_buf=NULL;
  
  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA does not exist");
  pe = modify->compute[icompute]; 
  
  int i;
  double EngCur = 0;
  
  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;

  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;
   
 // create a full array & buf if has not been created

  if (array_buf == NULL) memory->create(array_buf,natoms,"DA:array_buf");

  int *recvcounts, *displs;

  // gather the tag & property data info of all the atoms on each proc

  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
  

  sort(array_buf,0,natoms-1);
  /*if (comm->me == 0 && screen){
   for(int j=0;j<natoms;j++)
	fprintf(screen,"%f\n", array_buf[j]);
  }*/
  
  PENUM = array_buf[natoms-NUM-1];
  //if (comm->me == 0 && screen) fprintf(screen,"PENUM = %f\n", PENUM);

  //DA.
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
	int *tag = atom->tag;

  RanPark *random = new RanPark(lmp,1);
  double dx = domain->xprd*xratio;
  double dy = domain->yprd*yratio;
  double dz = domain->zprd*zratio;
  //if (comm->me == 0 && screen) fprintf(screen,"%f%f%f\n", dx, dy, dz);
	
	// store orginal coordinates of all atoms
	double *x_old, *send_buf, *recv_buf;
  x_old = new double [3*natoms];
  send_buf = new double [4*nlocal];
  recv_buf = new double [4*natoms];
	int find = 0, tag_id;
  
	if ( !atom->tag_consecutive() )
		error->all(FLERR, "Atoms should have unique ID from 1 to N");
  for (i = 0; i < nlocal; i++) {
    send_buf[4*i] = (double) tag[i];
    send_buf[4*i+1] = x[i][0];
		send_buf[4*i+2] = x[i][1];
    send_buf[4*i+3] = x[i][2];
  }
 
  for (int i=0; i<nprocs; ++i) {
    recvcounts[i] *= 4;
    displs[i] *= 4; 
  }
 
  MPI_Allgatherv(send_buf, 4*nlocal, MPI_DOUBLE, recv_buf, recvcounts,
    displs, MPI_DOUBLE, world); 
  
  for (int i=0; i<natoms; ++i) {
    tag_id = (int) (recv_buf[4*i] - 1);
    x_old[3*tag_id]   = recv_buf[4*i+1];
    x_old[3*tag_id+1] = recv_buf[4*i+2];
    x_old[3*tag_id+2] = recv_buf[4*i+3];
  }
  
  delete [] send_buf;
  delete [] recv_buf;
  delete [] recvcounts;
  delete [] displs;

	// minimizer creator
	Input::CommandCreatorMap * cmd_map = input->command_map;
	Input::CommandCreator min_creator = (*cmd_map)["minimize"];
  Input::CommandCreator run_creator = (*cmd_map)["run"];
	
  char *run_arg[1];
  run_arg[0] = new char[32]; sprintf(run_arg[0], "%d", 1);

	char *min_arg[4];
	min_arg[0] = new char[32]; sprintf(min_arg[0], "%f", e_tol);
	min_arg[1] = new char[32]; sprintf(min_arg[1], "%f", f_tol);
	min_arg[2] = new char[32]; sprintf(min_arg[2], "%d", max_iter);
	min_arg[3] = new char[32]; sprintf(min_arg[3], "%d", max_eval);
	int min_narg = 4;  

	// total_pe computer
	int total_pe_id = modify->find_compute("thermo_pe");
  if (total_pe_id < 0) error->all(FLERR,"pe_total ID for fix DA does not exist");
	class Compute *total_pe_compute = modify->compute[total_pe_id];
	
	double E_old = total_pe_compute->compute_scalar();
	double E_new;
	
	double ddx, ddy, ddz;
	int trial = 1;
	int move_id[NUM], n_move = 0;
  int flag = 0;
	while(flag==0) {
		// first DA trial
		if (trial == 1) {
			for (i = 0; i < nlocal; i++) {
			// DA move
				if (mask[i] & PE[i]>PENUM){
					random->reset(trial*seed,x[i]);
					ddx = dx * (random->uniform()-0.5);
					ddy = dy * (random->uniform()-0.5);
					ddz = dz * (random->uniform()-0.5);
					x[i][0] += ddx;
					x[i][1] += ddy;
					x[i][2] += ddz;
						
					// store id of atoms moved in DA
					move_id[n_move] = tag[i];
					n_move += 1;
					if( n_move == (NUM-1) )	
						break;	
				}
			}
		}
		// non-initial DA trials
		else {
      if (comm->me == 0)
        cout << "ENTER SECOND TRIAL" << endl;
			for (i = 0; i < nlocal; i++) {
				// go back to original position
				tag_id = tag[i];
				x[i][0] = x_old[3*tag_id];
				x[i][1] = x_old[3*tag_id+1];
				x[i][2] = x_old[3*tag_id+2];
//				if (comm->me == 0)
//          cout <<"tag\t" << tag_id << "\t" << x[i][0] << endl;
				// DA move
				for (int j=0; j<NUM; ++j) {
					if(tag_id == move_id[j]) {
						random->reset(trial*seed,x[i]);
						ddx = dx * (random->uniform()-0.5);
						ddy = dy * (random->uniform()-0.5);
						ddz = dz * (random->uniform()-0.5);
						x[i][0] += ddx;
						x[i][1] += ddy;
						x[i][2] += ddz;
						break;
					}
				}
			}
    }
		
		// move atoms back inside simulation box and to new processors
		// use remap() instead of pbc() in case atoms moved a long distance
		// use irregular() in case atoms moved a long distance

		tagint *image = atom->image;
		for (i = 0; i < nlocal; i++) domain->remap(x[i],image[i]);

		if (domain->triclinic) domain->x2lamda(atom->nlocal);
		domain->reset_box();
		Irregular *irregular = new Irregular(lmp);
		irregular->migrate_atoms();
		delete irregular;
		if (domain->triclinic) domain->lamda2x(atom->nlocal);

		// check if any atoms were lost

		bigint nblocal = atom->nlocal;
		MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
		if (natoms != atom->natoms && comm->me == 0) {
			char str[128];
			sprintf(str,"Lost atoms via DA: original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
			error->warning(FLERR,str);
		}
		
		// minimize
//		if (comm->me == 0) cout << "minimzation starts\t" << trial << endl;
    min_creator(lmp, min_narg, min_arg); 
//    run_creator(lmp, 1, run_arg);
//		if (comm->me == 0) cout << "minimzation ends" << endl;

		// Metropolis, decide whether to accept this DA step or not.
		E_new = total_pe_compute->compute_scalar();
		if (comm->me == 0) {
			cout << "ENERGY\t" << E_old << "\t" << E_new << endl;
			if (E_new < E_old) {
        cout << "Case A: DA movement accepted after " << trial << "trials" << endl;
        flag = 1;
			}
			else {
	  		random->reset(trial*seed, &E_new);
				double Ranf = random->uniform();
      	cout << "Ranf\t" << Ranf << endl;
				if (Ranf < exp(-(E_new-E_old)/sigma)) {
          cout << "Case B: DA movement accepted after " << trial << "trials" << endl;
					flag = 1;
				}
			}
		}		
		trial += 1;
    
    int flag_send = flag;
    MPI_Allreduce(&flag_send, &flag, 1, MPI_INT, MPI_SUM, world);
  }
  
  
  delete [] id_pe;
  delete random;
  delete [] x_old;
  memory->destroy(array_buf);
  
  if (comm->me == 0) cout << "DA_METRO ENDS" << endl;
  MPI_Barrier(world); 
  
}


void DA_Metro::sort(double * a_one, bigint low, bigint high)
{
  bigint i, j;
  double t;
  if(low<high)
  {
    i=low;j=high; t = a_one[low];
    while (i<j)
    {
      while(i<j&&a_one[j]>t)
        j--;
      if (i<j)
      {
          a_one[i]=a_one[j];
          i++;
      }
      while(i<j&&a_one[i]<=t)
          i++;
      if(i<j)
      {
          a_one[j]=a_one[i];
          j--;
      }
    }
    a_one[i]=t;
    sort(a_one,low, i-1);
    sort(a_one, i+1, high);
  }
}

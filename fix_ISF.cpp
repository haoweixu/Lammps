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
   Contributing author: Haowei Xu (MIT)
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "fix_ISF.h"
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
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixISF::FixISF(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) 
{
  if (narg < 4) error->all(FLERR,"Illegal fix ISF command. Fix 1 all ISF q TYPE");

  q = atof(arg[3]);
  
  TYPE = atoi(arg[4]);
// TYPE = 0 if all types shall be included

  
  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);
  
  
}


FixISF::~FixISF()
{
  if (comm->me == 0) {
    ofstream myfile;
    myfile.open("ISF");
    
    vector<int>::iterator i;
    for (i=timestep.begin(); i!=timestep.end(); i++)
      myfile << *i << "\t";
    myfile << "\n";
    
    vector<double>::iterator j;
    for ( j=isf_r.begin(); j!=isf_r.end(); j++ )
      myfile << *j << "\t";
    myfile << "\n";
    
    for ( j=isf_i.begin(); j!=isf_i.end(); j++ )
      myfile << *j << "\t";
    myfile << "\n";
  }
  
  int natoms = atom->natoms;
  for (int i=0; i<natoms; ++i) {
    delete [] xold[i];
		delete [] xnew[i];
	}
	delete [] type_global;
 
}


int FixISF::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

void FixISF::setup(int vflag)
{
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  
  double **x = atom->x;
  int *tag = atom->tag;
  tagint *image = atom->image;
  int *type = atom->type;
  
  xold = new double * [natoms];
  xnew = new double * [natoms];
  for (int i=0; i<natoms; ++i) {
    xold[i] = new double [3];
    xnew[i] = new double [3];
  }
  type_global = new int [natoms];
  
  
  if ( !atom->tag_consecutive() )
		error->all(FLERR, "Atoms should have unique IDs, ranging from 1 to N");
  
  double *send_buff, *recv_buff;
  send_buff = new double [5*nlocal];
  recv_buff = new double [5*natoms];
  
  double *unwrap = new double [3];
  for (int i=0; i<nlocal; ++i) {
    domain->unmap(x[i], image[i], unwrap);
    send_buff[5*i]    = (double) tag[i];
    send_buff[5*i+1]  = unwrap[0];
    send_buff[5*i+2]  = unwrap[1];
    send_buff[5*i+3]  = unwrap[2];
    send_buff[5*i+4]  = (double) type[i];
  }
  
  int *recvcounts = new int [nprocs];
  int *displs = new int [nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);
  
  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  for (int i=0; i<nprocs; ++i) {
    recvcounts[i] *= 5;
    displs[i] *= 5; 
  }
  
  MPI_Allgatherv(send_buff, 5*nlocal, MPI_DOUBLE, recv_buff, recvcounts,
    displs, MPI_DOUBLE, world); 
  
  int tag_id;
  for (int i=0; i<natoms; ++i) {
    tag_id = (int) (recv_buff[5*i] - 1);
    xold[tag_id][0] = recv_buff[5*i+1];
    xold[tag_id][1] = recv_buff[5*i+2];
    xold[tag_id][2] = recv_buff[5*i+3];
    type_global[tag_id] = recv_buff[5*i+4];
  }
  
  delete [] send_buff;
  delete [] recv_buff;
  delete [] recvcounts;
  delete [] displs;
  delete [] unwrap;
  
}

/* ---------------------------------------------------------------------- */

void FixISF::min_setup(int vflag)
{
  setup(vflag);
}

/* ---------------------------------------------------------------------- */

void FixISF::post_force(int vflag)
{
  int cur_step = update->ntimestep;
  int check;
  check = check_timestep(cur_step);
  if (check != 0)
    return;
  
  if ( !atom->tag_consecutive() )
		error->all(FLERR, "Atoms should have unique IDs, ranging from 1 to N");
  
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;
  
  // gather new atom configuration
  double **x = atom->x;
  int *tag = atom->tag;
  tagint *image = atom->image;
  
  double *send_buff, *recv_buff;
  send_buff = new double [4*nlocal];
  recv_buff = new double [4*natoms];
  
  double *unwrap = new double [3];
  for (int i=0; i<nlocal; ++i) {
    domain->unmap(x[i], image[i], unwrap);
    send_buff[4*i]    = (double) tag[i];
    send_buff[4*i+1]  = unwrap[0];
    send_buff[4*i+2]  = unwrap[1];
    send_buff[4*i+3]  = unwrap[2];
  }
  
  int *recvcounts = new int [nprocs];
  int *displs = new int [nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);
  
  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  for (int i=0; i<nprocs; ++i) {
    recvcounts[i] *= 4;
    displs[i] *= 4; 
  }
  
  MPI_Allgatherv(send_buff, 4*nlocal, MPI_DOUBLE, recv_buff, recvcounts,
    displs, MPI_DOUBLE, world); 
  
  int tag_id;
  for (int i=0; i<natoms; ++i) {
    tag_id = (int) (recv_buff[4*i] - 1);
    xnew[tag_id][0] = recv_buff[4*i+1];
    xnew[tag_id][1] = recv_buff[4*i+2];
    xnew[tag_id][2] = recv_buff[4*i+3];
  }
  
  delete [] send_buff;
  delete [] recv_buff;
  delete [] recvcounts;
  delete [] displs;
  delete [] unwrap;
  
  // calculate intermediate scattering function
  double cur_isf_i, cur_isf_r;
  cur_isf_i = 0; cur_isf_r = 0;
  
  double delx, dely, delz;
  int count = 0;
  for (int i=0; i<natoms; ++i){
    if (TYPE != 0 && type_global[i] != TYPE)
      continue;
    
    delx = xnew[i][0] - xold[i][0];
    dely = xnew[i][1] - xold[i][1];
    delz = xnew[i][2] - xold[i][2];
    
    cur_isf_r += ( cos(q*delx) + cos(q*dely) + cos(q*delz) ) / 3.0;
    cur_isf_i += ( sin(q*delx) + sin(q*dely) + sin(q*delz) ) / 3.0;  

    count += 1;
  }
  
  cur_isf_i /= count;
  cur_isf_r /= count;

//  cout << "number of A atoms " << count << endl;
  
  timestep.push_back(cur_step);
  isf_i.push_back(cur_isf_i);
  isf_r.push_back(cur_isf_r); 
  
}


/* ---------------------------------------------------------------------- */

void FixISF::min_post_force(int vflag)
{
  post_force(vflag);
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixISF::memory_usage()
{
  return 0;
}


int FixISF::check_timestep(int cur_step)
{
  if (cur_step <= 1e3)
    return 0;
  else if (cur_step>1e3 && cur_step<=1e4)
    return cur_step % (int)1e1;
  else if (cur_step>1e4 && cur_step<=1e5)
    return cur_step % (int)1e2;
  else if (cur_step>1e5 && cur_step<=1e6)
    return cur_step % (int)1e3;
  else if (cur_step>1e6 && cur_step<=1e7)
    return cur_step % (int)1e4;
	else if (cur_step>1e7 && cur_step<=1e8)
		return cur_step % (int)1e5;
  else
    return cur_step % (int)1e6;
}




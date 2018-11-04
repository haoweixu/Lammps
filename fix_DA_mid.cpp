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
   Contributing author: Yunwei Mao (MIT)
------------------------------------------------------------------------- */

#include "mpi.h"
#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "fix_DA_mid.h"
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

using namespace LAMMPS_NS;
using namespace FixConst;

#define V_MAX 1.0
#define SMALL -1e20
#define LARGE  1e20
#define SIZESCALE 1e10
#define INVOKED_PERATOM 8
#define MAXDIS 1e20
#define	ZL 0.02
/* ---------------------------------------------------------------------- */

DAmid::DAmid(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 13) error->all(FLERR,"Illegal fix DAmid command. Fix 3 all DAmid NumALL xratio yratio zratio mindis maxstep maxenegy minenergy seed.");

  global_freq = 1;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  int n = strlen(arg[3]) + 1;
  id_pe = new char[n];
  strcpy(id_pe,arg[3]);
  NUM = ATOBIGINT(arg[4]);
  xratio = atof(arg[5]);
  yratio = atof(arg[6]);
  zratio = atof(arg[7]);
  mindis = atof(arg[8]);
  maxstep = atoi(arg[9]);
  maxenergy = atof(arg[10]);
  minenergy = atof(arg[11]);
  seed = atoi(arg[12]);


  nmax = 0;
  if(comm->me==0&&screen)
	fprintf(screen, "parameters are %i %f %f %f %f %i %f %f %i\n", NUM,xratio,yratio,zratio,mindis,maxstep,maxenergy,minenergy,seed);
  if (NUM < 0.0){
	if (comm->me == 0)
           error->warning(FLERR,"Set Num=0");
	NUM=0;
   }
  if (minenergy >=0)
	if (comm->me==0)
	   error->all(FLERR, "minenergy is set to be a wrong number...try again...");
  if (minenergy > maxenergy) {
        if (comm->me == 0)
           error->warning(FLERR,"Set minenergy=-7.85; maxenergy=Current energy");
	minenergy=-7.85;
	maxenergy=-7.5;
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
  array_dis=NULL;
  // random number generator, same for all procs

  random = new RanPark(lmp,seed);
}

/* ---------------------------------------------------------------------- */

DAmid::~DAmid()
{
  delete []id_pe;
  delete random;
  memory->destroy(array_buf);
  memory->destroy(array_dis);
}

/* ---------------------------------------------------------------------- */

int DAmid::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void DAmid::init()
{

  int icompute = modify->find_compute(id_pe);
  if (icompute < 0) error->all(FLERR,"PE ID for fix DA does not exist");
  pe = modify->compute[icompute]; 
}

/* ---------------------------------------------------------------------- */

void DAmid::setup(int vflag)
{
  post_force(vflag);
}

void DAmid::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void DAmid::post_force(int vflag)
{
  int i,j,k,flagall,tep,nattempt;
  double alldismin,factor,dpx,dpy,dpz,dx,dy,dz;
  double EngCur = 0;
  double coord[3];
  int alltep=0;

  int nlocal = atom->nlocal;
  bigint natoms = atom->natoms;
  double *xex = new double[3*nlocal];
  double *xey = NULL;

  if (!(pe->invoked_flag & INVOKED_PERATOM)){
    pe->compute_peratom();
    pe->invoked_flag |= INVOKED_PERATOM;
  }
  double *PE = pe->vector_atom;

  // create a full array & buf if has not been created

  if (array_buf == NULL) memory->create(array_buf,natoms,"DA:array_buf");

  int *recvcounts,*displs;

  // gather the tag & property data info of all the atoms on each proc

  recvcounts = new int[nprocs];
  displs = new int[nprocs];
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);

  displs[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs[iproc] = displs[iproc-1] + recvcounts[iproc-1];
  }
  MPI_Allgatherv(&PE[0],nlocal,MPI_DOUBLE,array_buf,recvcounts,displs,MPI_DOUBLE,world); 
  delete [] recvcounts;
  delete [] displs;

  //calculate the avarage energy of atoms in the system
  for(i = 0;i< natoms; i++)
    EngCur = EngCur + array_buf[i];
  EngCur = EngCur/natoms;

  // calculate the corresponding factor that used
  factor = (EngCur-minenergy)/(maxenergy-minenergy);
  if (factor>1.0) factor=1.0;
  if (factor<0.0) factor=0.0;
  if (comm->me == 0 && screen)
	fprintf(screen, "factor is reset to be %f\n", factor);
 
  dpx = domain->xprd*xratio;
  dpy = domain->yprd*yratio;
  dpz = domain->zprd*zratio;
  
  // find the critical potential value for DA
  sort(array_buf,0,natoms-1);
  NUM = NUM;
  PENUM = array_buf[natoms-NUM-1];

  // find all bad atoms and store their position information to xex
  double **x = atom->x;
  int *mask = atom->mask;
  int *type = atom->type;
  tep=0;
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (PE[i]>PENUM){
        xex[3*tep+0] = x[i][0];
        xex[3*tep+1] = x[i][1];
        xex[3*tep+2] = x[i][2];
	tep++;
      }
    }
  }
  
  // transfer xex to xey reduce the size of memory.
  xey = new double[3*tep];
  for(i=0;i<tep;i++){
	xey[3*i+0]=xex[3*i+0];
	xey[3*i+1]=xex[3*i+1];
	xey[3*i+2]=xex[3*i+2];}
  delete []xex;

 // get all number of DA atom and this shall be equal to NUM.
  MPI_Allreduce(&tep,&alltep,1,MPI_INT,MPI_SUM,world);
  if (comm->me == 0 && screen)
        fprintf(screen, "alltep to be %i, NUM to be %i\n", alltep, NUM);
 
  if (alltep!=NUM)
        error->all(FLERR,"Illegal fix DAmid process. alltep shall equal to NUM.");

  // create a full array & dis if has not been created
  if (array_dis == NULL) memory->create(array_dis,3*NUM,"DA:array_dis");

  int *recvcounts2,*displs2;

  // gather the tag & property data info of all the atoms on each proc
  // store all position information in array_dis.
  recvcounts2 = new int[nprocs];
  displs2 = new int[nprocs];
  MPI_Allgather(&tep,1,MPI_INT,recvcounts2,1,MPI_INT,world);
  for(int ipl=0;ipl<nprocs;ipl++)
	recvcounts2[ipl]=3*recvcounts2[ipl];

  displs2[0] = 0;
  for (int iproc = 1; iproc < nprocs; iproc++) {
    displs2[iproc] = displs2[iproc-1] + recvcounts2[iproc-1];
  }

  MPI_Allgatherv(&xey[0],3*tep,MPI_DOUBLE,array_dis,recvcounts2,displs2,MPI_DOUBLE,world);
  delete [] recvcounts2;
  delete [] displs2;

  //begin to do DA
  for(i=0;i<NUM;i++){
    // read inf. into xcyczc
    double xc=array_dis[3*i+0];
    double yc=array_dis[3*i+1];
    double zc=array_dis[3*i+2];
    //if (comm->me==0)
    //fprintf(screen, "xcyczc is %f %f %f\n", xc,yc,zc);
    nattempt=0; flagall=0;

    //do loop
    dx=0.0;dy=0.0;dz=0.0;
    while (nattempt<maxstep&&flagall==0){
	 nattempt++;
         // find dx dy dz for every atom randomly according Rank.
  	 dx = dpx*(random->uniform()-0.5);
  	 dy = dpy*(random->uniform()-0.5);
  	 dz = dpz*(random->uniform()-0.5);
         //fprintf(screen, "dx dy dz attempt as %f, %f, %f\n", dx, dy,dz);
	 //double betap=sqrt(disp/(dx*dx+dy*dy+dz*dz));
	 //dx=dx*betap; dy=dy*betap; dz=dz*betap;
         
         //do an attempted mov and store in xyzcc
	 double xcc = xc+dx;
	 double ycc = yc+dy;
	 double zcc = zc+dz;
	 double tdism=MAXDIS;
	 for (int ia = 0; ia < nlocal; ia++) {
           // calculate the dis locally for each proc
           double delx = xc - x[ia][0];
           double dely = yc - x[ia][1];
           double delz = zc - x[ia][2];
	   double rsq = sqrt(delx*delx + dely*dely + delz*delz);
 
           //not account for the original pos for this atom.
           if (rsq>ZL){
             delx = xcc - x[ia][0];
             dely = ycc - x[ia][1];
             delz = zcc - x[ia][2];
             domain->minimum_image(delx,dely,delz);
             double rsq = sqrt(delx*delx + dely*dely + delz*delz);
             tdism = (tdism>rsq)?rsq:tdism;}
         }
    	 MPI_Allreduce(&tdism,&alldismin,1,MPI_DOUBLE,MPI_MIN,world);
	 if(alldismin>=mindis)
		flagall=1;
       }

       //fprintf(screen, "flagall is %i,nattempt is %i\n", flagall, nattempt);
       //fprintf(screen, "dx dy dz set as %f, %f, %f\n", dx, dy,dz);

       //update the coordinate
       xc=xc+dx;yc=yc+dy;zc=zc+dz;
       for (j = 0; j < nlocal; j++) {
    	if (mask[j] & groupbit){
	 double lx=array_dis[3*i+0]-x[j][0];
	 double ly=array_dis[3*i+1]-x[j][1];
	 double lz=array_dis[3*i+2]-x[j][2];
	 double dist=sqrt(lx*lx+ly*ly+lz*lz);
	 //fprintf(screen, "comm->me=%i,dis= %f\n", comm->me, dist);
	 if (dist<ZL){
	 x[j][0]=xc;
	 x[j][1]=yc;
	 x[j][2]=zc;
         //double ppj=sqrt(dx*dx+dy*dy+dz*dz);
	 //fprintf(screen, "%f %f %f %f\n", xc,yc,zc,ppj);
         }
	}
     }
     //end of update new coordinate.
  }

  // move atoms back inside simulation box and to new processors
  // use remap() instead of pbc() in case atoms moved a long distance
  // use irregular() in case atoms moved a long distance

  double **xf = atom->x;
  tagint *image = atom->image;
  for (i = 0; i < nlocal; i++) domain->remap(xf[i],image[i]);

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
    sprintf(str,"Lost atoms via DAmid original " BIGINT_FORMAT
            " current " BIGINT_FORMAT,atom->natoms,natoms);
    error->warning(FLERR,str);
  }

  delete []xey;

}

void DAmid::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double DAmid::memory_usage()
{
  double bytes = 0.0;
  return bytes;
}

void DAmid::sort(double * a_one, bigint low, bigint high)
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

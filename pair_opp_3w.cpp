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
Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair.h"
#include "pair_opp_3w.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairOPP3w::PairOPP3w(LAMMPS *lmp) : Pair(lmp)
{
	//respa_enable = 1;
	//writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairOPP3w::~PairOPP3w()
{
	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);

		memory->destroy(cut);
		memory->destroy(k);
		memory->destroy(phi);
		memory->destroy(offset);
	}
}

/* ---------------------------------------------------------------------- */

void PairOPP3w::compute(int eflag, int vflag)
{
	int i, j, ii, jj, inum, jnum, itype, jtype;
	double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
	int *ilist, *jlist, *numneigh, **firstneigh;

	evdwl = 0.0;
	if (eflag || vflag) ev_setup(eflag, vflag);
	else evflag = vflag_fdotr = 0;

	double **x = atom->x;
	double **f = atom->f;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	//double *special_lj = force->special_lj;
	int newton_pair = force->newton_pair;

	inum = list->inum;
	ilist = list->ilist;
	numneigh = list->numneigh;
	firstneigh = list->firstneigh;

	// loop over neighbors of my atoms

	for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		for (jj = 0; jj < jnum; jj++) {
			j = jlist[jj];
			//factor_lj = special_lj[sbmask(j)];
			j &= NEIGHMASK;

			delx = xtmp - x[j][0];
			dely = ytmp - x[j][1];
			delz = ztmp - x[j][2];
			double rsq = delx * delx + dely * dely + delz * delz;
			double r = sqrt(rsq);
			jtype = type[j];

			if (rsq < cutsq[itype][jtype]) {
				double r2inv = 1.0 / rsq;
				double r3inv = r2inv / r, r4inv = r2inv * r2inv;
				double r15inv = r4inv * r4inv * r4inv * r3inv;

				double phase = k[itype][jtype] * (r - 1.25) - phi[itype][jtype];
				
				fpair = (15 * r15inv + 3 * r3inv * cos(phase)
					+ k[itype][jtype] * r2inv * sin(phase)) * r2inv;

				f[i][0] += delx * fpair;
				f[i][1] += dely * fpair;
				f[i][2] += delz * fpair;
				if (newton_pair || j < nlocal) {
					f[j][0] -= delx * fpair;
					f[j][1] -= dely * fpair;
					f[j][2] -= delz * fpair;
				}

				if (eflag) {
					evdwl = r15inv + r3inv * cos(phase) - offset[itype][jtype];
				}

				if (evflag) ev_tally(i, j, nlocal, newton_pair,
					evdwl, 0.0, fpair, delx, dely, delz);
			}
		}
	}

	if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
allocate all arrays
------------------------------------------------------------------------- */

void PairOPP3w::allocate()
{
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
	memory->create(cut, n + 1, n + 1, "pair:cut");
	memory->create(phi, n + 1, n + 1, "pair:phi");
	memory->create(k, n + 1, n + 1, "pair:k");
	memory->create(offset, n + 1, n + 1, "pair:offset");
}

/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */

void PairOPP3w::settings(int narg, char **arg)
{
	if (narg != 1) error->all(FLERR, "Illegal pair_style command");

	cut_global = force->numeric(FLERR, arg[0]);

	// reset cutoffs that have been explicitly set

	if (allocated) {
		int i, j;
		for (i = 1; i <= atom->ntypes; i++)
			for (j = i; j <= atom->ntypes; j++)
				if (setflag[i][j]) cut[i][j] = cut_global;
	}
}

/* ----------------------------------------------------------------------
set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairOPP3w::coeff(int narg, char **arg)
{
	if (narg != 5)
		error->all(FLERR, "Incorrect args for pair coefficients");
	if (!allocated) allocate();

	int ilo, ihi, jlo, jhi;
	force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
	force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

	double k_one = force->numeric(FLERR, arg[2]);
	double phi_one = force->numeric(FLERR, arg[3]);
	double cut_one = force->numeric(FLERR, arg[4]);

	//double cut_one = cut_global;
	//if (narg == 6) cut_one = force->numeric(FLERR, arg[4]);

	int count = 0;
	for (int i = ilo; i <= ihi; i++) {
		for (int j = MAX(jlo, i); j <= jhi; j++) {
			k[i][j] = k_one;
			phi[i][j] = phi_one;
			cut[i][j] = cut_one;
			setflag[i][j] = 1;
			count++;
		}
	}

	if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
init specific to this pair style
------------------------------------------------------------------------- */

void PairOPP3w::init_style()
{
	int irequest;
	irequest = neighbor->request(this, instance_me);
}

/* ----------------------------------------------------------------------
neighbor callback to inform pair style of neighbor list to use
regular or rRESPA
------------------------------------------------------------------------- */

void PairOPP3w::init_list(int id, NeighList *ptr)
{
	if (id == 0) list = ptr;
	else if (id == 1) listinner = ptr;
	else if (id == 2) listmiddle = ptr;
	else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairOPP3w::init_one(int i, int j)
{
	if (setflag[i][j] == 0) {
		k[i][j] = mix_distance(k[i][i], k[j][j]);
		phi[i][j] = mix_distance(phi[i][i], phi[j][j]);
		cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
	}

	if (cut[i][j] > 0.0) {
		double c1inv = 1 / cut[i][j];
		double c3inv = c1inv * c1inv * c1inv;
		double c6inv = c3inv * c3inv;
		double c15inv = c6inv * c6inv * c3inv;
		double phase = k[i][j] * (cut[i][j] - 1.25) - phi[i][j];

		offset[i][j] = c15inv + c3inv * cos(phase);
	}
	else offset[i][j] = 0.0;

	offset[j][i] = offset[i][j];

	return cut[i][j];
}

/* ----------------------------------------------------------------------
proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairOPP3w::write_restart(FILE *fp)
{
	write_restart_settings(fp);

	int i, j;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			fwrite(&setflag[i][j], sizeof(int), 1, fp);
			if (setflag[i][j]) {
				fwrite(&k[i][j], sizeof(double), 1, fp);
				fwrite(&phi[i][j], sizeof(double), 1, fp);
				fwrite(&cut[i][j], sizeof(double), 1, fp);
			}
		}
}

/* ----------------------------------------------------------------------
proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairOPP3w::read_restart(FILE *fp)
{
	read_restart_settings(fp);
	allocate();

	int i, j;
	int me = comm->me;
	for (i = 1; i <= atom->ntypes; i++)
		for (j = i; j <= atom->ntypes; j++) {
			if (me == 0) fread(&setflag[i][j], sizeof(int), 1, fp);
			MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
			if (setflag[i][j]) {
				if (me == 0) {
					fread(&k[i][j], sizeof(double), 1, fp);
					fread(&phi[i][j], sizeof(double), 1, fp);
					fread(&cut[i][j], sizeof(double), 1, fp);
				}
				MPI_Bcast(&k[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&phi[i][j], 1, MPI_DOUBLE, 0, world);
				MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
			}
		}
}

/* ----------------------------------------------------------------------
proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairOPP3w::write_restart_settings(FILE *fp)
{
	fwrite(&cut_global, sizeof(double), 1, fp);
	fwrite(&offset_flag, sizeof(int), 1, fp);
	fwrite(&mix_flag, sizeof(int), 1, fp);
	fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairOPP3w::read_restart_settings(FILE *fp)
{
	int me = comm->me;
	if (me == 0) {
		fread(&cut_global, sizeof(double), 1, fp);
		fread(&offset_flag, sizeof(int), 1, fp);
		fread(&mix_flag, sizeof(int), 1, fp);
		fread(&tail_flag, sizeof(int), 1, fp);
	}
	MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
	MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
	MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
	MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
proc 0 writes to data file
------------------------------------------------------------------------- */

void PairOPP3w::write_data(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; i++)
		fprintf(fp, "%d %g %g\n", i, k[i][i], phi[i][i]);
}

/* ----------------------------------------------------------------------
proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairOPP3w::write_data_all(FILE *fp)
{
	for (int i = 1; i <= atom->ntypes; i++)
		for (int j = i; j <= atom->ntypes; j++)
			fprintf(fp, "%d %d %g %g %g\n", i, j, k[i][j], phi[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairOPP3w::single(int i, int j, int itype, int jtype, double rsq,
	double factor_coul, double factor_lj,
	double &fforce)
{
	double forcelj, philj;
	double r = sqrt(rsq);
	double r2inv = 1.0 / rsq;
	double r3inv = r2inv / r, r4inv = r2inv * r2inv;
	double r15inv = r4inv * r4inv * r4inv * r3inv;

	double phase = k[itype][jtype] * (r - 1.25) - phi[itype][jtype];

	forcelj = (15 * r15inv + 3 * r3inv * cos(phase)
		+ k[itype][jtype] * r2inv * sin(phase));
	
	fforce = forcelj*r2inv;

	philj = r15inv + r3inv * cos(phase) - offset[itype][jtype];
	return philj;
}

/* ---------------------------------------------------------------------- */

void *PairOPP3w::extract(const char *str, int &dim)
{
	dim = 2;
	if (strcmp(str, "k") == 0) return (void *)k;
	if (strcmp(str, "phi") == 0) return (void *)phi;
	return NULL;
}

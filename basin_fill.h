/*------------------------------------Haowei Xu--------------------------------------
12/01/2017
------------------------------------------------------------------------------------*/

#ifdef COMMAND_CLASS

CommandStyle(basin_fill, Basin_Fill)

#else

#ifndef LMP_BASIN_FILL_H
#define LMP_BASIN_FILL_H

#include "pointers.h"
#include <vector>

namespace LAMMPS_NS {
	
class Basin_Fill : protected Pointers {
	
	public:
	
	Basin_Fill(class LAMMPS *);
	void command(int, char **);
	
	private:
	
	double *E_min;
	std::vector<double> *E_fly;
	bigint n_minima;
	void clear_force();
	double f_sqr();
	void final_staff();
};
	
}

#endif
#endif

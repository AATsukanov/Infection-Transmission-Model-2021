/* VirusSimulator VSA, version A (works, but not optimized), 2020-2021. */
/* ==================================================================== */
/* Alexey Tsukanov, Alexandra Senjkevich, Maxim Fedorov and Nikolai Brilliantov */
/* Center for Computational and Data-Intensive Science and Engineering */
/* Skolkovo Institute of Science and Technology (Skoltech) */
/* using this code\data, please, cite our paper... */

#define DEF_USING_MPI
#define INITCONTROL
/*#define VERLET - no Euler*/

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>

#ifdef DEF_USING_MPI
#include <fstream>
#include <mpi.h>
#endif

#define real float
#define MAX_LENGTH_NAME 4096

using namespace std;

//Global MPI constants:
int comm_size = 1;
int comm_rank = 0;

//global series (parameters table):
real g_Io = 0.02; //0.004, 0.01, 0.02
int MAP_SINDEX = 1; //1, 2, 3

//stohastic behavior ctrl:
int debug_cross_n = 0;
int debug_cross_s = 0;
int debug_cross_e = 0;
int debug_cross_w = 0;
int debug_patience_n = 0;
int debug_patience_s = 0;
int debug_patience_e = 0;
int debug_patience_w = 0;
int debug_to_kasses = 0;

real gauss_random(real mean=0.0, real stddev=1.0)
{
    real u = ((real)rand()/((real)RAND_MAX))*2.0-1.0;
    real v = ((real)rand()/((real)RAND_MAX))*2.0-1.0;
    real r = u*u + v*v;

    if(r == 0 || r >= 1)
		return gauss_random(mean, stddev);

    return mean + stddev * u*sqrt(-2.0*log(r)/r);
}

//template <class Value>
real sign(real arg)
{
	if(arg == 0.0)
		return 0.0;
	if(arg > 0.0)
		return 1.0;
	return -1.0;
}

//pedestrian 'molecular' dynamics simulator - crossroad
class VSA
{
public:
	char** map;
	int nx;
	int ny;
	float map_scale;
	real area;
	//real free_area;
	int n; //number of pedestrians
	real rho; //' ~ density', [#/m^2]
	real L; //dimensions of computational domain
	real Lw; //street width
	//real hL; //half-length of the street
	//real hW; //half-width of the street
	real h; //half-of-spatial-step
	//simulation parameters:
	real dt; //timestep
	int step; //step number
	//force field parameters:
	//real* mass;
	//real mdelta;
	real mod_v_des; //absolute value of desired velocity
	real std_v_des; //stddev...
	real* v_des_x; //destination velocity
	real* v_des_y; //desired velocity
	real v_max; // [Helbing]: 1.3*mod_v_des;
	real chi; //chirality factor
	real tau; //relaxation time
	real B; //relaxation distance;
	real A; //
//	real dL;
//	real Uo;
	real D;
	real R; //characteristing size
	//downtime treatment:
	real* patience;
	real max_patience;
	real downtime_factor;
	//additional parameters:
	bool growing_R;
	real sigma; // F_fluc ~dispersion
	real r_cutoff;
	real predictor; //seconds to see future:)
	real v_slow;
	real v_kasses;
	real v_entrance;
//	real beta; //6a6ywka-coefficient
	//additional:
	real coef_vosstan; //NEW
//	real wall_cutoff;
//	real wall_accuracy;
	//shop parameters and statistics:
	vector<int> virus_exit;
	int* nPurchases;
	int* purchases;
	vector<int> purchases_stat;
	real mean_purchases;
	real stdev_purchases;
	bool PURCHASES_LIST_ON;
	int* time_in;
	vector<real> time_in_shop;
//	real xline; //line 
	//computes:
	real flux; //cumulative flux
	//initial infection percentage:
	real init_infection;
	int init_infected;
	int infected;
	int infect_acts;
	int infected_out;
	//positions and helth:
	real* x;
	real* y;
	real* xp;
	real* yp;
	int* z;	//zarajenie:
			// z > 0 means infection and infectiousness,
			// z = 0 means infection (incubation),
			// z < 0 means helth
	real virus_d100;
	real kappa; //charact.distance infection p ~ virus_A_iso * exp(-dist/kappa)
	real virus_A_iso;
	real virus_cutoff; //cutoff distance for virus
	//masks:
	real mask_factor_const; //NEW - how mask decreases a probability of infection transmission
	real in_mask; //NEW: share of infective visitors, who is in mask
	int in_mask_n; //NEW: number of infective visitors, who is in mask
	real* mask_factor; //NEW
	//velocities:
	real* vx;
	real* vy;
	//current accelerations:
//	real* ax;
//	real* ay;
	//future-step accelerations:
	real* axf;
	real* ayf;
	//type:
	char* type; //of pedestrian (defines direction)
	char* zone; //symbol of zone
	char* prev_zone;
	real change_direction; //from 0 to 1.0
	int stat_N;
	int stat_S;
	int stat_E;
	int stat_W;
	int stat_full_cart;
	// on-off computes/dumps while running:
	int on_thermo_output; //energies to file, if < 1 => off
	int on_dump_trajectories; //positions to xyz-file, if < 1 => off
	int on_compute_rdf; // pair distribution function
	int on_compute_time_histogram;
	int on_compute_Maxwell; // |velocity| distribution 
	int on_compute_density_map;
	int on_compute_flux_vs_time;
	real rdf_cutoff;
	//averaging parameters:
	int ave_last_steps;
	//visualization parameters:
	bool visible_virus;
	bool visible_velocity;
	bool visible_corners;
	bool visible_force;
	bool visible_acceleration;
	bool visible_patience;
	real xscale;//= 2.0 for VMD, 1 m -to- 2 A
	real vscale;
	real ascale;
	real pscale;
	int** density_map;
	int dm_nx;
	int dm_ny;
	real dm_scale;
	//map stat.:
	int n_walls;
	int n_sales;
	
	real load_map()
	{
		char fname[MAX_LENGTH_NAME];
		sprintf(fname, "model%d.map.txt", (int)MAP_SINDEX);
		FILE* f = fopen(fname, "r");
		if(f == NULL)
		{
			cout<<"Achtung: Map-file <"<<fname<<"> is not found!"<<endl;
			return 0.0;
		}
		//load map-file:
		int i, j, iline;
		char buff;
		int q = 0;
		n_walls = 0;
		n_sales = 0;
		int n_kasses = 0;
		int freemotion_zones = 0;
		int slow_zones = 0;
		int kass_zones = 0;
		int crossroad_area = 0;
		int entrance_zone = 0;
		
		fscanf(f, "%d %d %f\n", &nx, &ny, &map_scale);
		//fscanf(f, "%d %d %f %d", &nx, &ny, &map_scale, &iline);
		//xline = (real)iline * map_scale;
		map = (char**) new char*[nx];
		for(i=0; i<nx; i++)
			map[i] = (char*) new char[ny];
		
		for(j=ny-1; j>=0; j--)
			for(i=0; i<nx; i++)
			{
				fscanf(f, "%c", &buff);
				if((buff == '\n') || (buff == '\r') || (buff == '\0'))
				{
					i--;
					continue;
				}
				map[i][j] = buff;
				if((buff != '#') && (buff != '%') && (buff != 'X'))
					q++;
				switch(buff)
				{
				case '#':
					n_walls++;
					break;
				case '%':
					n_sales++;
					break;
				case ' ':
					freemotion_zones++;
					break;
				case '+':
					crossroad_area++;
					break;
				case 'X':
					n_kasses++;
					break;
				case '.':
					slow_zones++;
					break;
				case ':':
					kass_zones++;
					break;
				case 'E':
					entrance_zone++;
					break;
				}
			}
		fclose(f);
		area = (real)q * map_scale * map_scale;
		
		//save map stats:
		if(comm_rank == 0)
		{
			char filename[MAX_LENGTH_NAME];
			sprintf(filename, "STATS.%s", fname);
			f = fopen(filename, "w");
			fprintf(f, "nx = %d\n", nx);
			fprintf(f, "ny = %d\n", ny);
			fprintf(f, "scale = %f\n", (float)map_scale);
			//fprintf(f, "xline = %f\n", (float)xline);
			fprintf(f, "area (shop area), m2 = %f\n", (float)area);
			fprintf(f, "free motion zones (\' \'), m2 = %f\n", (float)freemotion_zones * map_scale * map_scale);
			fprintf(f, "walls area (\'#\'), m2 = %f\n", (float)n_walls * map_scale * map_scale);
			fprintf(f, "\"sales\" area (\'\%\'), m2 = %f\n", (float)n_sales * map_scale * map_scale);
			fprintf(f, "crossroad area (\'+\'), m2 = %f\n", (float)crossroad_area * map_scale * map_scale);
			fprintf(f, "slow motion zones (\'.\'), m2 = %f\n", (float)slow_zones * map_scale * map_scale);
			fprintf(f, "entrance zone area (\'E\'), m2 = %f\n", (float)entrance_zone * map_scale * map_scale);
			fprintf(f, "kasses zones (\':\'), m2 = %f\n", (float)kass_zones * map_scale * map_scale);
			fprintf(f, "number of kasses (\'X\') = %d\n", n_kasses);
			fclose(f);
			
			//copy to console:
			printf("area (shop area), m2 = %f\n", (float)area);
			printf("free motion zones (\' \'), m2 = %f\n", (float)freemotion_zones * map_scale * map_scale);
			printf("walls area (\'#\'), m2 = %f\n", (float)n_walls * map_scale * map_scale);
			printf("\"sales\" area (\'\%\'), m2 = %f\n", (float)n_sales * map_scale * map_scale);
			printf("crossroad area (\'+\'), m2 = %f\n", (float)crossroad_area * map_scale * map_scale);
			printf("slow motion zones (\'.\'), m2 = %f\n", (float)slow_zones * map_scale * map_scale);
			printf("entrance zone area (\'E\'), m2 = %f\n", (float)entrance_zone * map_scale * map_scale);
			printf("kasses zones (\':\'), m2 = %f\n", (float)kass_zones * map_scale * map_scale);
			printf("number of kasses (\'X\') = %d\n", n_kasses);
		}
		return area;
	}
	
	VSA(real input_rho = 0.1, real input_R = 0.2)
	{
		rho = input_rho;
		R = input_R;
		//=====================================================
		area = load_map();
		n = (int)round(area * rho); //total number of 'particles'	
		L = nx * map_scale; //length
		Lw = ny * map_scale; //width
//		hL = .5*L; //half-length of the street ?old
//		hW = .5*Lw; //half-width of the street ?old
	//	h = .5 * map_scale;
		dm_scale = map_scale/3.0;
		dm_nx = 3 * nx;
		dm_ny = 3 * ny;
		//simulation parameters:
		dt = 0.01; //0.1; //timestep
		//force field parameters:
//		mass = (real*) new real[n];
//		mdelta = 0.0; // mass = [1-mdelta, 1+mdelta], uniform or constant
		mod_v_des = 1.34; //absolute value of desired velocity
		std_v_des = 0.0; // =0.26; //stddev
		v_max = 1.3; // limit = v_max*mod_v_des
		v_des_x = (real*) new real[n]; //desired velocity
		v_des_y = (real*) new real[n]; //desired velocity
		chi = 0.14; //0.042; //chirality factor
		tau = 0.5; //relaxation time
		B = 0.3; //relaxation distance;
		A = 2.1; //
	//	dL = 0.02;//0.2;
	//	Uo = 10.0;
		D = 4.0;
		downtime_factor = 0.2; //0.35; //0.1;
		max_patience = 7.0; //10.0; //max downtime before changing the v_des
		patience = (real*) new real[n];
		//characteristing size of the particle:
		growing_R = false;
		sigma = 0.1;
		r_cutoff = 4.0;
	//	wall_cutoff = 0.5;
	//	wall_accuracy = 1.0e-8;
		coef_vosstan = 0.1;
		predictor = 1.5; //1.0 second
		v_slow = 0.3; //0.2;
		v_kasses = 0.03; //0.02;
		v_entrance = 0.05;
//		beta = 1.0; //6a6ywka-coefficient
		//viral parameters:
		init_infection = 0.01; //0.01; // = 1%
		init_infected = 0;
		infected = 0;
		infected_out = 0;
		infect_acts = 0;
		//positions and helth/virus:
		x = (real*) new real[n];
		y = (real*) new real[n];
		xp = (real*) new real[n];
		yp = (real*) new real[n];
		z = (int*) new int[n];
		virus_d100 = 2.0;
		kappa = virus_d100/log(100.0); //charact.distance infection p ~ exp(-dist/kappa)
		virus_A_iso = 0.01; //1.0/60.0; //1.0;
		virus_cutoff = 2.0 * virus_d100;
		//masks:
		mask_factor_const = 0.5; //NEW - how mask decreases a probability of infection transmission
		in_mask = 0.5; //NEW: share of infective visitors, who is in mask
		mask_factor = (real*) new real[n];
		//velocities:
		vx = (real*) new real[n];
		vy = (real*) new real[n];
		//current accelerations:
	//	ax = (real*) new real[n];
	//	ay = (real*) new real[n];
		//future-step accelerations:
		axf = (real*) new real[n];
		ayf = (real*) new real[n];
		//type:
		type = (char*) new char[n];
		//zone:
		zone = (char*) new char[n];
		prev_zone = (char*) new char[n];
		change_direction = 2.0/3.0;
		//shop-related variables:
		nPurchases = (int*) new int[n];
		purchases = (int*) new int[n];
		mean_purchases = 40.0; //15.0;
		stdev_purchases = 20.0;
		PURCHASES_LIST_ON = true;
		time_in = (int*) new int[n];
		//output parameters:
		on_thermo_output = 100; //energies to file, if < 1 => off
		on_dump_trajectories = 10; //positions to xyz-file, if < 1 => off
		on_compute_rdf = 0; // RDF(r)
		on_compute_density_map = 0; //accumulate every $n steps
		on_compute_time_histogram = 50;
		on_compute_Maxwell = 0; // nbins in |v| distribution if < 1 => 
		on_compute_flux_vs_time = 0;
		rdf_cutoff = 5.0;
		flux = 0.0;  //cumulative flux
		//dm:
		density_map = (int**) new int*[dm_nx];
		for(int i=0; i<dm_nx; i++)
			density_map[i] = (int*) new int[dm_ny];
		//visualization parameters:
		visible_virus = true;
		visible_velocity= false;
		visible_patience = !true;
		visible_corners = true;
		visible_force = false;
		visible_acceleration = false;
		xscale = 2.0; // 1 m -to- 2 A, for VMD
		vscale = 0.25;
		ascale = 0.25;
		pscale = 0.5;
	}
	
	~VSA()
	{
		//release memory:
		delete [] x;
		delete [] y;
		delete [] xp;
		delete [] yp;
		delete [] z;
		delete [] vx;
		delete [] vy;
	//	delete [] ax;
	//	delete [] ay;
		delete [] axf;
		delete [] ayf;
		delete [] type;
		delete [] zone;
		delete [] prev_zone;
	//	delete [] mass;
		delete [] v_des_x;
		delete [] v_des_y;
		delete [] map;
		delete [] nPurchases;
		delete [] purchases;
		delete [] density_map;
		delete [] patience;
		//delete [] flux_map;
		delete [] mask_factor;
	}
	
	void fillarray(real* array, int len, real value)
	{
		for(int i=0; i<len; i++)
			array[i] = value;
	}
	
	void fillarray_int(int* array, int len, int value)
	{
		for(int i=0; i<len; i++)
			array[i] = value;
	}
	
	void display_map()
	{
		int i, j;
		cout<<"map[nx][ny] = map["<<nx<<"]["<<ny<<"]"<<endl;
		cout<<"(n_walls, n_sales) = ("<<n_walls<<", "<<n_sales<<")"<<endl;
		for(j=ny-1; j>=0; j--)
		{
			for(i=0; i<nx; i++)
				printf("%c", map[i][j]);
			printf("\n");
		}
	}
	
	int print_types()
	{
#ifdef DEF_USING_MPI
		return -1;
#endif
		int n_n = 0;
		int n_s = 0;
		int n_e = 0;
		int n_w = 0;
		int n_V, n_H;
		n_V = n_H = 0;
		int n_unk = 0;
		for(int i=0; i<n; i++)
			switch(type[i])
			{
			case 'N':
				n_n++;
				break;
			case 'S':
				n_s++;
				break;
			case 'E':
				n_e++;
				break;
			case 'W':
				n_w++;
				break;
			case 'V':
				n_V++;
				break;
			case 'H':
				n_H++;
				break;
			default:
				n_unk++;
				cout<<"UNK> "<<i<<": -type = \'"<<type[i]<<"\'"<<", -zone = \'"<<zone[i]<<"\'"<<endl;
			}
		cout<<"type N: "<<n_n<<endl;
		cout<<"type S: "<<n_s<<endl;
		cout<<"type E: "<<n_e<<endl;
		cout<<"type W: "<<n_w<<endl;
		cout<<"------"<<endl;
		if(n_V == 0)
			n_V = n_n+n_s;
		if(n_H == 0)
			n_H = n_e+n_w;
		cout<<"type V: "<<n_V<<endl;
		cout<<"type H: "<<n_H<<endl;
		cout<<"unknown type: "<<n_unk<<endl;
		cout<<"n = "<<n<<endl;
		
		return n_unk;
	}
	
	bool print_list()
	{
#ifdef DEF_USING_MPI
		return false;
#endif
		for(int i=0; i<n; i++)
			cout<<"#:"<<i<<": -type = \'"<<type[i]<<"\', -zone = \'"<<zone[i]<<"\', -virus = \'"<<z[i]<<"\'"<<endl;
		
		return true;
	}
	
	char findout_zone(real xx, real yy)
	{
		int i, j;
		i = (int)(round(xx/map_scale));
		j = (int)(round(yy/map_scale));
		if((i<0) || (j<0) || (i>=nx) || (j>=ny))
			return '#'; //=wall
		//else:
		return map[i][j];
	}
	
	char update_type(int i)
	{
		//exclude crossroads:
		if((type[i]=='n') || (type[i]=='s') || (type[i]=='e') || (type[i]=='w')) 
			return type[i];
		//untypical case (some error?):
		if(v_des_x[i]*v_des_y[i] != 0.0)
			return 'B'; //has both components out of crossroads
		//typical directions:
		if(v_des_y[i] < 0.0)
			return 'S';
		if(v_des_y[i] > 0.0)
			return 'N';
		if(v_des_x[i] < 0.0)
			return 'W';
		if(v_des_x[i] > 0.0)
			return 'E';
		//stopped particle (some error):
		return 'o'; 
	}
	
	void init_positions(real noise = 0.0)
	{
		//initialize positions and infection:
		real free_area = 0.0;
		int i, j;
		real xx, yy;
		char zz;
		for(i=0; i<nx; i++)
			for(j=0; j<ny; j++)
				if((map[i][j] == ' ') || (map[i][j] == '.') || (map[i][j] == '+'))
					free_area += (map_scale*map_scale);
		
		real lattice = sqrt(2.0*free_area/((real)n*sqrt(3.0)));
		real lshift = .5 * lattice;
			
		i = 0;
		for(xx = 0.0; xx<=L; xx += (.5*sqrt(3.0)*lattice))
		{
			if(lshift != 0.0)
				lshift = 0.0;
			else
				lshift = .5 * lattice;
			for(yy = 0.0; yy<=Lw; yy+=lattice)
			{
				if(i >= n)
					return;
				x[i] = xx + noise*((real)rand()/((real)RAND_MAX)*2.0-1.0);
				y[i] = yy + noise*((real)rand()/((real)RAND_MAX)*2.0-1.0);
				//prev. coordinates:
				xp[i] = x[i];
				yp[i] = y[i];
				zz = findout_zone(x[i], y[i]);
				if((zz == ' ') || (zz == '+') || (zz == '.') || (zz == ':')) 
				{
					if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						type[i] = 'H';
					else
						type[i] = 'V';
					//all are healthy:
					z[i] = -1;
					//set zone:
					zone[i] = zz;
					i++;
				}
			}
		}

		//if filling is not complete:
		if(i >= n)
			return;

		j = i;
		while(j < n)
		{
			x[j] = ((real)rand()/((real)RAND_MAX)*L);
			y[j] = ((real)rand()/((real)RAND_MAX)*Lw);
			//prev. coordinates:
			xp[i] = x[i];
			yp[i] = y[i];
			if(findout_zone(x[j], y[j]) == 'E')
			{
				type[j] = 'N'; //Up
				v_des_x[j] = 0.0;
				v_des_y[j] = gauss_random(mod_v_des, std_v_des);
				//no virus
				z[j] = -1;
				zone[j] = 'E';
				j++;
			}
		}
	}
	
	real init_purchases()
	{
		int i;
		real ave = 0.0;
		
		for(i=0; i<n; i++)
		{
			nPurchases[i] = (int)round(gauss_random(mean_purchases, stdev_purchases));
			if(nPurchases[i] < 0)
				nPurchases[i] = 0;
			purchases[i] = 0;
			ave += (real)nPurchases[i];
		}
		
		return ave/((real)n);
	}
	
	void return_to_entrance(int j)
	{
		//store statistics:
		if(time_in >= 0)
		{
			time_in_shop.push_back((real)(step - time_in[j]) * dt);
			purchases_stat.push_back(purchases[j]);
			virus_exit.push_back(z[j]);
		}
		//reset timer and empty cart:
		time_in[j] = step;
		nPurchases[j] = (int)round(gauss_random(mean_purchases, stdev_purchases));
		if(nPurchases[j] < 0)
			nPurchases[j] = 0;
		
		purchases[j] = 0;
		
		if(z[j] >= 0)
			infected_out++;
		//healing:
		if(z[j] == 0)
			z[j] = -1;//obnulenie
		
		//this part of function requires OPTIMIZATION!!!
		bool searching = true;
		while(searching)
		{
			x[j] = ((real)rand()/((real)RAND_MAX)*L);
			y[j] = ((real)rand()/((real)RAND_MAX)*Lw);
			if(findout_zone(x[j], y[j]) == 'E')
			{
				type[j] = 'N'; //Up
				v_des_x[j] = 0.0;
				v_des_y[j] = gauss_random(mod_v_des, std_v_des);
				searching = false;
			}
		}
	}
	
	void init_velocities(real mean = -1.0, real stddev = -1.0)
	{	//initialze not only v[] but also v_des[]!
		if(mean >= 0)
			mod_v_des = mean;
		if(stddev >= 0)
			std_v_des = stddev;
		int i;
		//initialize velocities and masses:
		for(i=0; i<n; i++)
		{
			vx[i] = (((real)rand()/((real)RAND_MAX+1.0))*2.0-1.0)*mod_v_des;
			vy[i] = (((real)rand()/((real)RAND_MAX+1.0))*2.0-1.0)*mod_v_des;
//			mass[i] = 1.0 + mdelta*(((real)rand()/((real)RAND_MAX+1.0))*2.0-1.0);
			if(type[i]=='H')
				if(((real)2.0*(real)rand()) < (real)RAND_MAX)
				{
					type[i] = 'E'; //Right
					v_des_x[i] = gauss_random(mod_v_des, std_v_des);
					v_des_y[i] = 0.0;
				}
				else
				{
					type[i] = 'W'; //Left
					v_des_x[i] = -gauss_random(mod_v_des, std_v_des);
					v_des_y[i] = 0.0;
				}
			if(type[i]=='V')
				if(((real)2.0*(real)rand()) < (real)RAND_MAX)
				{
					type[i] = 'N'; //Up
					v_des_x[i] = 0.0;
					v_des_y[i] = gauss_random(mod_v_des, std_v_des);
				}
				else
				{
					type[i] = 'S'; //Down
					v_des_x[i] = 0.0;
					v_des_y[i] = -gauss_random(mod_v_des, std_v_des);
				}
		}
	}
	
//	void init_accel()
//	{
//		//initialize acceleration arrays:
//	//	fillarray(ax, n, 0.0);
//	//	fillarray(ay, n, 0.0);
//		fillarray(axf, n, 0.0);
//		fillarray(ayf, n, 0.0);
//	}
	
	int apply_virus(real probability = -1.0)//NEW: modification
	{
		if(probability <= 0.0)
			probability = init_infection;
		else
			init_infection = probability;
		
		int i;
		int j = 0;
		int masks = 0;
		
		//infective:
		init_infected = (int)round((real)n * init_infection);
		in_mask_n = (int)round((real)init_infected * in_mask);
		fillarray(mask_factor, n, 1.0); //1 means - no reducing of transmission = no mask
		
		while(j < init_infected)
		{
			i = rand() % n;
			if(z[i] <= 0)
			{
				z[i] = 1; //virus
				j++;
				if(masks <= in_mask_n)
				{
					mask_factor[i] = mask_factor_const;
					masks++;
				}
			}
			else
				z[i] = -1; //health
		}
		
		return init_infected;
	}
		
	real distance(int i, int j, real* u)
	{
		if(i == j)
		{
			u[0] = u[1] = 0.0;
			return 0.0;
		}
		real dx, dy;
		dx = x[i] - x[j];
		dy = y[i] - y[j];
		//no pbc:
	//	if(dx > hL)
	//		dx -= L;
	//	else if(dx < -hL)
	//		dx += L;
	//	if(dy > hL)
	//		dy -= L;
	//	else if(dx < -hL)
	//		dy += L;
		real dr = sqrt(dx*dx+dy*dy);
		u[0] = dx/dr; //return via pointer	
		u[1] = dy/dr; //return via pointer
		return dr;
	}
	
	real step_function(real arg) // Theta() - step-function
	{
		if(arg > 0.0)
			return 1.0;
		return 0.0;
	}
	
	bool dump_xyz(FILE* fdump, int step)
	{
		if(fdump == NULL)
			return false;
		int num = n;
		if(visible_virus)
			num += n;
		if(visible_velocity)
			num += n;
		if(visible_patience)
			num += n;
		if(visible_corners)
			num += 2 * n_walls + n_sales;
		//	num += 4 * (n_walls + n_sales);
		fprintf(fdump, "%d\n timestep %d\n", num, step);
		int i, j;
		real xx, yy;
		char cbuff;
		for(i=0; i<n; i++)
		{
			xx = xscale * x[i];
			yy = xscale * y[i];
			if(z[i] > 0)
				cbuff = 'x';
			else
				cbuff = type[i];
			fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, xx, yy, 0.0);
			if(visible_virus)
				fprintf(fdump, " X %.2f %.2f %.2f\n", xx, yy, .5*((real)z[i]+1.0));
			if(visible_velocity)
				fprintf(fdump, " V %.2f %.2f %.2f\n", xx+vscale*vx[i], yy+vscale*vy[i], 0.0);
			if(visible_patience)
				fprintf(fdump, " P %.2f %.2f %.2f\n", xx, yy, -pscale*(patience[i] + .1));
		}
		float mult = map_scale * xscale;
		float zshift = 0.0;
		if(visible_corners)
			for(i=0; i<nx; i++)
				for(j=0; j<ny; j++)
				{
					if((map[i][j] == '#') || (map[i][j] == '%'))
					{
						if(map[i][j] == '%')
						{
							cbuff = 'Y';
							zshift = -0.5;
							fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i)*mult, ((float)j)*mult, zshift);
						}
						else
						{
							cbuff = 'C';
							zshift = -0.5;
							fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i)*mult, ((float)j)*mult, zshift);
							fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i)*mult, ((float)j)*mult, 0.0);
						}
					/*	fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i+.25)*mult, ((float)j+.25)*mult, zshift);
						fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i+.75)*mult, ((float)j+.25)*mult, zshift);
						fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i+.25)*mult, ((float)j+.75)*mult, zshift);
						fprintf(fdump, " %c %.2f %.2f %.2f\n", cbuff, ((float)i+.75)*mult, ((float)j+.75)*mult, zshift);
				*/	}
				}
	}
	
	void reset_density_map()
	{
		int i, j;
		for(i=0; i<dm_nx; i++)
			for(j=0; j<dm_ny; j++)
				density_map[i][j] = 0;
	}
	
	void accumulate_density_map()
	{
		int ii, jj;
		for(int i=0; i<n; i++)
		{
			ii = (int)(round(x[i]/dm_scale));
			jj = (int)(round(y[i]/dm_scale));
			if(ii < 0)
			{
			//	cout<<"Wrong x coordinate for density_map[][], x < 0"<<endl;
				ii = 0;
			}
			if(ii >= dm_nx)
			{
			//	cout<<"Wrong x coordinate for density_map[][], x > max"<<endl;
				ii = dm_nx-1;
			}
			if(jj < 0)
			{
			//	cout<<"Wrong y coordinate for density_map[][], y < 0"<<endl;
				jj = 0;
			}
			if(jj >= dm_ny)
			{
			//	cout<<"Wrong y coordinate for density_map[][], y > max"<<endl;
				jj = dm_ny-1;
			}
			density_map[ii][jj]++;
		}
	}
	
	void force(int i, real* f)
	{
		real fx = 0.0;
		real fy = 0.0;
		char zz;
		
		//F_des:
		zz = findout_zone(x[i], y[i]);
		real mult = 1.0;
		switch(zz)
		{
		case '.':
			mult = v_slow;
			break;
		case ':':
			mult = v_kasses;
			v_des_x[i] = -mod_v_des;
			v_des_y[i] = 0.0;
			type[i] = 'W';
			break;
		case '<':
			v_des_x[i] = -mod_v_des;
			v_des_y[i] = 0.0;
			type[i] = 'W';
			break;
		case '>':
			v_des_x[i] = mod_v_des;
			v_des_y[i] = 0.0;
			type[i] = 'E';
			break;
		case '^':
			v_des_y[i] = mod_v_des;
			v_des_x[i] = 0.0;
			type[i] = 'N';
			break;
		case 'E': //entrance
			mult = v_entrance;
			v_des_y[i] = mod_v_des;
			v_des_x[i] = 0.0;
			type[i] = 'N';
			break;
		case 'v':
			v_des_y[i] = -mod_v_des;
			v_des_x[i] = 0.0;
			type[i] = 'S';
			break;
		case 'X':
			x[i] -= L;
			flux += 1.0;
			return_to_entrance(i);
			break;
		}
		
		//update force components F_des:	
		fx += (-vx[i] + mult * v_des_x[i]) / tau;
    	fy += (-vy[i] + mult * v_des_y[i]) / tau;
	
    	//F_par + F_chiral:
    	int j;
		real rx, ry, rxf, ryf;
		real d, df, eps;
		real cx, cy, vv;
		real Nx, Ny;
		for(j=0; j<n; j++)
			if(i != j)
			{
				rx = x[i] - x[j];
				ry = y[i] - y[j];
    		//	//PBC: x-axis
    		//	if(rx > hL)
			//		rx -= L;
			//	else if(rx < -hL)
			//		rx += L;
			//	if(ry > hL)
			//		ry -= L;
			//	else if(ry < -hL)
			//		ry += L;
				//distance	
				d = sqrt(rx*rx + ry*ry); 
    			if(d <= r_cutoff)
    			{
					eps = d - 2.0*R;
    				vv = sqrt(vx[i]*vx[i] + vy[i]*vy[i]);
					cx = vx[i] / vv;
    				cy = vy[i] / vv;
    				//F_par:
					fx += A*exp(-eps/B)*.5*(rx / d)*(1.0 - cx*rx/d);
					fy += A*exp(-eps/B)*.5*(ry / d)*(1.0 - cy*ry/d);
				}
				//F_chiral:
				rxf = rx + (vx[i]-vx[j])*dt;//(vx[j]-vx[i])*dt; 
				ryf = ry + (vy[i]-vy[j])*dt;//(vy[j]-vy[i])*dt;
    			df = sqrt(rxf*rxf + ryf*ryf); //distance in future timestep
    
			    if(df < d) //distance is reducing
			    	if((-rx*vx[i]-ry*vy[i]) > 0.0) // 'i' can see 'j'
				        if((vx[i]*vx[j]+vy[i]*vy[j]) < 0.0) // move in opposite directions
				        {
				            Nx = -ry / d; //(-(ry...))
							Ny = rx / d; //(-(-rx...))
							fx += chi * step_function(D - d) * Nx;
							fy += chi * step_function(D - d) * Ny;
						}
			}
			
		//F_fluc:
		fx += gauss_random(0.0, sigma); // *sqrt~dt?
		fy += gauss_random(0.0, sigma); // *sqrt~dt?
		
		f[0] = fx;
		f[1] = fy;
	}
	
	void stb_soft_collisions()// stb-stochastic behavior
	{
		int i;
		char zz;
		char zzp;
		
		for(i=0; i<n; i++)
		{
			//Velocity recovery factor coef_vosstan of a particle repelled from a solid surface,
			// reflect vx or vy if collision is predicted in next step:
			zz = findout_zone(x[i], y[i]);
			if((zz != '%') && (zz != '#'))
		    {
		    //check x:
				//for Verlet:
				  //zzp = findout_zone(x[i] + vx[i]*dt + .5*ax[i]*dt*dt, y[i]);
				//for Euler:
				zzp = findout_zone(x[i] + vx[i]*dt, y[i]);
				if((zzp == '%') || (zzp == '#'))
				{		
		    		vx[i] *= (real)(-1.0 * coef_vosstan); //soft collision
		    	//	ax[i] = 0.0;
		    	}
		    //check y:
				//for Verlet:
				  //zzp = findout_zone(x[i], y[i] + vy[i]*dt + .5*ay[i]*dt*dt);
				//for Euler:
				zzp = findout_zone(x[i], y[i] + vy[i]*dt);
				if((zzp == '%') || (zzp == '#'))
				{		
		    		vy[i] *= (real)(-1.0 * coef_vosstan); //soft collision
		    	//	ay[i] = 0.0;
		    	}
		    //check CORNERS:
				zzp = findout_zone(x[i] + vx[i]*dt, y[i] + vy[i]*dt);
				if((zzp == '%') || (zzp == '#'))
				{
					vx[i] *= (real)(-1.0 * coef_vosstan); //soft collision		
		    		vy[i] *= (real)(-1.0 * coef_vosstan); //soft collision
		    	}
			}
		}
	}
	
	void stb_walls_and_obstacles()// stb-stochastic behavior
	{
		int i;
		char zz;
		
		for(i=0; i<n; i++)
		{
		    //go around walls and obstacles:
			zz = findout_zone(x[i]+predictor*v_des_x[i], y[i]+predictor*v_des_y[i]);
		//    zz = findout_zone(x[i]+predictor*vx[i], y[i]+predictor*vy[i]);
			if((zz == '%') || (zz == '#'))
   		 	{
   		 		//calculate purchases:
   		 		if(zz == '%')
    				purchases[i]++;
    			
    			switch(type[i])
    			{
    			case 'E':
    				if((real)rand() < ((real)0.8*(real)RAND_MAX))
					{	//to left 40% or to right 40%:
						if((real)2.0*rand() < (real)RAND_MAX)
						{
							v_des_y[i] = -mod_v_des;//-v_des_x[i];
							type[i] = 'S';
						}
						else
						{
							v_des_y[i] = mod_v_des;//v_des_x[i];
							type[i] = 'N';
						}
						v_des_x[i] = 0.0;
					}
					else //revers 20%
					{
						v_des_x[i] = -mod_v_des;
						type[i] = 'W';
					}
					break;
	    		case 'W':
					if((real)rand() < ((real)0.8*(real)RAND_MAX))
					{	//to left 40% or to right 40%:
						if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						{
							v_des_y[i] = -mod_v_des;//-v_des_x[i];
							type[i] = 'S';
						}
						else
						{
							v_des_y[i] = mod_v_des;//v_des_x[i];
							type[i] = 'N';
						}
						v_des_x[i] = 0.0;
					}
					else //revers 20%
					{
						v_des_x[i] = mod_v_des;
						type[i] = 'E';
					}
					break;
				case 'N':
					if((real)rand() < ((real)0.8*(real)RAND_MAX))
					{	//to left 40% or to right 40%:
						if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						{
							v_des_x[i] = mod_v_des;//-v_des_x[i];
							type[i] = 'E';
						}
						else
						{
							v_des_x[i] = -mod_v_des;//v_des_x[i];
							type[i] = 'W';
						}
						v_des_y[i] = 0.0;
					}
					else //revers 20%
					{
						v_des_y[i] = -mod_v_des;
						type[i] = 'S';
					}
					break;
				case 'S':
					if((real)rand() < ((real)0.8*(real)RAND_MAX))
					{	//to left 40% or to right 40%:
						if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						{
							v_des_x[i] = mod_v_des;//-v_des_x[i];
							type[i] = 'E';
						}
						else
						{
							v_des_x[i] = -mod_v_des;//v_des_x[i];
							type[i] = 'W';
						}
						v_des_y[i] = 0.0;
					}
					else //revers 20%
					{
						v_des_y[i] = mod_v_des;
						type[i] = 'N';
					}
					break;
				default:
					//reflect one component:
					if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						v_des_x[i] *= -1.0;
					else
						v_des_y[i] *= -1.0;
					update_type(i);
				}
			}
		}
	}
	
	void stb_crossroads() //stb - stochastic_behavior in crossroads
	{
		int i;
		
		for(i=0; i<n; i++)
		{
			//crossroads:
			switch(type[i])
			{
				case 'N':
				case 'S':
					if((zone[i] == '+') && (prev_zone[i] != '+'))
						if((real)rand() < ((real)change_direction*(real)RAND_MAX))
						{
							if(((real)2.0*(real)rand()) < (real)RAND_MAX)
								v_des_x[i] = -mod_v_des/sqrt(2.0); //-v_des_y[i];
							else
								v_des_x[i] = mod_v_des/sqrt(2.0); //v_des_y[i];
							v_des_y[i] /= sqrt(2.0);
							if(type[i] == 'N')
								type[i] = 'n';
							else
								type[i] = 's'; 
						}
					break;
				case 'n':
				case 's':
					if((zone[i] != prev_zone[i]) && (prev_zone[i] == '+'))
						if(v_des_x[i] != 0.0)
						{
							v_des_y[i] = 0.0;
							v_des_x[i] = sign(v_des_x[i]) * mod_v_des;
							if(v_des_x[i] < 0.0)
							{
								type[i] = 'W';
								debug_cross_w++;//////
							}
							else
							{
								type[i] = 'E';
								debug_cross_e++;//////
							}
						}
					break;
				case 'E':
				case 'W':
					if((zone[i]=='+') && (prev_zone[i] != '+'))
						if((real)rand() < ((real)change_direction*(real)RAND_MAX))
						{
							if(((real)2.0*(real)rand()) < (real)RAND_MAX)
								v_des_y[i] = -mod_v_des/sqrt(2.0); //-v_des_x[i];
							else
								v_des_y[i] = mod_v_des/sqrt(2.0); //v_des_x[i];
							v_des_x[i] /= sqrt(2.0);
							if(type[i] == 'E')
								type[i] = 'e';
							else
								type[i] = 'w'; 
						}
					break;
				case 'e':
				case 'w':
					if((zone[i] != prev_zone[i]) && (prev_zone[i] == '+'))
						if(v_des_y[i] != 0.0)
						{
							v_des_x[i] = 0.0;
							v_des_y[i] = sign(v_des_y[i]) * mod_v_des;
							if(v_des_y[i] < 0.0)
							{
								type[i] = 'S';
								debug_cross_s++;
							}
							else
							{
								type[i] = 'N';
								debug_cross_n++;
							}
						}
					break;
			}
			if(PURCHASES_LIST_ON)
				if(purchases[i] >= nPurchases[i])
				{
					//all purchases are in a cart:
					if((type[i] == 'E'))
						if((zone[i] != '+') && (prev_zone[i] == '+'))
						{
							v_des_x[i] = -mod_v_des;
							v_des_y[i] = 0.0;
							type[i] = 'W';
							debug_to_kasses++;
						}
				}
			//update prev_zone:
			prev_zone[i] = zone[i];
		}
	}
	
	void stb_downtime_vs_patience()//stochastic: downtime vs patience:
	{
		int i;
		real vx_fact;
		real vy_fact;
		real sp;
		
		for(i=0; i<n; i++)
			if((zone[i] != ':') && (zone[i] != 'E') && (zone[i] != '^'))// && (zone[i] != '<') && (zone[i] != '>')...)
			{
				vx_fact = (x[i] - xp[i]) / dt;
				vy_fact = (y[i] - yp[i]) / dt;
				sp = vx_fact * v_des_x[i] + vy_fact * v_des_y[i];
				if(sp < (downtime_factor * mod_v_des*mod_v_des))
					patience[i] += dt;
				else
					patience[i] = 0.0;
				//if patience has overflowed:
				if((real)patience[i] > (real)max_patience)
				{
					//reset patience:
					patience[i] = 0.0;
					//change (or not) v_des direction:
					if(((real)2.0*(real)rand()) < (real)RAND_MAX)
					{
						if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						{
							type[i] = 'N';
							v_des_x[i] = 0.0;
							v_des_y[i] = mod_v_des;
							debug_patience_n++;////////////////
						}
						else
						{
							type[i] = 'S';
							v_des_x[i] = 0.0;
							v_des_y[i] = -mod_v_des;
							debug_patience_s++;////////////////
						}
					}
					else
					{
						if(((real)2.0*(real)rand()) < (real)RAND_MAX)
						{
							type[i] = 'E';
							v_des_x[i] = mod_v_des;
							v_des_y[i] = 0.0;
							debug_patience_e++;/////////////////
						}
						else
						{
							type[i] = 'W';
							v_des_x[i] = -mod_v_des;
							v_des_y[i] = 0.0;
							debug_patience_w++;/////////////////
						}
					} 
				}
			}
	}
	
	void check_speed_limit() //check speed limit
	{
		int i;
		real vv;
		
		for(i=0; i<n; i++)
		{
			vv = vx[i]*vx[i] + vy[i]*vy[i];
			if(vv > (v_max*v_max * mod_v_des*mod_v_des))
			{
				vv = v_max * mod_v_des / sqrt(vv);
				vx[i] *= vv;
				vy[i] *= vv;
			}
		}
	}
	
	void Euler()//simple Euler method
	{
		int i;
		
		//evaluate forces:
		real F[2];
		for(i=0; i<n; i++)
		{
			force(i, F);
		//	axf[i] = F[0];// / mass[i];
		//	ayf[i] = F[1];// / mass[i];
			//update velocities:
			vx[i] += F[0] * dt;
			vy[i] += F[1] * dt;
		}
		
		check_speed_limit();
		
		stb_soft_collisions();//after check_speed_limit()
		
		//update atomic positions:
		for(i=0; i<n; i++)
		{
			//past = present:
			xp[i] = x[i];
			yp[i] = y[i];
			//present = future:
			x[i] += vx[i] * dt;
			y[i] += vy[i] * dt;
		}	
	}
	
/*	void Verlet()
	{
		int i;
		//check speed limit:
		real vv;
		for(i=0; i<n; i++)
		{
			vv = vx[i]*vx[i] + vy[i]*vy[i];
			if(vv > (v_max*v_max * mod_v_des*mod_v_des))
			{
				vv = v_max * mod_v_des / sqrt(vv);
				vx[i] *= vv;
				vy[i] *= vv;
				ax[i] = 0.0;
				ay[i] = 0.0;
			}
		}
		//update atomic positions:
		for(i=0; i<n; i++)
		{
			x[i] += vx[i]*dt + .5*ax[i]*dt*dt;
			y[i] += vy[i]*dt + .5*ay[i]*dt*dt;
			//wrap coordinates
			// pbc if needed.
		}
		//evaluate new accelerations:
		fillarray(axf, n, 0.0);
		fillarray(ayf, n, 0.0);
		real F[2];
		for(i=0; i<n; i++)
		{
			force(i, F);
			axf[i] += F[0] / mass[i];
			ayf[i] += F[1] / mass[i];
		}
		//update velocities:
		for(i=0; i<n; i++)
		{
			vx[i] += .5*(axf[i] + ax[i])*dt;
			vy[i] += .5*(ayf[i] + ay[i])*dt;
		}
		//walking through time:
		for(i=0; i<n; i++)
		{
			ax[i] = axf[i];
			ay[i] = ayf[i];
		}
	}*/
	
	void run(int nSteps, char* run_name)
	{
		if(nSteps < 1)
			return;
			
		cout<<"running: run_name = \'"<<run_name<<"\' for "<<nSteps<<" steps\n"<<endl;
		step = 0;
		int i, j;
		char fparam_name[MAX_LENGTH_NAME];
		char file_name[MAX_LENGTH_NAME];
		char logfile_name[MAX_LENGTH_NAME];
		char stbfile_name[MAX_LENGTH_NAME];
		char rdffile_name[MAX_LENGTH_NAME];
		char statfile_name[MAX_LENGTH_NAME];
		char fluxfile_name[MAX_LENGTH_NAME];
		sprintf(fparam_name, "result_and_prm.%s.%d.txt", run_name, comm_rank);
		sprintf(file_name, "XYZ-traj.%s.%d.xyz", run_name, comm_rank);
		sprintf(logfile_name, "log.%s.%d.txt", run_name, comm_rank);
		sprintf(stbfile_name, "stoh.ctrl.%s.%d.txt", run_name, comm_rank);
		sprintf(rdffile_name, "RDF.%s.%d.txt", run_name, comm_rank);
		sprintf(statfile_name, "statistics.%s.%d.txt", run_name, comm_rank);
		sprintf(fluxfile_name, "flux.%s.%d.txt", run_name, comm_rank);
		
		FILE* fparam = fopen(fparam_name, "w");
		FILE* file;
		if(on_dump_trajectories > 0)
			file = fopen(file_name, "w");
		FILE* rdffile;
		FILE* logfile = fopen(logfile_name, "w");
		FILE* statfile;
		FILE* stbfile = fopen(stbfile_name, "w");
		FILE* fluxfile;
		if(on_compute_flux_vs_time > 0)
			fluxfile = fopen(fluxfile_name, "w");
		//for 'computes':
		real* maxwell;
		real Vsup;
		//averaging parameter:
		ave_last_steps = (int)round(.5 * (real)nSteps);
		real infected_ave = 0.0;
		int current_ave_steps = 0;
		
		//write simulation parameters:
		fprintf(fparam, "n = %d\n", n);
		fprintf(fparam, "L = %f\n", (float)L);
		fprintf(fparam, "Lw = %f\n", (float)Lw);
		fprintf(fparam, "rho = %f\n", (float)rho);
		fprintf(fparam, "dt = %f\n", (float)dt);
		//fprintf(fparam, "mdelta = %f\n", (float)mdelta);
		fprintf(fparam, "mod_v_des = %f\n", (float)mod_v_des);
		fprintf(fparam, "std_v_des = %f\n", (float)std_v_des);
		fprintf(fparam, "v_max = %f\n", (float)v_max);//1.3;
		fprintf(fparam, "chi = %f\n", (float)chi);
		fprintf(fparam, "tau = %f\n", (float)tau);
		fprintf(fparam, "B = %f\n", (float)B);
		fprintf(fparam, "A = %f\n", (float)A);
		//fprintf(fparam, "dL = %f\n", (float)dL);
		//fprintf(fparam, "Uo = %f\n", (float)Uo);
		fprintf(fparam, "D = %f\n", (float)D);
	//	if(growing_R)
	//	{
	//		fprintf(fparam, "R = 0.2 - %f\n", (float)R);
	//		//fprintf(fparam, "# R is growing linearly\n)
	//	}
	//	else
		fprintf(fparam, "R = %f\n", (float)R);
		fprintf(fparam, "sigma = %f\n", (float)sigma);
	//	fprintf(fparam, "beta = %f\n", (float)beta);
		fprintf(fparam, "r_cutoff = %f\n", (float)r_cutoff);//4.0;
		//fprintf(fparam, "wall_cutoff = %f\n", (float)wall_cutoff);//0.5;
		//fprintf(fparam, "wall_accuracy = %g\n", (float)wall_accuracy);//1.0e-8;
		fprintf(fparam, "coef_vosstan = %g\n", (float)coef_vosstan);
		fprintf(fparam, "predictor = %f\n", (float)predictor);//1.5;
		fprintf(fparam, "change_direction = %f\n", (float)change_direction);
		fprintf(fparam, "init_infection = %f\n", (float)init_infection);
		fprintf(fparam, "virus_d100 = %f\n", (float)virus_d100);
		fprintf(fparam, "kappa = %f\n", (float)kappa);
		fprintf(fparam, "virus_A_iso = %f\n", (float)virus_A_iso);
		fprintf(fparam, "virus_cutoff = %f\n", (float)virus_cutoff);
		fprintf(fparam, "in_mask = %f\n", (float)in_mask);
		fprintf(fparam, "in_mask_n = %d\n", (int)in_mask_n);
		fprintf(fparam, "mask_factor_const = %f\n", (float)mask_factor_const);
		fprintf(fparam, "v_slow = %f\n", (float)v_slow);//0.3;
		fprintf(fparam, "v_kasses = %f\n", (float)v_kasses);//0.03;
		fprintf(fparam, "v_entrance = %f\n", (float)v_entrance);//0.05;
		fprintf(fparam, "mean_purchases = %f\n", (float)mean_purchases);//40.0;
		fprintf(fparam, "stdev_purchases = %f\n", (float)stdev_purchases);//20.0;
		fclose(fparam);
		
		//restart fluxes:
		flux = 0.0;
		infected_out = 0.0;
		
		//reset zones:
		for(i=0; i<n; i++)
		{
			zone[i] = findout_zone(x[i], y[i]);
			prev_zone[i] = zone[i];
		}
		
		//patience:
		fillarray(patience, n, 0.0);
		
		//initialize purchases and reset timer:
		fillarray_int(time_in, n, -1);
		init_purchases();
		
		//RDF:
		real* rdf;
		int bin;
		if(on_compute_rdf > 1) // equal to nbins
		{
			rdf = (real*)new real[on_compute_rdf];
			fillarray(rdf, on_compute_rdf, 0.0);
		}
		
		//density map:
		if(on_compute_density_map > 0)
			reset_density_map();
		
		//velocity distribution:
		if(on_compute_Maxwell > 0)
		{
			maxwell = (real*) new real[on_compute_Maxwell];
			fillarray(maxwell, on_compute_Maxwell, 0.0);
			Vsup = 0.0;
			for(i=0; i<n; i++)
			{
				real sqrV = vx[i]*vx[i] + vy[i]*vy[i];
				if(Vsup < sqrV)
					Vsup = sqrV;
			}
			Vsup = 1.10 * sqrt(Vsup); //upper limit for calculation of velocity distribution +10%
			Vsup /= (real)on_compute_Maxwell;
		}
		
		//if R grows
		real R_stored, incR;
		if(growing_R)
		{
			R_stored = R;
			R = 0.2;
			incR = (R_stored-R) / (real)nSteps;
		}
		
#ifndef DEF_USING_MPI
		printf("step: | flux(#) | flux(#/s) | infect.acts | infected | infected_out | ave infected | #N #S #E #W | #ready\n");
#endif
/* ******** time cycle *********** */
/* ******************************* */
		for(step=1; step <= nSteps; step++)
		{
			//stochastic behavior:
			stb_crossroads();
			stb_walls_and_obstacles();
			//stb_soft_collisions(); -- replaced to Euler()
			stb_downtime_vs_patience();
			//stochastic_behavior(); -- old function
			
			//update r, v, a:
			Euler();
			//Verlet();
			
			//update zones and calculate statistics:
			stat_N = 0;
			stat_S = 0;
			stat_E = 0;
			stat_W = 0;
			stat_full_cart = 0;
			for(i=0; i<n; i++)
			{
				//update zone:
				zone[i] = findout_zone(x[i], y[i]);
				if((zone[i] == '#') || (zone[i] == '%'))
					if((step % on_thermo_output) == 0)
						cout<<"Warning: pedestrian ["<<i<<"] is in the wall."<<endl;
				//directions statistics:
				if(type[i] == 'N')
					stat_N++;
				if(type[i] == 'S')
					stat_S++;
				if(type[i] == 'E')
					stat_E++;
				if(type[i] == 'W')
					stat_W++;
				//check cart:
				if(purchases[i] >= nPurchases[i])
					stat_full_cart++;
			}
			
			if(growing_R)
				R += incR;
			//virus, infection:
			for(i=0; i<n; i++)
				if(z[i] > 0)
					for(j=0; j<n; j++)
						if((i != j) && (z[j] < 0))
						{
							real buff[2];
							real dd = distance(i, j, buff);
							if(dd < virus_cutoff)
								if(((real)rand()) <= ((real)RAND_MAX * dt * virus_A_iso * mask_factor[i] * exp(-dd/kappa)))
								{	//important to multiply by dt...
									z[j] = 0; //infect
									infect_acts++;
								}
						}
			infected = 0;
			for(i=0; i<n; i++)
				if(z[i] >= 0)
					infected++;
			if(step >= (nSteps-ave_last_steps))
			{
				infected_ave += (real)infected;
				current_ave_steps++;
			}

			//outputs:
			if(on_thermo_output > 0)
				if(step % on_thermo_output == 0)
				{
					real time = (real)step * dt;
					fprintf(logfile, "%d\t%f\t%f\t%d\t%d\tN %d\tS %d\tE %d\tW %d\t%d\n", step, (float)flux, (float)flux/time, (int)(infect_acts), (int)infected, stat_N, stat_S, stat_E, stat_W, stat_full_cart);
					fprintf(stbfile, "%d: || %d %d %d %d | %d %d %d %d | %d \n", step, debug_patience_n, debug_patience_s, debug_patience_e, debug_patience_w, debug_cross_n, debug_cross_s, debug_cross_e, debug_cross_w, debug_to_kasses);
#ifndef DEF_USING_MPI
					if(current_ave_steps > 0)
						printf("%d: | %.0f | %.3f | %d | %d | %d | %.3f || N%d S%d E%d W%d | %d \n", step, (float)flux, (float)flux/time, (int)(infect_acts), (int)(infected), (int)(infected_out), (float)infected_ave/(float)current_ave_steps, stat_N, stat_S, stat_E, stat_W, stat_full_cart);
					else
						printf("%d: | %.0f | %.3f | %d | %d | %d | <off> || N%d S%d E%d W%d | %d \n", step, (float)flux, (float)flux/time, (int)(infect_acts), (int)(infected), (int)(infected_out), stat_N, stat_S, stat_E, stat_W, stat_full_cart);
#endif
				}
			if(on_dump_trajectories > 0)
				if(step % on_dump_trajectories == 0)
					dump_xyz(file, step);
			//statistics:
			if(on_compute_rdf > 1)
				if(step >= (nSteps-ave_last_steps))
					for(i=1; i<n; i++)
						for(j=0; j<i; j++)
						{
							real buf[2];
							real dd = distance(i, j, buf);
							if(dd <= rdf_cutoff)
							{
								bin = (int) floor(dd/(real)rdf_cutoff*(real)on_compute_rdf);
								if(bin < on_compute_rdf)
									rdf[bin] += 1.0;
							}
						}
			if(on_compute_density_map > 0)
				if(step % on_compute_density_map == 0)
					accumulate_density_map();
			//velocity distribution:			
			if(on_compute_Maxwell > 0)
				if(step >= (nSteps-ave_last_steps))
				{
					for(i=0; i<n; i++)
					{
						int j = (int)floor(sqrt(vx[i]*vx[i] + vy[i]*vy[i])/Vsup+.5);
						if(j < on_compute_Maxwell)
							maxwell[j]++;
					}
				}
			//flux:
			if(on_compute_flux_vs_time > 0)
				if(step % on_compute_flux_vs_time == 0)
				{
					//without header...
					fprintf(fluxfile, "%f\t%f\n", (float)step*dt, (float)flux);
				}
		}/* ******** end of time cycle *********** */
		/* *************************************** */
		
		fclose(logfile);
		fclose(stbfile);
		if(on_dump_trajectories > 0)
			fclose(file);
		if(on_compute_flux_vs_time > 0)
			fclose(fluxfile);
			
		if(growing_R)
			R = R_stored;
		
		//statistics to file:
		real mean_time_in_shop = 0.0;
		real mean_purchases = 0.0;
		
		statfile = fopen(statfile_name, "w");
		fprintf(statfile, "time_in_shop\tvirus_exit\tpurchases_stat\n");
		for(i=0; i<time_in_shop.size(); i++)
		{
			mean_time_in_shop += (real)time_in_shop[i];
			mean_purchases += (real)purchases_stat[i];
			
			//save to file:
			fprintf(statfile, "%f\t%d\t%d\n", (float)time_in_shop[i], virus_exit[i], purchases_stat[i]);
		}
		mean_time_in_shop /= (real)time_in_shop.size();
		mean_purchases /= (real)purchases_stat.size();
		
		fclose(statfile);
		
		//analyze number of pedestrians who is forever in the shop:
		int forever_in_shop = 0;
		for(i=0; i<n; i++)
		{
			if(time_in[i] < 0)
				forever_in_shop++;
		}
		
		//append to file:
		infected_ave /= (real)current_ave_steps;
		real acts_per_minute = (real)infect_acts * 60.0 / ((real)nSteps * dt);
		real percentage_of_infected = infected_ave / (real)n;
		real increase_factor = infected_ave / (real)init_infected;
		fparam = fopen(fparam_name, "a");
		//fprintf(fparam, "_______\nRESULTS:\n");
		fprintf(fparam, "acts_per_minute = %f\n", (float)acts_per_minute);
		fprintf(fparam, "acts_per_minute/init_infected = %f\n", (float)acts_per_minute/(float)init_infected);
		fprintf(fparam, "percentage_of_infected [%] = %f\n", (float)100.0*percentage_of_infected);
		fprintf(fparam, "init_infected = %d\n", init_infected);
		fprintf(fparam, "init_infected/n*100% = %f\n", (float)init_infected*100.0/(float)n);
		fprintf(fparam, "infected_ave = %f\n", (float)infected_ave);
		fprintf(fparam, "increase_factor (infected_ave/init_infected) = %f\n", (float)increase_factor);
		fprintf(fparam, "flux [pedestrians / sec] = %f\n", (float)flux / ((real)nSteps*dt));
		fprintf(fparam, "infected_out = %d\n", (int)infected_out);
		fprintf(fparam, "probability of getting infected [%] = %f\n", (float)infected_out*100.0/(float)flux);
		fprintf(fparam, "mean time in shop [minutes] = %f\n", (float)mean_time_in_shop/60.0);
		fprintf(fparam, "number of who forever in shop = %d\n", forever_in_shop);
		fprintf(fparam, "mean purchases = %f\n", (float)mean_purchases);
		fclose(fparam);
		
		//copy to console:
#ifndef DEF_USING_MPI
		cout<<"RESULTS:"<<endl;
		cout<<"n = "<<n<<endl;
		cout<<"init_infected = "<<init_infected<<endl;
		cout<<"acts_per_minute = "<<acts_per_minute<<endl;
		cout<<"acts_per_minute/init_infected = "<<acts_per_minute/init_infected<<endl;
		cout<<"infected = "<<infected<<endl;
		cout<<"infected_ave = "<<infected_ave<<endl;
		cout<<"percentage_of_infected [%] = "<<100.0 * percentage_of_infected<<endl;
		if(init_infected > 0)
			cout<<"increase_factor (infected_ave / init_infected) = "<<increase_factor<<endl;
		else
			cout<<"increase_factor (0 / 0) = 1"<<endl;
		cout<<"flux_out [pedestrians / sec] = "<<(float)flux / ((real)nSteps*dt)<<endl;
		printf("infected_out = %d\n", (int)infected_out);
		printf("probability of getting infected [%] = %f\n", (float)infected_out*100.0/(float)flux);
		printf("mean time in shop [minutes] = %f\n", (float)mean_time_in_shop/60.0);
		printf("number of who forever in shop = %d\n", forever_in_shop);
		printf("mean purchases = %f\n", (float)mean_purchases);
#endif
		//compute results to files:
		if(on_compute_rdf > 1)
		{
			rdffile = fopen(rdffile_name, "w");
			fprintf(rdffile, "r\tdN(r)/r\tRDF(r)\n");
			real normalize = 0.0;
			for(bin=0; bin<on_compute_rdf; bin++)
				normalize += rdf[bin];
			for(bin=0; bin<on_compute_rdf; bin++)
			{
				real r_pos = ((float)bin+.5)/(float)on_compute_rdf*rdf_cutoff;
				fprintf(rdffile, "%f\t%f\t%f\n", (float)r_pos, (float)rdf[bin]/((float)current_ave_steps*r_pos), (float)rdf[bin]/normalize);
			}
			fclose(rdffile);
		}
		
			
		char analysis_file_name[MAX_LENGTH_NAME];
		FILE* fa;
		
		if(on_compute_time_histogram > 1)
		{
			sprintf(analysis_file_name, "time_histogram.%s.%d.txt", run_name, comm_rank);
			fa = fopen(analysis_file_name, "w");
			real time_max = 0.0;
			for(i=0; i<time_in_shop.size(); i++)
				if(time_max < time_in_shop[i])
					time_max = time_in_shop[i];
			real dTime = time_max / (real)(on_compute_time_histogram+1);
			int* histogram = (int*) new int[on_compute_time_histogram];
			fillarray_int(histogram, on_compute_time_histogram, 0);
			for(i=0; i<time_in_shop.size(); i++)
			{
				int bin = (int)round(time_in_shop[i]/dTime);
				if(bin < on_compute_time_histogram)
					histogram[bin]++;
			}
			fprintf(fa, "time in shop, minutes\tfrequency\n");
			for(i=0; i<on_compute_time_histogram; i++)
				fprintf(fa, "%f\t%d\n", (float)dTime*(float)i/60.0, histogram[i]);
			//the last point:
			fprintf(fa, ">%f\t%d\n", (float)nSteps*dt/60.0, forever_in_shop);
			fclose(fa);
			delete [] histogram;
		}
		
		if(on_compute_density_map > 0)
		{
			sprintf(analysis_file_name, "density_map.%s.%d.txt", run_name, comm_rank);
			fa = fopen(analysis_file_name, "w");
			fprintf(fa, "%d %d %f\n", dm_nx, dm_ny, (float)dm_scale);
			for(i=0; i<dm_nx; i++)
			{
				for(j=0; j<dm_ny; j++)
					fprintf(fa, "%d ", density_map[i][j]);
				fprintf(fa, "\n");
			}
			fclose(fa);
		}
		
		if(on_compute_Maxwell > 0)
		{
			sprintf(analysis_file_name, "velocity_distribution.%s.%d.txt", run_name, comm_rank);
			fa = fopen(analysis_file_name, "w");
			fprintf(fa, "|v|\tf(.)\n", (float)1.0);
			for(i=0; i<on_compute_Maxwell; i++)
				fprintf(fa, "%f\t%f\n", (float)i*Vsup, (float)maxwell[i]/(n*nSteps));
			fclose(fa);
		}
		
		//memory release:
		if(on_compute_rdf > 1)
			delete [] rdf;
		if(on_compute_Maxwell > 0)
			delete [] maxwell;
	}
};

int main(int argc, char** argv)
{
#ifdef DEF_USING_MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
	const real rho = 0.1; // or 0.06 + (real)comm_rank * (0.2 - 0.06) / ((real)comm_size-1.0);
	const real R = 0.4 + (real)comm_rank * (1.35 - 0.4) / ((real)comm_size-1.0);	
#else
	const real rho = 0.1; 
	const real R = .8; 
#endif
	if(comm_rank == 0)
		cout<<"main(): "<<comm_size<<" process(es) is(are) running\n"<<endl;
	
	VSA md(rho, R);
	if(comm_rank == 0)
		cout<<">> VSA md(rho, R);"<<endl;
	//initialize:
	unsigned int seed = (unsigned int)time(NULL);
	srand(seed);
	cout<<"srand("<<seed<<");"<<endl;
	if(comm_rank == 0)
		cout<<">> md.init_positions();"<<endl;
	md.init_positions();
	if(comm_rank == 0)
		cout<<">> md.init_velocities();"<<endl;
	md.init_velocities();
	//md.print_list();
	if(comm_rank == 0)
		cout<<">> md.init_accel();"<<endl;
//	md.init_accel();
	//PURCHASES_LIST ON or OFF:
//	md.PURCHASES_LIST_ON = !true;
	
#ifdef INITCONTROL
	if(comm_rank == 0)
		cout<<">> init.control"<<endl;
	//check initial positions:
	char iPos[MAX_LENGTH_NAME];
	sprintf(iPos, "init_positions.%d.xyz", (int)comm_rank);
	FILE* dump = fopen(iPos,"w");
	if(comm_rank == 0)
		cout<<">> md.dump_xyz(dump, 0);"<<endl;
	md.dump_xyz(dump, 0);
	fclose(dump);

	//check types:
	md.print_types();
	//md.print_list();

	//check map:
	if(comm_rank == 0)
	{
		md.display_map();
		cout<<"R = "<<md.R<<endl;
		cout<<"rho = "<<md.rho<<endl;
	}
#endif

	const real VirusInitRatio = g_Io; //or g_Io[comm_rank];
	char SIMNAME[MAX_LENGTH_NAME];
	char plOnOff[4];
	if(md.PURCHASES_LIST_ON)
		sprintf(plOnOff, "on");
	else
		sprintf(plOnOff, "off");
	sprintf(SIMNAME, "VSA%d-%.1fp-rho%.3f-R%.2f-mask=%.2f_%.2f-PL=%s", (int)MAP_SINDEX, (float)VirusInitRatio*100.0, (float)md.rho, (float)md.R, (float)md.in_mask, (float)md.mask_factor_const, plOnOff);

	if(comm_rank == 0)
		cout<<">> md.run((int)(round(120.0/md.dt)), (char*)\"EQ\");"<<endl;
	md.dt = 0.05;
	md.on_dump_trajectories = -1;
	md.on_thermo_output = 200;
	md.run((int)(round(120.0/md.dt)), (char*)"EQ");

	//main run:
	if(comm_rank == 0)
		cout<<">> md.apply_virus(VirusInitRatio);"<<endl;
	md.dt = 0.1;
	md.apply_virus(VirusInitRatio);
//	md.on_dump_trajectories = 900;//200;
	md.on_thermo_output = 3000;
//	md.on_compute_density_map = 300;
//	md.on_compute_rdf = 500;
	md.on_compute_flux_vs_time = 6000;
	if(comm_rank == 0)
		cout<<">> md.run((int)(round(14400.0/md.dt)), (char*)SIMNAME);"<<endl;
	md.run((int)(round(14400.0/md.dt)), (char*)SIMNAME);

	//fin:
	if(comm_rank == 0)
		cout<<">> task complete"<<endl;
#ifdef DEF_USING_MPI
	if(comm_rank == 0)
		cout<<"MPI_Finalize()"<<endl;
	MPI_Finalize();
	return EXIT_SUCCESS;
#endif
	return 0;
}


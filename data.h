#ifndef _DATA_H_
#define _DATA_H_

#include <stdlib.h>
#include <time.h>

namespace MY_NET
{
	class type_1D
	{
		public:
			double* val;
			double* err;
			double* bias;
			int size;
	
		type_1D(int sz);
		void show_val();
		void show_err();
		int max_idx();
		~type_1D();
	};
	
	class type_2D
	{
		public:
			double** val;
			double** err;
			double** bias;
			int row;
			int col;
		type_2D(int i_row,int i_col);
		void show_val();
		void show_err();
		void zero_err();
		~type_2D();
	};
	
	class Layer_type_2D
	{
		public:
			int row;
			int col;
			int size;
			type_2D** image;
	
		Layer_type_2D(int i_size,int i_row,int i_col);
		void show_val(int idx);
		void show_err(int idx);
		void zero_err(int idx);
		~Layer_type_2D();
	};
}

#endif

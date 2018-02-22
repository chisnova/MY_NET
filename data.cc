

class type_1D
{
	public:
		double* val;
		double* err;
		double* bias;
		int size;

	type_1D(int sz)
	{
		size=sz;
		val=(double*)malloc(sizeof(double)*size);
		err=(double*)malloc(sizeof(double)*size);
		bias=(double*)malloc(sizeof(double)*size);
		double tmp;
		for(int i=0;i<size;++i){
			tmp=double(rand()%1000-500);
			tmp=tmp*0.001;
			val[i]=tmp;	
			
			tmp=double(rand()%1000-500);
			tmp=tmp*0.001;
			err[i]=tmp;	

			tmp=double(rand()%1000-500);
			tmp=tmp*0.001;
			bias[i]=tmp;	
		}
	}
	void show_val()
	{
		for(int i=0;i<size;++i) printf("%lf ",val[i]);
		printf("\n");
	}

	void show_err()
	{
		for(int i=0;i<size;++i) printf("%lf ",err[i]);
		printf("\n");
	}

	int max_idx()
	{
		double max=-987654321;
		int idx=-1;
		for(int i=0;i<size;++i){
			if(val[i]>max){
				max=val[i];
				idx=i;
			}
		}
		return idx;
	}

	~type_1D()
	{
		free(val);
		free(err);
	}
};

class type_2D
{
	public:
		double** val;
		double** err;
		double** bias;
		int row;
		int col;
	type_2D(int i_row,int i_col)
	{
		row=i_row;
		col=i_col;

		val=(double**)malloc(sizeof(double*)*row);
		err=(double**)malloc(sizeof(double*)*row);
		bias=(double**)malloc(sizeof(double*)*row);
		for(int i=0;i<row;++i) val[i]=(double*)malloc(sizeof(double)*col);
		for(int i=0;i<row;++i) err[i]=(double*)malloc(sizeof(double)*col);
		for(int i=0;i<row;++i) bias[i]=(double*)malloc(sizeof(double)*col);
		double tmp;
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				tmp=double(rand()%1000-500);
				tmp=tmp*0.001;
				val[i][j]=tmp;	

				tmp=double(rand()%1000-500);
				tmp=tmp*0.001;
				err[i][j]=tmp;	
			}
		}
	}

	void show_val()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",val[i][j]);
			}
			printf("\n");
		}
	}

	void show_err()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",err[i][j]);
			}
			printf("\n");
		}
	}
	void zero_err()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				err[i][j]=0;
			}
		}
	}

	~type_2D()
	{
		for(int i=0;i<row;++i) free(val[i]);
		free(val);
	}
};

class Layer_type_2D
{
	public:
		int row;
		int col;
		int size;
		type_2D** image;

	Layer_type_2D(int i_size,int i_row,int i_col)
	{
		size=i_size;
		row=i_row;
		col=i_col;
		image = (type_2D**)malloc(sizeof(type_2D*)*size);
		for(int i=0;i<size;++i) image[i] = new type_2D(row,col);
	}

	void show_val(int idx)
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",image[idx]->val[i][j]);
			}
			printf("\n");
		}
	}

	void show_err(int idx)
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",image[idx]->err[i][j]);
			}
			printf("\n");
		}
	}
	void zero_err(int idx)
	{
		image[idx]->zero_err();
	}

	~Layer_type_2D()
	{
		for(int i=0;i<size;++i) delete image[i];
		free(image);
	}
};

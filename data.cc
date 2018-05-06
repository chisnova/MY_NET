#include "data.h"
#include "math.h"

namespace MY_NET
{
	double inverse_cdf(double pick,double mean,double var,double min,double max)
	{
		double step=0.01;		
		double pi=3.1415927;
		double stv=sqrt(var);
		int len;
		double sum,x,val;
		double mid;
		double b_min=min;
		double b_max=max;
	
		while(1){
			mid=(b_min+b_max)/2.0;
			len=int((mid-min)/step);
			sum=0;
			for(int i=0;i<len;++i){
				x=min+(double)i*step-mean;
				val=exp(-(x*x)/(2*var))*(1/(sqrt(2*pi)*stv))*step;
				sum+=val;
			}
			//printf("sum %lf pick %lf min %lf mid %lf max %lf\n",sum,pick,b_min,mid,b_max);
			if(abs(sum-pick)<0.000001 || abs(b_min-b_max)<0.00001) break;
			else if(sum>pick){
				b_max=mid;	
			}
			else{
				b_min=mid;
			}
		}
		return mid;
	}

	type_1D::type_1D(int sz)
	{
		size=sz;
		val=(double*)malloc(sizeof(double)*size);
		err=(double*)malloc(sizeof(double)*size);
		bias=(double*)malloc(sizeof(double)*size);
		double tmp,var;
		for(int i=0;i<size;++i){
			val[i]=0;	
			err[i]=0;	
			bias[i]=0;	
		}
	}
	void type_1D::show_val()
	{
		for(int i=0;i<size;++i) printf("%lf ",val[i]);
		printf("\n");
	}
	
	void type_1D::show_err()
	{
		for(int i=0;i<size;++i) printf("%lf ",err[i]);
		printf("\n");
	}
	
	int type_1D::max_idx()
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
	
	type_1D::~type_1D()
	{
		free(val);
		free(err);
	}
	
	type_2D::type_2D(int i_row,int i_col)
	{
		row=i_row;
		col=i_col;
	
		val=(double**)malloc(sizeof(double*)*row);
		err=(double**)malloc(sizeof(double*)*row);
		bias=(double**)malloc(sizeof(double*)*row);
		for(int i=0;i<row;++i) val[i]=(double*)malloc(sizeof(double)*col);
		for(int i=0;i<row;++i) err[i]=(double*)malloc(sizeof(double)*col);
		for(int i=0;i<row;++i) bias[i]=(double*)malloc(sizeof(double)*col);
		double tmp,var;
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				tmp=double(rand()%10000);
				tmp=tmp*0.00001;
				var=2.0/double(row+col)*32.0*32.0;
				val[i][j]=inverse_cdf(tmp,0,var,-1,1);	
				err[i][j]=0;	
			}
		}
	}
	
	void type_2D::show_val()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",val[i][j]);
			}
			printf("\n");
		}
	}
	
	void type_2D::show_err()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",err[i][j]);
			}
			printf("\n");
		}
	}
	void type_2D::zero_err()
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				err[i][j]=0;
			}
		}
	}
	
	type_2D::~type_2D()
	{
		for(int i=0;i<row;++i) free(val[i]);
		free(val);
	}
	
	
	Layer_type_2D::Layer_type_2D(int i_size,int i_row,int i_col)
	{
		size=i_size;
		row=i_row;
		col=i_col;
		image = (type_2D**)malloc(sizeof(type_2D*)*size);
		for(int i=0;i<size;++i) image[i] = new type_2D(row,col);
	}
	
	void Layer_type_2D::show_val(int idx)
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",image[idx]->val[i][j]);
			}
			printf("\n");
		}
	}
	
	void Layer_type_2D::show_err(int idx)
	{
		for(int i=0;i<row;++i){
			for(int j=0;j<col;++j){
				printf("%lf ",image[idx]->err[i][j]);
			}
			printf("\n");
		}
	}
	void Layer_type_2D::zero_err(int idx)
	{
		image[idx]->zero_err();
	}
	
	Layer_type_2D::~Layer_type_2D()
	{
		for(int i=0;i<size;++i) delete image[i];
		free(image);
	}

};

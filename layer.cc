#include "layer.h"

namespace MY_NET
{
	bool layer::isRange(int w,int w_max,int d,int d_max)
	{
		if(w <= w_max && d <= d_max) return true;
		return false;
	}

	void layer::MatMul(type_1D* in, type_2D* w,type_1D* out)
	{
		for(int i=0;i<w->row;++i){
		double sum=0;
			for(int j=0;j<w->col;++j){
				sum+=(w->val[i][j])*(in->val[j]);	
			}
			out->val[i]=sum;
		}
	}
	void layer::MatMul(type_1D* in, type_2D* w,type_1D* out,double lr)
	{
		for(int i=0;i<w->row;++i){
		double sum=0;
			for(int j=0;j<w->col;++j){
				sum+=(w->val[i][j])*(in->err[j]);	
			}
			out->err[i]=sum;
		}
	}

	void layer::sigmoid(type_1D* in,type_1D* out)
	{
		for(int i=0;i<in->size;++i){
			out->val[i]=1/(1+exp(-(in->val[i])));	
		}
	}
	void layer::sigmoid(type_1D* in,type_1D* out,double lr)
	{
		for(int i=0;i<in->size;++i){
			double y=1/(1+exp(-(in->val[i])));
			in->err[i]=y*(1-y)*(out->err[i]);	
		}
	}

	void layer::relu(type_1D* in,type_1D* out)
	{
		for(int i=0;i<in->size;++i){
			if(in->val[i]>=0) out->val[i]=in->val[i];
			else out->val[i]=0;
		}
	}
	void layer::relu(type_1D* in,type_1D* out,double lr)
	{
		for(int i=0;i<in->size;++i){
			if(in->val[i]<=0) in->err[i]=0;
			else in->err[i]=out->err[i];
		}
	}
	
	void layer::tanh(type_1D* in,type_1D* out)
	{
		for(int i=0;i<in->size;++i){
			out->val[i]=(exp(2*(in->val[i]))-1)/(exp(2*(in->val[i]))+1);	
		}
	}
	void layer::tanh(type_1D* in,type_1D* out,double lr)
	{
		for(int i=0;i<in->size;++i){
			double y=out->val[i];
			in->err[i]=(1+y)*(1-y)*(out->err[i]);
		}
	}

	void layer::softmax(type_1D* in,type_1D* out)
	{
		double sum=0;
		for(int i=0;i<in->size;++i){
			sum+=exp(in->val[i]);
		}
		for(int i=0;i<in->size;++i){
			out->val[i]=exp(in->val[i])/sum;	
		}
	}
	void layer::softmax(type_1D* in,type_1D* out,double lr)
	{
		for(int i=0;i<in->size;++i){
			in->err[i]=0;
			double yi=out->val[i];
			for(int j=0;j<in->size;++j){
				double yj=out->val[j];
				if(i==j) {
					in->err[i]+=yi*(1-yj)*(out->err[j]);
				}
				else{
					in->err[i]+=-yi*yj*(out->err[j]);
				}
			}
		}
	}

	void layer::affine(type_1D* in,type_2D* w,type_1D* out)
	{
		MatMul(in,w,out);
		for(int i=0;i<out->size;++i) out->val[i]-=out->bias[i];
	}

	void layer::affine(type_1D* in,type_2D* w,type_1D* out,double lr)
	{
		type_2D* Trans=new type_2D(w->col,w->row);
		for(int i=0;i<w->row;++i){
			for(int j=0;j<w->col;++j){
				Trans->val[j][i]=w->val[i][j];	
			}
		}
		MatMul(out,Trans,in,lr);
		for(int i=0;i<w->row;++i){
			for(int j=0;j<w->col;++j){
				w->val[i][j]=0.99*(w->val[i][j])-lr*(in->val[j])*(out->err[i]);
			}
		}
		for(int i=0;i<out->size;++i){
			out->bias[i]=0.99*out->bias[i]+(out->err[i])*lr;
		}
		delete Trans;
	}
	//Layer 2D
	void layer::sigmoid(Layer_type_2D* in,Layer_type_2D* out)
	{
		double x;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					x=in->image[k]->val[i][j];
					out->image[k]->val[i][j]=1/(1+exp(-x));	
				}
			}
		}
	}
	void layer::sigmoid(Layer_type_2D* in,Layer_type_2D* out,double lr)
	{
		double y;
		double err;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					y=out->image[k]->val[i][j];
					err=out->image[k]->err[i][j];
					in->image[k]->err[i][j]=y*(1-y)*err;	
				}
			}
		}
	}

	void layer::relu(Layer_type_2D* in,Layer_type_2D* out)
	{
		double x;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					x=in->image[k]->val[i][j];
					if(in->image[k]->val[i][j]>=0) out->image[k]->val[i][j]=x;
					else out->image[k]->val[i][j]=0;
				}
			}
		}
	}
	void layer::relu(Layer_type_2D* in,Layer_type_2D* out,double lr)
	{
		double err;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					err=in->image[k]->err[i][j];
					if(in->image[k]->val[i][j]<=0) in->image[i]->err[i][j]=0;
					else in->image[k]->err[i][j]=err;
				}
			}
		}
	}

	void layer::tanh(Layer_type_2D* in,Layer_type_2D* out)
	{
		double x;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					x=in->image[k]->val[i][j];
					out->image[k]->val[i][j]=(exp(2*x-1)/(exp(2*x)+1));	
				}
			}
		}
	}

	void layer::tanh(Layer_type_2D* in,Layer_type_2D* out,double lr)
	{
		double err;
		double y;
		for(int k=0;k<in->size;++k){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					y=out->image[k]->val[i][j];
					err=in->image[k]->err[i][j];
					in->image[k]->err[i][j]=(1+y)*(1-y)*err;	
				}
			}
		}
	}
	void layer::softmax(Layer_type_2D* in,Layer_type_2D* out)
	{
		double sum;
		double x;
		for(int k=0;k<in->size;++k){
			sum=0;
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					sum+=exp(in->image[k]->val[i][j]);
				}
			}
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					out->image[k]->val[i][j]=exp(in->image[k]->val[i][j])/sum;	
				}
			}
		}
	}

	void layer::softmax(Layer_type_2D* in,Layer_type_2D* out,double lr)
	{
		double yi;
		double yj;
		double err;
		for(int l=0;l<in->size;++l){
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					yi=out->image[l]->val[i][j];
					in->image[l]->err[i][j]=0;
					for(int k=0;k<in->row;++k){
						for(int n=0;n<in->col;++n){
							yj=out->image[l]->val[k][n];
							err=out->image[l]->err[k][n];
							if(i==k && j ==n) in->image[l]->err[i][j]+=yi*(1-yi)*err;	
							else in->image[l]->err[i][j]+=-yi*yj*err;	
						}
					}
				}
			}
		}
	}

	void layer::conv(Layer_type_2D* in,Layer_type_2D* kernel,Layer_type_2D* out)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int i_size=in->size;
		int k_width=kernel->col;
		int k_depth=kernel->row;
		int k_size=(k_width)*(k_depth);

		for(int m=0;m<kernel->size;++m){
			for(int n=0;n<in->size;++n){
				for(int i=0;i<i_depth;++i){
					for(int j=0;j<i_width;++j){
						if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
						double sum=0;
						for(int k=0;k<k_size;++k){
							int dy=k/k_width;
							int dx=k%k_width;
							sum+=(in->image[n]->val[i+dy][j+dx])*(kernel->image[m]->val[dy][dx]);	
						}
						if(n==0) out->image[m]->val[i][j]=sum;	
						else out->image[m]->val[i][j]+=sum;	
					}	
				}
			}
		}
	}

	void layer::conv(Layer_type_2D* in,Layer_type_2D* kernel,Layer_type_2D* out,double lr)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int i_size=in->size;
		int k_width=kernel->col;
		int k_depth=kernel->row;
		int k_size=(k_width)*(k_depth);
		int dy,dx;	
		double err;
		double k_value;

		for(int n=0;n<in->size;++n){
			in->image[n]->zero_err();
			for(int m=0;m<kernel->size;++m){
	   			for(int i=0;i<i_depth;++i){
	   				for(int j=0;j<i_width;++j){
	   					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
	   					for(int k=0;k<k_size;++k){
	   						dy=k/k_width;
	   						dx=k%k_width;
							k_value=kernel->image[m]->val[dy][dx];
							err=out->image[m]->err[i][j];
							in->image[n]->err[i+dy][j+dx]+=k_value*err;	
	   					}
	   				}	
	    		}
			}
		}

		double update;
		for(int m=0;m<kernel->size;++m){
			for(int n=0;n<in->size;++n){
				for(int i=0;i<i_depth;++i){
	   				for(int j=0;j<i_width;++j){
	   					for(int k=0;k<k_size;++k){
	   						dy=k/k_width;
	   						dx=k%k_width;
	   						if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
	   						update=(in->image[n]->val[i+dy][j+dx])*(out->image[m]->err[i][j]);	
							kernel->image[m]->val[dy][dx]=0.99*(kernel->image[m]->val[dy][dx])-update*lr;
	   					}
	   				}	
				}
			}
		}
	}

	void layer::max_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int k_width=poolsize;
		int k_depth=poolsize;
		int k_size=(k_width)*(k_depth);
		double max;
		int dy,dx;

		for(int n=0;n<in->size;++n){
			for(int i=0;i<i_depth;i+=poolsize){
				for(int j=0;j<i_width;j+=poolsize){
					max=-987654321;
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					for(int k=0;k<k_size;++k){
						dy=k/k_width;
						dx=k%k_width;
						if(in->image[n]->val[i+dy][j+dx]>max) max=in->image[n]->val[i+dy][j+dx];
					}
					out->image[n]->val[i/poolsize][j/poolsize]=max;	
				}
			}
		}
	}

	void layer::max_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize,double lr)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int k_width=poolsize;
		int k_depth=poolsize;
		int k_size=(k_width)*(k_depth);
		int dy,dx;
		int max_x,max_y;
		double max;

		for(int n=0;n<in->size;++n){
			in->image[n]->zero_err();
			for(int i=0;i<i_depth;i+=poolsize){
				for(int j=0;j<i_width;j+=poolsize){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					max=-987654321;
					for(int k=0;k<k_size;++k){
						dy=k/k_width;
						dx=k%k_width;
						if(in->image[n]->val[i+dy][j+dx]>max){
							max_y=i+dy;
							max_x=j+dx;
							max=in->image[n]->val[i+dy][j+dx];
						}
					}
					in->image[n]->err[max_y][max_x]=out->image[n]->err[i/poolsize][j/poolsize];
				}
			}
		}
		
	}

	void layer::average_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int k_width=poolsize;
		int k_depth=poolsize;
		int k_size=(k_width)*(k_depth);
		double sum;
		int dy,dx;

		for(int n=0;n<in->size;++n){
			for(int i=0;i<i_depth;i+=poolsize){
				for(int j=0;j<i_width;j+=poolsize){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					sum=0;
					for(int k=0;k<k_size;++k){
						dy=k/k_width;
						dx=k%k_width;
						sum+=(in->image[n]->val[i+dy][j+dx]);	
					}
					out->image[n]->val[i/poolsize][j/poolsize]=sum/(double)k_size;	
				}
			}
		}
	}

	void layer::average_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize,double lr)
	{
		int i_width=in->col;
		int i_depth=in->row;
		int k_width=poolsize;
		int k_depth=poolsize;
		int k_size=(k_width)*(k_depth);
		int dy,dx;

		for(int n=0;n<in->size;++n){
			for(int i=0;i<i_depth;i+=poolsize){
				for(int j=0;j<i_width;j+=poolsize){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					for(int k=0;k<k_size;++k){
						dy=k/k_width;
						dx=k%k_width;
						in->image[n]->err[i+dy][j+dx]=(out->image[n]->err[i/poolsize][j/poolsize])/(double)k_size;
					}
				}
			}
		}
	}
}

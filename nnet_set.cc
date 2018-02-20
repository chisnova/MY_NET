#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct _MNIST_
{
	double image[60000][32][32];
	double label[60000][10];
	int idx[60000];
}mnist;

mnist train_data;
mnist test_data;

int readMnist(mnist* train,mnist* test)
{
	unsigned char image_info[16] = {1,};
	unsigned char temp_image[28][28] = {0,};

	unsigned char label_info[16] = {1,};
	unsigned char temp_label=0;
	
	FILE* train_label_f;
	FILE* train_image_f;
	FILE* test_label_f;
	FILE* test_image_f;
	
	test_image_f = fopen("MNIST/t10k-images-idx3-ubyte","rb");
	test_label_f = fopen("MNIST/t10k-labels-idx1-ubyte","rb");
	train_image_f = fopen("MNIST/train-images-idx3-ubyte","rb");	
	train_label_f = fopen("MNIST/train-labels-idx1-ubyte","rb");	
	
	fread(label_info,sizeof(unsigned char),sizeof(unsigned char)*8,train_label_f);//2049,60000-number
	fread(image_info,sizeof(unsigned char),sizeof(unsigned char)*16,train_image_f);//2051,60000-number,28by28-size

	fread(label_info,sizeof(unsigned char),sizeof(unsigned char)*8,test_label_f);//2049,60000-number
	fread(image_info,sizeof(unsigned char),sizeof(unsigned char)*16,test_image_f);//2051,60000-number,28by28-size

	for(int k = 0; k<60000;k++)
	{
		fread(&temp_label,sizeof(unsigned char),1,train_label_f);
		fread(temp_image,sizeof(unsigned char),28*28,train_image_f);
	
		for(int i=0;i<10;++i) train->label[k][i]=0;
		train->label[k][int(temp_label)]=1;
		train->idx[k]=int(temp_label);
		for(int i = 0;i<32;i++){
			for(int j = 0;j<32;j++){
				train->image[k][i][j]=0;
			}
		}
		for(int i = 2;i<30;i++){
			for(int j = 2;j<30;j++){
				train->image[k][i][j]=(double)temp_image[i-2][j-2];
			}
		}
	}
	for(int k = 0; k<10000;k++)
	{
		fread(&temp_label,sizeof(unsigned char),1,test_label_f);
		fread(temp_image,sizeof(unsigned char),28*28,test_image_f);
	
		for(int i=0;i<10;++i) test->label[k][i]=0;
		test->label[k][int(temp_label)]=1;
		test->idx[k]=int(temp_label);
		for(int i = 0;i<32;i++){
			for(int j = 0;j<32;j++){
				test->image[k][i][j]=0;
			}
		}
		for(int i = 2;i<30;i++){
			for(int j = 2;j<30;j++){
				test->image[k][i][j]=(double)temp_image[i-2][j-2];
			}
		}
	}
	fclose(train_image_f);
	fclose(train_label_f);
	fclose(test_image_f);
	fclose(test_label_f);
	return 1;
}



class type_1D
{
	public:
		double* val;
		double* err;
		int size;

	type_1D(int sz)
	{
		size=sz;
		val=(double*)malloc(sizeof(double)*size);
		err=(double*)malloc(sizeof(double)*size);
		srand(time(NULL));
		double tmp;
		for(int i=0;i<size;++i){
			tmp=double(rand()%1000-500);
			tmp=tmp*0.001;
			val[i]=tmp;	
			
			tmp=double(rand()%1000-500);
			tmp=tmp*0.001;
			err[i]=tmp;	

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
		int row;
		int col;
	type_2D(int i_row,int i_col)
	{
		row=i_row;
		col=i_col;

		val=(double**)malloc(sizeof(double*)*row);
		err=(double**)malloc(sizeof(double*)*row);
		for(int i=0;i<row;++i) val[i]=(double*)malloc(sizeof(double)*col);
		for(int i=0;i<row;++i) err[i]=(double*)malloc(sizeof(double)*col);
		srand(time(NULL));
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
				printf("%lf ",val[i][j]);
			}
			printf("\n");
		}
	}

	~type_2D()
	{
		for(int i=0;i<row;++i) free(val[i]);
		free(val);
	}
};

class Layer
{
	private:
		void MatMul(type_1D* in, type_2D* w,type_1D* out)
		{
			for(int i=0;i<w->row;++i){
			double sum=0;
				for(int j=0;j<w->col;++j){
					sum+=(w->val[i][j])*(in->val[j]);	
				}
				out->val[i]=sum;
			}
		}
		void MatMul(type_1D* in, type_2D* w,type_1D* out,double lr)
		{
			for(int i=0;i<w->row;++i){
			double sum=0;
				for(int j=0;j<w->col;++j){
					sum+=(w->val[i][j])*(in->err[j]);	
				}
				out->err[i]=sum;
			}
		}
	public:
		void sigmoid(type_1D* in,type_1D* out)
		{
			for(int i=0;i<in->size;++i){
				out->val[i]=1/(1+exp(-(in->val[i])));	
			}
		}
		void sigmoid(type_1D* in,type_1D* out,double lr)
		{
			for(int i=0;i<in->size;++i){
				double y=1/(1+exp(-(in->val[i])));
				in->err[i]=y*(1-y)*(out->err[i]);	
			}
		}

		void relu(type_1D* in,type_1D* out)
		{
			for(int i=0;i<in->size;++i){
				if(in->val[i]>=0) out->val[i]=in->val[i];
				else out->val[i]=0;
			}
		}
		void relu(type_1D* in,type_1D* out,double lr)
		{
			for(int i=0;i<in->size;++i){
				if(in->val[i]<=0) in->err[i]=0;
				else in->err[i]=out->err[i];
			}
		}
		
		void tanh(type_1D* in,type_1D* out)
		{
			for(int i=0;i<in->size;++i){
				out->val[i]=(exp(2*(in->val[i]))-1)/(exp(2*(in->val[i]))+1);	
			}
		}
		void tanh(type_1D* in,type_1D* out,double lr)
		{
			for(int i=0;i<in->size;++i){
				double y=out->val[i];
				in->err[i]=(1+y)*(1-y)*(out->err[i]);
			}
		}

		void softmax(type_1D* in,type_1D* out)
		{
			double sum=0;
			for(int i=0;i<in->size;++i){
				sum+=exp(in->val[i]);
			}
			for(int i=0;i<in->size;++i){
				out->val[i]=exp(in->val[i])/sum;	
			}
		}
		void softmax(type_1D* in,type_1D* out,double lr)
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

		void affine(type_1D* in,type_2D* w,type_1D* out)
		{
			MatMul(in,w,out);
		}

		void affine(type_1D* in,type_2D* w,type_1D* out,double lr)
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
					w->val[i][j]-=lr*(in->val[j])*(out->err[i]);
				}
			}
			delete Trans;
		}
};



class Image
{
	private:
		bool isRange(int w,int w_max,int d,int d_max)
		{
			if(w <= w_max && d <= d_max) return true;
			return false;
		}
	public:
		void sigmoid(type_2D* in,type_2D* out)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					out->val[i][j]=1/(1+exp(-(in->val[i][j])));	
				}
			}
		}
		void sigmoid(type_2D* in,type_2D* out,double lr)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					double y=out->val[i][j];
					in->err[i][j]=y*(1-y)*(out->err[i][j]);	
				}
			}
		}

		void relu(type_2D* in,type_2D* out)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					if(in->val[i][j]>=0) out->val[i][j]=in->val[i][j];
					else out->val[i][j]=0;
				}
			}
		}
		void relu(type_2D* in,type_2D* out,double lr)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					if(in->val[i][j]<=0) in->err[i][j]=0;
					else in->err[i][j]=out->err[i][j];
				}
			}
		}

		void tanh(type_2D* in,type_2D* out)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					out->val[i][j]=(exp(2*(in->val[i][j]))-1)/(exp(2*(in->val[i][j]))+1);	
				}
			}
		}

		void tanh(type_2D* in,type_2D* out,double lr)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					double y=out->val[i][j];
					in->err[i][j]=(1+y)*(1-y)*(out->err[i][j]);	
				}
			}
		}
		void softmax(type_2D* in,type_2D* out)
		{
			double sum=0;
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					sum+=exp(in->val[i][j]);
				}
			}
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					out->val[i][j]=exp(in->val[i][j])/sum;	
				}
			}
		}

		void softmax(type_2D* in,type_2D* out,double lr)
		{
			for(int i=0;i<in->row;++i){
				for(int j=0;j<in->col;++j){
					double yi=out->val[i][j];
					in->err[i][j]=0;
					for(int k=0;k<in->row;++k){
						for(int n=0;n<in->row;++n){
							double yj=out->val[k][n];
							if(i==k && j ==n) in->err[i][j]+=yi*(1-yi)*(out->err[k][n]);	
							else in->err[i][j]+=-yi*yj*(out->err[k][n]);	
						}
					}
				}
			}
		}

		void conv(type_2D* in,type_2D* kernel,type_2D* out,int acc)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=kernel->col;
			int k_depth=kernel->row;
			int k_size=(k_width)*(k_depth);

			for(int i=0;i<i_depth;++i){
				for(int j=0;j<i_width;++j){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					double sum=0;
					for(int k=0;k<k_size;++k){
						int dy=k/k_width;
						int dx=k%k_width;
						sum+=(in->val[i+dy][j+dx])*(kernel->val[dy][dx]);	
					}
					if(acc==0) out->val[i][j]=sum;	
					else out->val[i][j]+=sum;	
				}	
			}
		}

		void conv(type_2D* in,type_2D* kernel,type_2D* out,int acc,double lr)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=kernel->col;
			int k_depth=kernel->row;
			int k_size=(k_width)*(k_depth);
		
			if(acc==0){
				for(int i=0;i<i_depth;++i){
		   			for(int j=0;j<i_width;++j){
		   				in->err[i][j]=0;
		   			}
		   		}
			}
		   	for(int i=0;i<i_depth;++i){
		   		for(int j=0;j<i_width;++j){
		   			if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
		   			for(int k=0;k<k_size;++k){
		   				int dy=k/k_width;
		   				int dx=k%k_width;
		   				in->err[i+dy][j+dx]+=(kernel->val[dy][dx])*(out->err[i][j]);	
		   			}
		   		}	
		   }
		}

		void conv_update(type_2D* in,type_2D* kernel,type_2D* out,int acc,double lr)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=kernel->col;
			int k_depth=kernel->row;
			int k_size=(k_width)*(k_depth);
		
			double update=0;
			for(int i=0;i<i_depth;++i){
		   		for(int j=0;j<i_width;++j){
		   			for(int k=0;k<k_size;++k){
		   				int dy=k/k_width;
		   				int dx=k%k_width;
		   				if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
		   				update=(in->val[i+dy][j+dx])*(out->err[i][j]);	
						kernel->val[dy][dx]-=update*lr;
		   			}
		   		}	
			}
		}

		void max_pooling(type_2D* in,type_2D* out,int poolsize)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=poolsize;
			int k_depth=poolsize;
			int k_size=(k_width)*(k_depth);
			for(int i=0;i<i_depth;i+=2){
				for(int j=0;j<i_width;j+=2){
					double max=-987654321;
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					double sum=0;
					for(int k=0;k<k_size;++k){
						int dx=k/k_width;
						int dy=k%k_width;
						if(in->val[i+dx][j+dy]>max) max=in->val[i+dx][j+dy];
					}
					out->val[i/2][j/2]=max;	
				}
			}
		}

		void max_pooling(type_2D* in,type_2D* out,int poolsize,double lr)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=poolsize;
			int k_depth=poolsize;
			int k_size=(k_width)*(k_depth);
			for(int i=0;i<i_depth;i+=2){
				for(int j=0;j<i_width;j+=2){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					int max_x,max_y;
					double max=-987654321;
					for(int k=0;k<k_size;++k){
						int dx=k/k_width;
						int dy=k%k_width;
						if(in->val[i+dx][j+dy]>max){
							max_x=i+dx;
							max_y=j+dy;
							max=in->val[i+dx][j+dy];
						}
					}
					for(int k=0;k<k_size;++k){
						int dx=k/k_width;
						int dy=k%k_width;
						if(max_x==(i+dx) && max_y==(j+dy)){
							in->err[i+dx][j+dy]=out->err[i/2][j/2];
						}
						else{
							in->err[i+dx][j+dy]=0;
						}
					}
				}
			}
		}

		void average_pooling(type_2D* in,type_2D* out,int poolsize)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=poolsize;
			int k_depth=poolsize;
			int k_size=(k_width)*(k_depth);
			for(int i=0;i<i_depth;i+=2){
				for(int j=0;j<i_width;j+=2){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					double sum=0;
					for(int k=0;k<k_size;++k){
						int dx=k/k_width;
						int dy=k%k_width;
						sum+=(in->val[i+dx][j+dy]);	
					}
					out->val[i/2][j/2]=sum/(double)k_size;	
				}
			}
		}

		void average_pooling(type_2D* in,type_2D* out,int poolsize,double lr)
		{
			int i_width=in->col;
			int i_depth=in->row;
			int k_width=poolsize;
			int k_depth=poolsize;
			int k_size=(k_width)*(k_depth);
			for(int i=0;i<i_depth;i+=2){
				for(int j=0;j<i_width;j+=2){
					if(!isRange(i+k_depth,i_depth,j+k_width,i_width)) break;
					for(int k=0;k<k_size;++k){
						int dx=k/k_width;
						int dy=k%k_width;
						in->err[i+dx][j+dy]=(out->err[i/2][j/2])/(double)k_size;
					}
				}
			}
		}
};

#define layer_size 5
void DNN()
{
	readMnist(&train_data,&test_data);

	int tr_size=100;
	int test_size=10;
	double lr=0.01;
	int layer_info[layer_size]={32*32,100,100,10,10};
	char function_info[layer_size-1][100]={"affine","relu","affine","sigmoid"};

	type_1D* x[layer_size];
	for(int i=0;i<layer_size;++i) x[i]=new type_1D(layer_info[i]);

	type_2D* w[layer_size];
	int cnt=0;
	for(int i=0;i<layer_size-1;++i){
		if(strcmp(function_info[i],"affine")==0){
			w[i] = new type_2D(layer_info[i+1],layer_info[i]);
		}
	}
	
	Layer test;

	int hit;
	int miss;
	double mse;
	while(1){

		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<tr_size;++tr){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=train_data.image[tr][i][j]*0.001;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) test.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) test.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) test.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) test.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0)test.softmax(x[i],x[i+1]);
			}
			
			if(x[layer_size-1]->max_idx()==train_data.idx[tr]) hit++;
			else miss++;
			
			for(int i=0;i<10;++i){x[layer_size-1]->err[i]=x[layer_size-1]->val[i]-train_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(x[layer_size-1]->err[i])*(x[layer_size-1]->err[i]);}

			for(int i=layer_size-2;i>-1;--i){
				if(strcmp(function_info[i],"affine")==0) test.affine(x[i],w[i],x[i+1],lr);
				if(strcmp(function_info[i],"relu")==0) test.relu(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"sigmoid")==0) test.sigmoid(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"tanh")==0) test.tanh(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"softmax")==0)test.softmax(x[i],x[i+1],lr);
			}
		}
		printf("Train Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Train MSE %lf\n",sqrt(mse/double(hit+miss)));

		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<test_size;++tr){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=test_data.image[tr][i][j]*0.001;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) test.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) test.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) test.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) test.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0)test.softmax(x[i],x[i+1]);
			}
			
			if(x[layer_size-1]->max_idx()==test_data.idx[tr]) hit++;
			else miss++;
			
			for(int i=0;i<10;++i){x[layer_size-1]->err[i]=x[layer_size-1]->val[i]-test_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(x[layer_size-1]->err[i])*(x[layer_size-1]->err[i]);}

		}
		printf("Test Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Test MSE %lf\n",sqrt(mse/double(hit+miss)));
	}
}

void LeNet()
{
	readMnist(&train_data,&test_data);

	int tr_size=100;
	int test_size=10;
	double lr=0.001;

	type_2D* L1_x = new type_2D(32,32);

	type_2D* L1_k[6];
	for(int i=0;i<6;++i) L1_k[i] = new type_2D(5,5);

	type_2D* L2_x[6];
	for(int i=0;i<6;++i) L2_x[i] = new type_2D(28,28);

	type_2D* L3_s[6];
	for(int i=0;i<6;++i) L3_s[i] = new type_2D(14,14);
	
	type_2D* L3_k[16];
	for(int i=0;i<16;++i) L3_k[i] = new type_2D(5,5);
	
	type_2D* L4_x[16];
	for(int i=0;i<16;++i) L4_x[i] = new type_2D(10,10);

	type_2D* L5_s[16];
	for(int i=0;i<16;++i) L5_s[i] = new type_2D(5,5);
	
	type_2D* L5_k[120];
	for(int i=0;i<120;++i) L5_k[i] = new type_2D(5,5);
	
	type_2D* L6_x[120];
	for(int i=0;i<120;++i) L6_x[i] =new type_2D(1,1);

	type_1D* x1 = new type_1D(120);
	type_2D* w1 = new type_2D(84,120);

	type_1D* x2 = new type_1D(84);
	type_1D* a2 = new type_1D(84);
	type_2D* w2 = new type_2D(10,84);

	type_1D* x3 = new type_1D(10);
	type_1D* a3 = new type_1D(10);

	Image test;
	Layer test2;
	int hit;
	int miss;
	double mse;
	while(1)
	{
		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<tr_size;++tr){
			//FORWARD
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){L1_x->val[i][j]=train_data.image[tr][i][j]*0.001;}}
			for(int i=0;i<6;++i)	test.conv(L1_x,L1_k[i],L2_x[i],0);	
			for(int i=0;i<6;++i)	test.average_pooling(L2_x[i],L3_s[i],2);	
			for(int i=0;i<16;++i){	for(int j=0;j<6;++j){test.conv(L3_s[j],L3_k[i],L4_x[i],j);}}
			for(int i=0;i<16;++i)	test.average_pooling(L4_x[i],L5_s[i],2);
			for(int i=0;i<120;++i){	for(int j=0;j<16;++j){test.conv(L5_s[j],L5_k[i],L6_x[i],j);}}
			for(int i=0;i<120;++i)	x1->val[i]=L6_x[i]->val[0][0];
			test2.affine(x1,w1,x2);
			test2.sigmoid(x2,a2);
			test2.affine(a2,w2,x3);
			test2.sigmoid(x3,a3);


			if(a3->max_idx()==train_data.idx[tr]) hit++;
			else miss++;
			for(int i=0;i<10;++i){a3->err[i]=a3->val[i]-train_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(a3->err[i])*(a3->err[i]);}

			//BACKWARD
			test2.sigmoid(x3,a3,lr);
			test2.affine(a2,w2,x3,lr);
			test2.sigmoid(x2,a2,lr);
			test2.affine(x1,w1,x2,lr);
			for(int i=0;i<120;++i) L6_x[i]->err[0][0]=x1->err[i];
			for(int i=0;i<16;++i){for(int j=0;j<120;++j){test.conv(L5_s[i],L5_k[j],L6_x[j],j,lr);}}
			for(int i=0;i<16;++i){for(int j=0;j<120;++j){test.conv_update(L5_s[i],L5_k[j],L6_x[j],j,lr);}}
			for(int i=0;i<16;++i)	test.average_pooling(L4_x[i],L5_s[i],2,lr);
			for(int i=0;i<6;++i){for(int j=0;j<16;++j){	test.conv(L3_s[i],L3_k[j],L4_x[j],j,lr);}}
			for(int i=0;i<6;++i){for(int j=0;j<16;++j){	test.conv_update(L3_s[i],L3_k[j],L4_x[j],j,lr);}}
			for(int i=0;i<6;++i)	test.average_pooling(L2_x[i],L3_s[i],2,lr);	
			for(int i=0;i<6;++i)	test.conv(L1_x,L1_k[i],L2_x[i],0,lr);	
			for(int i=0;i<6;++i)	test.conv_update(L1_x,L1_k[i],L2_x[i],0,lr);	
		}
		printf("Train Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Train MSE %lf\n",sqrt(mse/double(hit+miss)));

		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<test_size;++tr){
			//FORWARD
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){L1_x->val[i][j]=test_data.image[tr][i][j]*0.001;}}
			for(int i=0;i<6;++i)	test.conv(L1_x,L1_k[i],L2_x[i],0);	
			for(int i=0;i<6;++i)	test.average_pooling(L2_x[i],L3_s[i],2);	
			for(int i=0;i<16;++i){	for(int j=0;j<6;++j){test.conv(L3_s[j],L3_k[i],L4_x[i],j);}}
			for(int i=0;i<16;++i)	test.average_pooling(L4_x[i],L5_s[i],2);
			for(int i=0;i<120;++i){	for(int j=0;j<16;++j){test.conv(L5_s[j],L5_k[i],L6_x[i],j);}}
			for(int i=0;i<120;++i)	x1->val[i]=L6_x[i]->val[0][0];
			test2.affine(x1,w1,x2);
			test2.sigmoid(x2,a2);
			test2.affine(a2,w2,x3);
			test2.sigmoid(x3,a3);

			if(a3->max_idx()==test_data.idx[tr]) hit++;
			else miss++;

			for(int i=0;i<10;++i){a3->err[i]=a3->val[i]-test_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(a3->err[i])*(a3->err[i]);}

		}
		printf("Test Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Test MSE %lf\n",sqrt(mse/double(hit+miss)));
	}
}


int main()
{
	//DNN();	
	LeNet();	
	return 0;	
}


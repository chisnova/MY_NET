#include "w2v.h"
using namespace MY_NET;


#define layer_size 5
void w2v_DNN()
{
	srand((unsigned int)time(NULL));
	//35661
	int tr_iter=500,dev_iter=10,batch_size=500,hit,miss;
	double lr=0.001;
	double mse,mean=0,var=0,stv=0,Recog,MSE,err;

	int idx;
	int layer_info[layer_size]={35661*4,200,200,35661,35661};
	char function_info[layer_size-1][100]={"affine","relu","affine","sigmoid"};

	layer dev;

	type_1D* x[layer_size];
	for(int i=0;i<layer_size;++i) x[i]=new type_1D(layer_info[i]);

	type_2D* w[layer_size];
	int cnt=0;
	for(int i=0;i<layer_size-1;++i){
		if(strcmp(function_info[i],"affine")==0){
			w[i] = new type_2D(layer_info[i+1],layer_info[i]);
		}
	}
	
	double*** batch_tr;
	double** batch_label;
	int* batch_idx;

	batch_tr=(double***)malloc(sizeof(double**)*batch_size);
	batch_idx=(int*)malloc(sizeof(int)*batch_size);

	for(int i=0;i<batch_size;++i){
		batch_tr[i]=(double**)malloc(sizeof(double*)*32);
		for(int j=0;j<32;++j) batch_tr[i][j]=(double*)malloc(sizeof(double)*32);
	}

	batch_label=(double**)malloc(sizeof(double*)*batch_size);
	for(int i=0;i<batch_size;++i){
		batch_label[i]=(double*)malloc(sizeof(double)*10);
	}

	int iteration=0;
	while(iteration<tr_iter){
		iteration++;

		hit=0,miss=0,mse=0,mean=0,var=0,stv=0;
		for(int i=0;i<batch_size;++i){
			idx=rand()%60000;
			for(int j=0;j<32;++j){
				for(int k=0;k<32;++k){
					batch_tr[i][j][k]=train_data.image[idx][j][k];
					mean+=batch_tr[i][j][k];
					var+=batch_tr[i][j][k]*batch_tr[i][j][k];
				}
			}
			for(int j=0;j<10;++j){
				batch_label[i][j]=train_data.label[idx][j];
			}
			batch_idx[i]=train_data.idx[idx];
		}

		mean=mean/double(batch_size*32*32);
		var=(var/double(batch_size*32*32))-mean*mean;
		stv=sqrt(var);

		for(int batch=0;batch<batch_size;++batch){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=(batch_tr[batch][i][j]-mean)/stv;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) dev.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) dev.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) dev.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) dev.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0) dev.softmax(x[i],x[i+1]);
			}
			
			if(x[layer_size-1]->max_idx()==batch_idx[batch]) hit++;
			else miss++;
			
			for(int i=0;i<10;++i){x[layer_size-1]->err[i]=x[layer_size-1]->val[i]-batch_label[batch][i];}
			for(int i=0;i<10;++i){mse+=(x[layer_size-1]->err[i])*(x[layer_size-1]->err[i]);}

			for(int i=layer_size-2;i>-1;--i){
				if(strcmp(function_info[i],"affine")==0) dev.affine(x[i],w[i],x[i+1],lr);
				if(strcmp(function_info[i],"relu")==0) dev.relu(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"sigmoid")==0) dev.sigmoid(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"tanh")==0) dev.tanh(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"softmax")==0) dev.softmax(x[i],x[i+1],lr);
			}
		}

		Recog=double(hit)/(double(hit)+double(miss))*100;
		MSE=sqrt(mse/double(hit+miss));
		printf("Train Recognition Rate %lf Train MSE %lf\n",Recog,MSE);

		hit=0,miss=0,mse=0;

		for(int tr=0;tr<dev_iter;++tr){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=(test_data.image[tr][i][j]-mean)/stv;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) dev.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) dev.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) dev.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) dev.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0) dev.softmax(x[i],x[i+1]);
			}
			
			if(x[layer_size-1]->max_idx()==test_data.idx[tr]) hit++;
			else miss++;
			
			for(int i=0;i<10;++i){x[layer_size-1]->err[i]=x[layer_size-1]->val[i]-test_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(x[layer_size-1]->err[i])*(x[layer_size-1]->err[i]);}

		}
		Recog=double(hit)/(double(hit)+double(miss))*100;
		MSE=sqrt(mse/double(hit+miss));
		printf("Test Recognition Rate %lf Test MSE %lf\n",Recog,MSE);
	}
}


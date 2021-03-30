#include "dnn.h"

using namespace MY_NET;

typedef struct _DNN_INFO
{
	int layer_size=5;
	double lr;
	int* topology[layer_size]={32*32,100,100,10,10};
	char* function[layer_size-1][100]={"affine","relu","affine","sigmoid"};

	_DNN_INFO(int* topo_input,c)
	{

	}
}DNN_INFO

typedef struct _INPUT_
{
	int dim;
	int batch;
	double** data;
	double** raw;
	double mean;
	double var;
	double std;

	_INPUT_(int _dim,int _batch)
	{
		batch=_batch;
		dim=_dim;
		data=(double**)malloc(sizeof(double*)*batch);
		raw=(double**)malloc(sizeof(double*)*batch);
		for(int i=0;i<batch;++i){
			data[i]=(double*)malloc(sizeof(double)*dim);	
			raw[i]=(double*)malloc(sizeof(double)*dim);	
		}
	}
	void Insert(int batch_idx,int dim_idx,double val)
	{
		raw[batch_idx][dim_idx]=val;
	}

	void normalize()
	{
		for(int i=0;i<batch;++i){
			for(int j=0;j<dim;++j){
				mean+=raw[i][j]/(double)(batch*dim);
				var+=(raw[i][j]*raw[i][j])/(double)(batch*dim);
			}
		}
		var=var-mean*mean;
		std=sqrt(var);
		for(int i=0;i<batch;++i){
			for(int j=0;j<dim;++j){
				data[i][j]=(raw[i][j]-mean)/std;
			}
		}
	}

}INPUT;

typedef struct _LABEL_
{
	int num_pdf;
	double** data;
	int* raw;

	_LABEL_(int _num_pdf)
	{
		num_pdf=_num_pdf;
		data=(double**)malloc(sizeof(double*)*batch);
		raw=(int*)malloc(sizeof(int)*batch);

		for(int i=0;i<batch;++i){
			data[i]=(double*)malloc(sizeof(double)*num_pdf);	
		}

		for(int i=0;i<batch;++i){
			for(int j=0;j<num_pdf;++j){
			data[i][j]=0.0;	
		}
	}

	void Insert(int batch_idx,int val)
	{
		raw[batch_idx]=val;
	}

	void onehot()
	{
		for(int i=0;i<batch;++i){
			data[i][raw[i]]=1.0;	
		}
	}
}LABEL

class DNN
{
	public:
	type_1D** x;
	type_2D** w;
	DNN_INFO info;
	layer dev;

	DNN(DNN_INFO _info)
	{
		srand((unsigned int)time(NULL));
		info=_info;
		layer_size=info.layer_size;

		x=(type_1D**)malloc(sizeof(type_1D)*info.layer_size);
		for(int i=0;i<info.layer_size;++i) x[i]=new type_1D(info.topology[i]);

		w=(type_2D**)malloc(sizeof(type_2D)*info.layer_size);
		for(int i=0;i<info.layer_size-1;++i){
			if(strcmp(info.function[i],"affine")==0){
				w[i] = new type_2D(info.topology[i+1],info.topology[i]);
			}
		}
	}

	int train(INPUT* input,LABEL* label)
	{
		for(int i=0;i<10;++i) x[info.layer_size-1]->err[i]=0;
		double mse=0;
		for(int j=0;j<input->batch;++j){
			for(int i=0;i<info.topology[0];++i){ 
				x[0]->val[i]=input->data[j][i];
			}
			
			for(int i=0;i<info.layer_size-1;++i){
				if(strcmp(info.topology[i],"affine")==0) dev.affine(x[i],w[i],x[i+1]);
				if(strcmp(info.topology[i],"relu")==0) dev.relu(x[i],x[i+1]);
				if(strcmp(info.topology[i],"sigmoid")==0) dev.sigmoid(x[i],x[i+1]);
				if(strcmp(info.topology[i],"tanh")==0) dev.tanh(x[i],x[i+1]);
				if(strcmp(info.topology[i],"softmax")==0) dev.softmax(x[i],x[i+1]);
			}
			
			if(x[info.layer_size-1]->max_idx()==test_data.idx[j]) hit++;
			for(int i=0;i<10;++i){
				x[info.layer_size-1]->err[i]+=(x[info.layer_size-1]->val[i]-label->data[j][i])/(double)(input->batch);
			}
			for(int i=0;i<10;++i){
				mse+=(x[info.layer_size-1]->err[i])*(x[info.layer_size-1]->err[i]);
			}
		}
		for(int i=layer_size-2;i>-1;--i){
			if(strcmp(function_info[i],"affine")==0) dev.affine(x[i],w[i],x[i+1],info.lr);
			if(strcmp(function_info[i],"relu")==0) dev.relu(x[i],x[i+1],info.lr);
			if(strcmp(function_info[i],"sigmoid")==0) dev.sigmoid(x[i],x[i+1],info.lr);
			if(strcmp(function_info[i],"tanh")==0) dev.tanh(x[i],x[i+1],info.lr);
			if(strcmp(function_info[i],"softmax")==0) dev.softmax(x[i],x[i+1],info.lr);
		}
	}

	int infer(INPUT* input,INPUT* output)
	{
		double mse;
		for(int j=0;j<input->batch;++j){
			for(int i=0;i<info.topology[0];++i){ 
				x[0]->val[i]=input->data[j][i];
			}
			
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
	}
};


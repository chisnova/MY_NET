#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "mnist.h"
#include "data.h"
#include "layer.h"

using namespace MY_NET;

mnist train_data;
mnist test_data;


#define layer_size 5
void DNN()
{
	srand((unsigned int)time(NULL));
	readMnist(&train_data,&test_data);

	int tr_size=100;
	int test_size=10;
	double lr=0.01;
	int layer_info[layer_size]={32*32,100,100,10,10};
	char function_info[layer_size-1][100]={"affine","relu","affine","sigmoid"};

	type_1D::type_1D* x[layer_size];
	for(int i=0;i<layer_size;++i) x[i]=new type_1D::type_1D(layer_info[i]);

	type_2D::type_2D* w[layer_size];
	int cnt=0;
	for(int i=0;i<layer_size-1;++i){
		if(strcmp(function_info[i],"affine")==0){
			w[i] = new type_2D::type_2D(layer_info[i+1],layer_info[i]);
		}
	}
	

	int hit;
	int miss;
	double mse;
	layer train;
	while(1){
		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<tr_size;++tr){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=train_data.image[tr][i][j]*0.001;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) train.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) train.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) train.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) train.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0) train.softmax(x[i],x[i+1]);
			}
			
			if(x[layer_size-1]->max_idx()==train_data.idx[tr]) hit++;
			else miss++;
			
			for(int i=0;i<10;++i){x[layer_size-1]->err[i]=x[layer_size-1]->val[i]-train_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(x[layer_size-1]->err[i])*(x[layer_size-1]->err[i]);}

			for(int i=layer_size-2;i>-1;--i){
				if(strcmp(function_info[i],"affine")==0) train.affine(x[i],w[i],x[i+1],lr);
				if(strcmp(function_info[i],"relu")==0) train.relu(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"sigmoid")==0) train.sigmoid(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"tanh")==0) train.tanh(x[i],x[i+1],lr);
				if(strcmp(function_info[i],"softmax")==0) train.softmax(x[i],x[i+1],lr);
			}
		}
		printf("Train Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Train MSE %lf\n",sqrt(mse/double(hit+miss)));

		hit=0;
		miss=0;
		mse=0;
		layer test;
		for(int tr=0;tr<test_size;++tr){
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){ 
				x[0]->val[i*32+j]=test_data.image[tr][i][j]*0.001;
			}}
			
			for(int i=0;i<layer_size-1;++i){
				if(strcmp(function_info[i],"affine")==0) test.affine(x[i],w[i],x[i+1]);
				if(strcmp(function_info[i],"relu")==0) test.relu(x[i],x[i+1]);
				if(strcmp(function_info[i],"sigmoid")==0) test.sigmoid(x[i],x[i+1]);
				if(strcmp(function_info[i],"tanh")==0) test.tanh(x[i],x[i+1]);
				if(strcmp(function_info[i],"softmax")==0) test.softmax(x[i],x[i+1]);
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
	srand((unsigned int)time(NULL));
	readMnist(&train_data,&test_data);
	int tr_size=300;
	int test_size=20;
	double lr=0.001;

    //conv,pool,conv,pool,conv//
	Layer_type_2D* x0 = new Layer_type_2D::Layer_type_2D(1,32,32);
	Layer_type_2D* k0 = new Layer_type_2D::Layer_type_2D(6,5,5);
	Layer_type_2D* x1 = new Layer_type_2D::Layer_type_2D(6,28,28);
	Layer_type_2D* s1 = new Layer_type_2D::Layer_type_2D(6,14,14);
	Layer_type_2D* k1 = new Layer_type_2D::Layer_type_2D(16,5,5);
	Layer_type_2D* x2 = new Layer_type_2D::Layer_type_2D(16,10,10);
	Layer_type_2D* s2 = new Layer_type_2D::Layer_type_2D(16,5,5);
	Layer_type_2D* k2 = new Layer_type_2D::Layer_type_2D(120,5,5);
	Layer_type_2D* x3 = new Layer_type_2D::Layer_type_2D(120,1,1);

	type_1D* dx1 = new type_1D::type_1D(120);
	type_2D* dw1 = new type_2D::type_2D(84,120);
	type_1D* dx2 = new type_1D::type_1D(84);
	type_1D* da2 = new type_1D::type_1D(84);
	type_2D* dw2 = new type_2D::type_2D(10,84);
	type_1D* dx3 = new type_1D::type_1D(10);
	type_1D* da3 = new type_1D::type_1D(10);

	int hit;
	int miss;
	double mse;
	layer train;
	while(1)
	{
		hit=0;
		miss=0;
		mse=0;
		for(int tr=0;tr<tr_size;++tr){
			//FORWARD
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){
				x0->image[0]->val[i][j]=train_data.image[tr][i][j]*0.001;
			}}

			train.conv(x0,k0,x1);	
			train.average_pooling(x1,s1,2);	
			train.conv(s1,k1,x2);
			train.average_pooling(x2,s2,2);
			train.conv(s2,k2,x3);
			for(int i=0;i<120;++i)	dx1->val[i]=x3->image[i]->val[0][0];
			train.affine(dx1,dw1,dx2);
			train.sigmoid(dx2,da2);
			train.affine(da2,dw2,dx3);
			train.sigmoid(dx3,da3);

			if(da3->max_idx()==train_data.idx[tr]) hit++;
			else miss++;
			for(int i=0;i<10;++i){da3->err[i]=da3->val[i]-train_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(da3->err[i])*(da3->err[i]);}

			//BACKWARD
			train.sigmoid(dx3,da3,lr);
			train.affine(da2,dw2,dx3,lr);
			train.sigmoid(dx2,da2,lr);
			train.affine(dx1,dw1,dx2,lr);
			for(int i=0;i<120;++i) x3->image[i]->err[0][0]=dx1->err[i];
			train.conv(s2,k2,x3,lr);
			train.average_pooling(x2,s2,2,lr);
			train.conv(s1,k1,x2,lr);
			train.average_pooling(x1,s1,2,lr);	
			train.conv(x0,k0,x1,lr);	
		}
		printf("Train Recognition Rate %lf\n",double(hit)/(double(hit)+double(miss))*100);
		printf("Train MSE %lf\n",sqrt(mse/double(hit+miss)));

		hit=0;
		miss=0;
		mse=0;
		layer test;
		for(int tr=0;tr<test_size;++tr){
			//FORWARD
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){
				x0->image[0]->val[i][j]=test_data.image[tr][i][j]*0.001;
			}}
			
			test.conv(x0,k0,x1);	
			test.average_pooling(x1,s1,2);	
			test.conv(s1,k1,x2);
			test.average_pooling(x2,s2,2);
			test.conv(s2,k2,x3);
			for(int i=0;i<120;++i)	dx1->val[i]=x3->image[i]->val[0][0];
			test.affine(dx1,dw1,dx2);
			test.sigmoid(dx2,da2);
			test.affine(da2,dw2,dx3);
			test.sigmoid(dx3,da3);

			if(da3->max_idx()==test_data.idx[tr]) hit++;
			else miss++;

			for(int i=0;i<10;++i){da3->err[i]=da3->val[i]-test_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(da3->err[i])*(da3->err[i]);}

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

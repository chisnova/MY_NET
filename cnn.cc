#include "cnn.h"

using namespace MY_NET;
void LeNet()
{
	srand((unsigned int)time(NULL));
	readMnist(&train_data,&test_data);
	int tr_iter=500,test_size=100,batch_size=100;
	double lr=0.001;

	int hit,miss;
	layer train;
	double*** batch_tr;
	double** batch_label;
	int* batch_idx;
	double mse,mean=0,var=0,stv=0,Recog,MSE,err;
	int idx;

    //conv,pool,conv,pool,conv//
	Layer_type_2D* x0 = new Layer_type_2D(1,32,32);
	Layer_type_2D* k0 = new Layer_type_2D(6,5,5);
	Layer_type_2D* x1 = new Layer_type_2D(6,28,28);
	Layer_type_2D* a1 = new Layer_type_2D(6,28,28);
	Layer_type_2D* s1 = new Layer_type_2D(6,14,14);
	Layer_type_2D* k1 = new Layer_type_2D(16,5,5);
	Layer_type_2D* x2 = new Layer_type_2D(16,10,10);
	Layer_type_2D* a2 = new Layer_type_2D(16,10,10);
	Layer_type_2D* s2 = new Layer_type_2D(16,5,5);
	Layer_type_2D* k2 = new Layer_type_2D(120,5,5);
	Layer_type_2D* x3 = new Layer_type_2D(120,1,1);
	Layer_type_2D* a3 = new Layer_type_2D(120,1,1);

	type_1D* dx1 = new type_1D(120);
	type_2D* dw1 = new type_2D(84,120);
	type_1D* dx2 = new type_1D(84);
	type_1D* da2 = new type_1D(84);
	type_2D* dw2 = new type_2D(10,84);
	type_1D* dx3 = new type_1D(10);
	type_1D* da3 = new type_1D(10);

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
	while(iteration<tr_iter)
	{
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
			for(int i=0;i<32;++i){
				for(int j=0;j<32;++j){ 
					x0->image[0]->val[i][j]=(batch_tr[batch][i][j]-mean)/stv;
				}
			}

			train.conv(x0,k0,x1);	
			train.relu(x1,a1);	
			train.average_pooling(a1,s1,2);	
			train.conv(s1,k1,x2);
			train.relu(x2,a2);	
			train.average_pooling(a2,s2,2);
			train.conv(s2,k2,x3);
			train.relu(x3,a3);
			for(int i=0;i<120;++i)	dx1->val[i]=a3->image[i]->val[0][0];
			train.affine(dx1,dw1,dx2);
			train.relu(dx2,da2);
			train.affine(da2,dw2,dx3);
			train.sigmoid(dx3,da3);

			if(da3->max_idx()==batch_idx[batch]) hit++;
			else miss++;
			for(int i=0;i<10;++i){
				err=da3->val[i]-batch_label[batch][i];
				//da3->err[i]+=err/double(batch_size);
				da3->err[i]=err;
				mse+=err*err;
			}
			train.sigmoid(dx3,da3,lr);
			train.affine(da2,dw2,dx3,lr);
			train.relu(dx2,da2,lr);
			train.affine(dx1,dw1,dx2,lr);
			for(int i=0;i<120;++i) a3->image[i]->err[0][0]=dx1->err[i];
			train.relu(x3,a3,lr);
			train.conv(s2,k2,x3,lr);
			train.average_pooling(a2,s2,2,lr);
			train.relu(x2,a2,lr);	
			train.conv(s1,k1,x2,lr);
			train.average_pooling(a1,s1,2,lr);	
			train.relu(x1,a1,lr);	
			train.conv(x0,k0,x1,lr);	
		}

		Recog=double(hit)/(double(hit)+double(miss))*100;
		MSE=sqrt(mse/double(hit+miss));
		printf("Iteration %d Train Recognition Rate %lf Train MSE %lf\n",iteration,Recog,MSE);

		hit=0,miss=0,mse=0;
		layer test;
		for(int tr=0;tr<test_size;++tr){
			//FORWARD
			for(int i=0;i<32;++i){for(int j=0;j<32;++j){
				x0->image[0]->val[i][j]=(test_data.image[tr][i][j]-mean)/stv;
			}}
			
			train.conv(x0,k0,x1);	
			train.relu(x1,a1);	
			train.average_pooling(a1,s1,2);	
			train.conv(s1,k1,x2);
			train.relu(x2,a2);	
			train.average_pooling(a2,s2,2);
			train.conv(s2,k2,x3);
			train.relu(x3,a3);
			for(int i=0;i<120;++i)	dx1->val[i]=a3->image[i]->val[0][0];
			train.affine(dx1,dw1,dx2);
			train.relu(dx2,da2);
			train.affine(da2,dw2,dx3);
			train.sigmoid(dx3,da3);

			if(da3->max_idx()==test_data.idx[tr]) hit++;
			else miss++;

			for(int i=0;i<10;++i){da3->err[i]=da3->val[i]-test_data.label[tr][i];}
			for(int i=0;i<10;++i){mse+=(da3->err[i])*(da3->err[i]);}

		}

		Recog=double(hit)/(double(hit)+double(miss))*100;
		MSE=sqrt(mse/double(hit+miss));
		printf("Iteration %d Test Recognition Rate %lf Test MSE %lf\n",iteration,Recog,MSE);
	}
}


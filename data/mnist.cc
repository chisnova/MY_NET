#include "mnist.h"

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



#ifndef _MINST_
#define _MINST_
#include <stdlib.h>
typedef struct _MNIST_
{
	double image[60000][32][32];
	double label[60000][10];
	int idx[60000];
}mnist;

int readMnist(mnist* train,mnist* test);

#endif

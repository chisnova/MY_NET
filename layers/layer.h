#ifndef _layer_h_
#define _layer_h_
#include <math.h>
#include "data.h"

namespace MY_NET
{
	class layer
	{
		private:
			bool isRange(int w,int w_max,int d,int d_max);
			void MatMul(type_1D* in, type_2D* w,type_1D* out);
			void MatMul(type_1D* in, type_2D* w,type_1D* out,double lr);
		public:
			void sigmoid(type_1D* in,type_1D* out);
			void sigmoid(type_1D* in,type_1D* out,double lr);
			void relu(type_1D* in,type_1D* out);
			void relu(type_1D* in,type_1D* out,double lr);
			void tanh(type_1D* in,type_1D* out);
			void tanh(type_1D* in,type_1D* out,double lr);
			void softmax(type_1D* in,type_1D* out);
			void softmax(type_1D* in,type_1D* out,double lr);
			void affine(type_1D* in,type_2D* w,type_1D* out);
			void affine(type_1D* in,type_2D* w,type_1D* out,double lr);
			//La er 2D
			void sigmoid(Layer_type_2D* in,Layer_type_2D* out);
			void sigmoid(Layer_type_2D* in,Layer_type_2D* out,double lr);
			void relu(Layer_type_2D* in,Layer_type_2D* out);
			void relu(Layer_type_2D* in,Layer_type_2D* out,double lr);
			void tanh(Layer_type_2D* in,Layer_type_2D* out);
			void tanh(Layer_type_2D* in,Layer_type_2D* out,double lr);
			void softmax(Layer_type_2D* in,Layer_type_2D* out);
			void softmax(Layer_type_2D* in,Layer_type_2D* out,double lr);
			void conv(Layer_type_2D* in,Layer_type_2D* kernel,Layer_type_2D* out);
			void conv(Layer_type_2D* in,Layer_type_2D* kernel,Layer_type_2D* out,double lr);
			void max_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize);
			void max_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize,double lr);
			void average_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize);
			void average_pooling(Layer_type_2D* in,Layer_type_2D* out,int poolsize,double lr);
	};
}

#endif

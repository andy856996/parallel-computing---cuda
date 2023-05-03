#include <stdio.h>
#include <math.h>
int main() {
	float x_data[10] = { 338, 333, 328, 207,
	226, 25, 179, 60, 208, 606 };
	float y_data[10] = { 640, 633, 619, 393,
	428, 27, 193, 66, 226, 1591 };
	float b;
	float w;
	b = -120;
	w = -4;
	int lr = 1;
	int iteration = 500;
	float lr_b = 0;
	float lr_w = 0;
	for (int i = 0; i < iteration; i++) {

		float b_grad = 0;
		float w_grad = 0;
		for (int n = 0; n < 10; n++) {
			//L(w,b)對b偏微分
			b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0;
			//L(w,b)對w偏微分
			w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n];
		}
		//AdaGrad
		lr_b = lr_b + b_grad * b_grad;
		lr_w = lr_w + w_grad * w_grad;
		b = b - lr / sqrt(lr_b) * b_grad;
		w = w - lr / sqrt(lr_w) * w_grad;
	}
	printf("=====host=====\n");
	printf("w=%f\n", w);
	printf("b=%f\n", b);
	printf("==============\n");
	return 0;
}
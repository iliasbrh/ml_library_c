#include <stdio.h>
#include <stdlib.h>
#include "arena.h"
#include "value.h"
#include "matrix.h"
#include "read_ubyte.h"

int main(void) {
		mem_arena* perm_arena = create_arena(GiB(3));


		u64 train_sample_length = 50000; // <= 50000
		u64 test_sample_length = 10000; // <= 10000
		
		FILE* train_img_file = fopen("train-images.idx3-ubyte", "rb");
		img_dataset train_img = load_img_dataset(perm_arena, train_img_file, train_sample_length);
		
		FILE* train_lbl_file = fopen("train-labels.idx1-ubyte", "rb");
		lbl_dataset train_lbl = load_lbl_dataset(perm_arena, train_lbl_file, train_sample_length);


		FILE* test_img_file = fopen("test-images.idx3-ubyte", "rb");
		img_dataset test_img = load_img_dataset(perm_arena, test_img_file, test_sample_length);
		
		FILE* test_lbl_file = fopen("test-labels.idx1-ubyte", "rb");
		lbl_dataset test_lbl = load_lbl_dataset(perm_arena, test_lbl_file, test_sample_length);


		matrix* linear1 = random(perm_arena, 16, 784, -0.1, 0.1);
		matrix* linear2 = random(perm_arena, 16, 16, -0.1, 0.1);
		matrix* linear3 = random(perm_arena, 10, 16, -0.1, 0.1);
		
		u64 pos_after_params = perm_arena->pos;	

		f32 learning_rate = 0.6;
		u64 epochs = 5;
		
		for (u64 epoch=0; epoch<epochs; ++epoch) {
				f32 running_loss = 0.;

				// training_loop
				for (u64 i=0; i<train_sample_length; ++i) {

						// zero out the grads of the dataset and of the parameters
						zero_grad(train_img[i]);
						zero_grad(linear1);
						zero_grad(linear2);
						zero_grad(linear3);

						// forward pass
						matrix* x1 = mat_multiply(perm_arena, linear1, train_img[i]);
						matrix* x1_rel = mat_relu(perm_arena, x1);
						matrix* x2 = mat_multiply(perm_arena, linear2, x1_rel);
						matrix* x2_rel = mat_relu(perm_arena, x2);
						matrix* x3 = mat_multiply(perm_arena, linear3, x2_rel);

						matrix* lbl = one_hot(perm_arena, 10, train_lbl[i]);

						value* loss = mse_loss(perm_arena, x3, lbl);
						running_loss += loss->data;
						
						// back propagation
						backward(loss);
						
						// step
						for (u64 j=0; j<linear1->rows * linear1->columns; ++j)
								linear1->values[j]->data -= learning_rate * linear1->values[j]->grad;
						for (u64 j=0; j<linear2->rows * linear2->columns; ++j)
								linear2->values[j]->data -= learning_rate * linear2->values[j]->grad;
						for (u64 j=0; j<linear3->rows * linear3->columns; ++j)
								linear3->values[j]->data -= learning_rate * linear3->values[j]->grad;

						arena_pop_to(perm_arena, pos_after_params);
				}
				printf("Loss after %i epochs : %f\n", epoch+1, running_loss/train_sample_length);

				// testing loop
				u64 correct = 0;
				for (u64 i=0; i<test_sample_length; ++i) {
						matrix* x1 = mat_multiply(perm_arena, linear1, test_img[i]);
						matrix* x1_rel = mat_relu(perm_arena, x1);
						matrix* x2 = mat_multiply(perm_arena, linear2, x1_rel);
						matrix* x2_rel = mat_relu(perm_arena, x2);
						matrix* x3 = mat_multiply(perm_arena, linear3, x2_rel);

						u8 argmax = 0;
						for (u8 i=1; i<10; ++i) {
								if(x3->values[i]->data > x3->values[argmax]->data) argmax = i;
						}
						if (argmax == test_lbl[i]) correct++;

						arena_pop_to(perm_arena, pos_after_params);
				}
				f32 accuracy = (f32)correct / (f32)test_sample_length;
				printf("Accuracy : %.2f %\n", accuracy*100.);
		}


		destroy_arena(perm_arena);

		return 0;
}

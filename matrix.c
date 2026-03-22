#include <stdio.h>
#include "value.h"
#include "matrix.h"

matrix* create_matrix(mem_arena* arena, u64 rows, u64 columns) {
		matrix* out = (matrix*)arena_push(arena, sizeof(matrix), false);
		out->rows = rows;
		out->columns = columns;

		out->values = (value**)arena_push(arena, sizeof(value*)*rows*columns, false);

		return out;
}

matrix* full(mem_arena* arena, f32 val, u64 rows, u64 columns) {
		matrix* out = (matrix*)create_matrix(arena, rows, columns);

		for (u64 i=0; i<rows; ++i)
				for (u64 j=0; j<columns; ++j)
						out->values[i*columns+j] = create_value(arena, val);

		return out;
}

matrix* zeros(mem_arena* arena, u64 rows, u64 columns) {
		return full(arena, 0., rows, columns);
}

matrix* ones(mem_arena* arena, u64 rows, u64 columns) {
		return full(arena, 1., rows, columns);
}

matrix* random(mem_arena* arena, u64 rows, u64 columns, f32 min_value, f32 max_value) {
		matrix* out = (matrix*)create_matrix(arena, rows, columns);
		f32 range = max_value - min_value;

		for (u64 i=0; i<rows; ++i)
				for (u64 j=0; j<columns; ++j)
						out->values[i*columns+j] = create_value(arena, min_value + (float)rand()/((float)(RAND_MAX)/range));

		return out;
}

matrix* one_hot(mem_arena* arena, u64 n_classes, u64 idx) {
		// column vector
		matrix* out = zeros(arena, n_classes, 1);
		out->values[idx]->data = 1.;
		
		return out;
}

matrix* mat_add(mem_arena* arena, matrix* mat1, matrix* mat2) {
		// do security checks later
		u64 rows = mat1->rows;
		u64 columns = mat1->columns;
		matrix* out = (matrix*)create_matrix(arena, rows, columns);

		for (u64 i=0; i<rows; ++i)
				for (u64 j=0; j<columns; ++j)
						out->values[i*columns+j] = val_add(arena, mat1->values[i*columns+j], mat2->values[i*columns+j]);

		return out;
}

matrix* mat_multiply(mem_arena* arena, matrix* mat1, matrix* mat2) {
		u64 rows = mat1->rows;
		u64 columns = mat2->columns;
		matrix* out = (matrix*)create_matrix(arena, rows, columns);

		for (u64 i=0; i<rows; ++i)
				for (u64 j=0; j<columns; ++j) {
						value* sum = create_value(arena, 0.); 
						for (u64 k=0; k<mat1->columns; ++k) {
								value* prod = val_multiply(arena, mat1->values[i*mat1->columns+k], mat2->values[k*mat2->columns+j]);
								sum = val_add(arena, sum, prod);
						}
						out->values[i*columns+j] = sum;
				}

		return out;
}

matrix* mat_operation(mem_arena* arena, matrix* mat, value* (*operation)(mem_arena*, value*)) {
		u64 rows = mat->rows;
		u64 columns = mat->columns;
		matrix* out = (matrix*)create_matrix(arena, rows, columns);

		for (u64 i=0; i<rows; ++i)
				for (u64 j=0; j<columns; ++j)
						out->values[i*columns+j] = operation(arena, mat->values[i*columns+j]);

		return out;
		
}
matrix* mat_neg(mem_arena* arena, matrix* mat) {
		return mat_operation(arena, mat, val_neg);
}

matrix* mat_invert(mem_arena* arena, matrix* mat) {
		return mat_operation(arena, mat, val_invert);
}

matrix* mat_exp(mem_arena* arena, matrix* mat) {
		return mat_operation(arena, mat, val_exp);
}

matrix* mat_log(mem_arena* arena, matrix* mat) {
		return mat_operation(arena, mat, val_log);
}

matrix* mat_relu(mem_arena* arena, matrix* mat) {
		return mat_operation(arena, mat, val_relu);
}

matrix* softmax(mem_arena* arena, matrix* mat) {
		// mat has to be a column vector
		f32 maximum = 0.;
		for (u64 i=0; i<mat->rows; ++i) maximum = MAX(maximum, mat->values[i]->data);

		matrix* to_subtract = full(arena, maximum, mat->rows, mat->columns);
		matrix* subtracted = mat_add(arena, mat, mat_neg(arena, to_subtract));
		matrix* exponentials = mat_exp(arena, subtracted);

		value* sum = create_value(arena, 0);
		for (u64 i=0; i<mat->rows; ++i) sum = val_add(arena, sum, exponentials->values[i]);
		value* divider = val_invert(arena, sum);
		matrix* divider_mat = zeros(arena, 1, 1);
		divider_mat->values[0] = divider;

		matrix* out = mat_multiply(arena, exponentials, divider_mat);

		return out;
}

value* mse_loss(mem_arena* arena, matrix* mat1, matrix* mat2) {
		// both have to be column vectors
		// only mat1 goes through softmax, mat2 is supposed to be the one hot label
		matrix* softmax1 = softmax(arena, mat1);
		matrix* neg2 = mat_neg(arena, mat2);
		matrix* difference = mat_add(arena, softmax1, neg2);

		value* sum_squares = create_value(arena, 0);
		for (u64 i=0; i<difference->rows; ++i) 
				sum_squares = val_add(arena, sum_squares, val_multiply(arena, difference->values[i], difference->values[i]));

		value* divider = val_invert(arena, create_value(arena, (f32)mat1->rows));
		value* out = val_multiply(arena, sum_squares, divider);

		return out;
}

void zero_grad(matrix* mat) {
		for (u64 row=0; row<mat->rows; row++)
				for (u64 col=0; col<mat->columns; col++)
						mat->values[row*mat->columns + col]->grad = 0.;
}

void print_matrix(matrix* mat) {
		for (u64 row=0; row<mat->rows; row++) {
				for (u64 col=0; col<mat->columns; col++)
						printf("%.2f ", mat->values[row*mat->columns+col]->data);
				printf("\n");
}
		printf("\n\n");
}


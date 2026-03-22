#ifndef MATRIX_H
#define MATRIX_H

#include "value.h"

typedef struct {
		u64 rows;
		u64 columns;
		value** values;
} matrix;

matrix* create_matrix(mem_arena* arena, u64 rows, u64 columns);

matrix* full(mem_arena* arena, f32 val, u64 rows, u64 columns);
matrix* zeros(mem_arena* arena, u64 rows, u64 columns);
matrix* ones(mem_arena* arena, u64 rows, u64 columns);
matrix* random(mem_arena* arena, u64 rows, u64 columns, f32 min_value, f32 max_value);
matrix* one_hot(mem_arena* arena, u64 n_classes, u64 idx);

matrix* mat_add(mem_arena* arena, matrix* mat1, matrix* mat2);
matrix* mat_multiply(mem_arena* arena, matrix* mat1, matrix* mat2);
matrix* mat_neg(mem_arena* arena, matrix* mat);
matrix* mat_invert(mem_arena* arena, matrix* mat);
matrix* mat_exp(mem_arena* arena, matrix* mat);
matrix* mat_log(mem_arena* arena, matrix* mat);
matrix* mat_relu(mem_arena* arena, matrix* mat);

matrix* softmax(mem_arena* arena, matrix* mat);
value* mse_loss(mem_arena* arena, matrix* mat1, matrix* mat2);

void zero_grad(matrix* mat);

void print_matrix(matrix* mat);

#endif

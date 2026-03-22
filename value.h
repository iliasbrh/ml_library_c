#ifndef VALUE_H
#define VALUE_H

#include "arena.h"

typedef struct value value;
struct value {
		f32 data;
		f32 grad;
		b32 visited;

		value* child1;
		value* child2;
		f32 local_grad1;
		f32 local_grad2;
};

// among all existing operations, there's a maximum of 2 children (for addition and multiplication) so the amount of children will never be bigger than 2

value* create_value(mem_arena* arena, f32 data);
value* val_add(mem_arena* arena, value* v1, value* v2);
value* val_multiply(mem_arena* arena, value* v1, value* v2);
value* val_neg(mem_arena* arena, value* v);
value* val_invert(mem_arena* arena, value* v);
value* val_exp(mem_arena* arena, value* v);
value* val_log(mem_arena* arena, value* v);
value* val_relu(mem_arena* arena, value* v);


typedef struct {
		u64 capacity;
		u64 count;
		value** data;
} dynamic_array;

void array_push(dynamic_array* array, value* val);
void array_pop(dynamic_array* array);
void build_topo(dynamic_array* topo, value* node);
void backward(value* val);

#endif

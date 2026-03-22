#include <math.h>
#include "arena.h"
#include "value.h"

value* create_value(mem_arena* arena, f32 data) {
		value* out = (value*)arena_push(arena, sizeof(value), false);
		out->data = data;
		
		return out;
}

// arena will be a parameter for each operation so we don't have to allocate a result manually before computing the operation
value* val_add(mem_arena* arena, value* v1, value* v2) {
		value* out = (value*)create_value(arena, v1->data + v2->data);
		out->child1 = v1;
		out->child2 = v2;
		out->local_grad1 = 1.;
		out->local_grad2 = 1.;

		return out;
}

value* val_multiply(mem_arena* arena, value* v1, value* v2) {
		value* out = (value*)create_value(arena, v1->data * v2->data);
		out->child1 = v1;
		out->child2 = v2;
		out->local_grad1 = v2->data;
		out->local_grad2 = v1->data;

		return out;
}

value* val_neg(mem_arena* arena, value* v) {
		value* out = (value*)create_value(arena, -v->data);
		out->child1 = v;
		out->local_grad1 = -1.;
		out->child2 = NULL;
		out->local_grad2 = 0.;

		return out;
}

value* val_invert(mem_arena* arena, value* v) {
		value* out = (value*)create_value(arena, powf(v->data, -1.));
		out->child1 = v;
		out->local_grad1 = -powf(v->data, -2.);
		out->child2 = NULL;
		out->local_grad2 = 0.;

		return out;
}

value* val_exp(mem_arena* arena, value* v) {
		value* out = (value*)create_value(arena, expf(v->data));
		out->child1 = v;
		out->local_grad1 = expf(v->data);
		out->child2 = NULL;
		out->local_grad2 = 0.;

		return out;
}

value* val_log(mem_arena* arena, value* v) {
		value* out = (value*)create_value(arena, logf(v->data));
		out->child1 = v;
		out->local_grad1 = powf(v->data, -1.);
		out->child2 = NULL;
		out->local_grad2 = 0.;
		
		return out;
}

value* val_relu(mem_arena* arena, value* v) {
		value* out = (value*)create_value(arena, 0);
		if (v->data > 0) { 
				out->data = v->data;
				out->child1 = v;
				out->local_grad1 = 1.;
				out->child2 = NULL;
				out->local_grad2 = 0.;
		}
		
		return out;
}

// DFS for backpropagation in the operational graph, using a dynamic_array

void array_push(dynamic_array* array, value* val) {
		if (array->capacity <= array->count) {
				array->capacity *= 2;
				array->data = (value**)realloc(array->data, sizeof(value*)*array->capacity);
		}
		array->data[array->count++] = val;
}

void array_pop(dynamic_array* array) {
		array->count--;
}

void build_topo(dynamic_array* topo, value* node) {
		if (!node->visited) {
				if (node->child1) build_topo(topo, node->child1);
				if (node->child2) build_topo(topo, node->child2);
				array_push(topo, node);
				node->visited = true;
		}
}

void backward(value* val) {
		// the topology is not allocated in the arena since the realloc would be too hard to handle
		dynamic_array* topology = (dynamic_array*)malloc(sizeof(dynamic_array));
		topology->capacity = 256;
		topology->count = 0;
		topology->data = (value**)malloc(sizeof(value*)*topology->capacity);

		build_topo(topology, val);
		
		val->grad = 1.;
		for (i64 i=topology->count - 1; i>=0; --i) {
				topology->data[i]->visited = false;
				if(topology->data[i]->child1) topology->data[i]->child1->grad += topology->data[i]->grad * topology->data[i]->local_grad1;
				if(topology->data[i]->child2) topology->data[i]->child2->grad += topology->data[i]->grad * topology->data[i]->local_grad2;
		}

		free(topology->data);
		free(topology);
}


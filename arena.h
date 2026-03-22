#ifndef ARENA_H
#define ARENA_H 

#include "types.h"

typedef struct {
		u64 capacity;
		u64 pos;
} mem_arena;

#define ARENA_BASE_POS (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void*))
#define ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & ~((u64)(p) - 1))

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

mem_arena* create_arena(u64 capacity);
void destroy_arena(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size, b32 non_zero);
void arena_pop(mem_arena* arena, u64 size);
void arena_pop_to(mem_arena* arena, u64 pos);
void arena_clear(mem_arena* arena);

#endif


#include <string.h>
#include <assert.h>
#include "arena.h"

mem_arena* create_arena(u64 capacity) {
		mem_arena* arena = (mem_arena*)malloc(capacity);

		arena->pos = ARENA_BASE_POS;
		arena->capacity = capacity;

		return arena;
}

void destroy_arena(mem_arena* arena) {
		free(arena);
}

void* arena_push(mem_arena* arena, u64 size, b32 non_zero) {
		u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
		u64 new_pos = pos_aligned + size;

		assert(new_pos <= arena->capacity && "Arena out of memory");
		
		arena->pos = new_pos;

		u8* out = (u8*)arena + pos_aligned;

		if (!non_zero) {
				memset(out, 0, size);
		}

		return out;
}

void arena_pop(mem_arena* arena, u64 size) {
		size = MIN(size, arena->pos - ARENA_BASE_POS);
		arena->pos -= size;
}

void arena_pop_to(mem_arena* arena, u64 pos) {
		u64 size = pos < arena->pos ? arena->pos - pos : 0;
		arena_pop(arena, size);
}

void arena_clear(mem_arena* arena) {
		arena_pop_to(arena, ARENA_BASE_POS);
}

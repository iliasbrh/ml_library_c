#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "read_ubyte.h"

matrix* read_ubyte_image(mem_arena* arena, FILE* input, u64 rows, u64 columns) {
		matrix* out = zeros(arena, rows, columns);
		
		u8 buffer;
		for (u64 row=0; row<rows; row++)
				for (u64 col=0; col<columns; col++) {
						fread(&buffer, 1, 1, input);
						out->values[row*columns+col]->data = (f32)(buffer) / 255.;
				}

		return out;
}

u8 read_ubyte_label(mem_arena* arena, FILE* input) {
		u8 buffer;
		fread(&buffer, 1, 1, input);

		return buffer;
}

		
img_dataset load_img_dataset(mem_arena* arena, FILE* input, u64 size) {
		u8 buffer;
		for (u8 i=0; i<16; ++i) fread(&buffer, 1, 1, input); // reading header

		img_dataset out = arena_push(arena, size*sizeof(matrix*), false);
		for (u64 i=0; i<size; ++i) out[i] = read_ubyte_image(arena, input, 784, 1);

		return out;
}

lbl_dataset load_lbl_dataset(mem_arena* arena, FILE* input, u64 size) {
		u8 buffer;
		for (u8 i=0; i<8; ++i) fread(&buffer, 1, 1, input); // reading header

		lbl_dataset out = arena_push(arena, size*sizeof(u8), false);
		for (u64 i=0; i<size; ++i) out[i] = read_ubyte_label(arena, input);

		return out;
}


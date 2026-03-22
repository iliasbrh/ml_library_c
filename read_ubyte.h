#ifndef READ_UBYTE_H
#define READ_UBYTE_H

#include "matrix.h"

matrix* read_ubyte_image(mem_arena* arena, FILE* input, u64 rows, u64 columns);
u8 read_ubyte_label(mem_arena* arena, FILE* input);

typedef matrix** img_dataset;
typedef u8* lbl_dataset;
img_dataset load_img_dataset(mem_arena* arena, FILE* input, u64 size);
lbl_dataset load_lbl_dataset(mem_arena* arena, FILE* input, u64 size);

#endif

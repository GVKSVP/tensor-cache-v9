/*
 * tensor_map_hook.h - force-included (gcc -include) into the "map" builds of
 * the AI kernels. Records every aligned_alloc (address, size, order) and
 * prints the list to stderr at exit, i.e. after the ROI. The kernel sources
 * and ai_common.h are not changed. Used only to label trace addresses with
 * tensor names; gem5 runs for results use the original binaries.
 */
#include <stdio.h>
#include <stdlib.h>

#define TM_MAX 128
static void *tm_ptr[TM_MAX];
static size_t tm_size[TM_MAX];
static int tm_n;

static void tm_dump(void)
{
    for (int i = 0; i < tm_n; i++)
        fprintf(stderr, "TENSORMAP %d 0x%lx %zu\n", i, (unsigned long)tm_ptr[i], tm_size[i]);
}

static void *tm_aligned_alloc(size_t align, size_t size)
{
    void *p = aligned_alloc(align, size);
    if (tm_n == 0)
        atexit(tm_dump);
    if (tm_n < TM_MAX) {
        tm_ptr[tm_n] = p;
        tm_size[tm_n] = size;
        tm_n++;
    }
    return p;
}

#define aligned_alloc tm_aligned_alloc

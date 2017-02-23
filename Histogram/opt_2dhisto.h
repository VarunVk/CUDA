#ifndef OPT_KERNEL
#define OPT_KERNEL

#define BLOCK_SIZE 512
//#define BLOCK_SIZE 1024
//#define BLOCK_SIZE 256
//#define BLOCK_SIZE 64
// checked and found performance better for 512

//CHECK_BELOW for GTX480
#define GRID_SIZE_MAX 65535

#endif

/* Include below the function headers of any other functions that you implement */
void opt_2dhisto(uint32_t *input, uint32_t *device_bins);
//( /*Define your own function parameters*/ );

uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int size_of_element);

void free_cuda(uint32_t *ptr);

void CopyToDevice(uint32_t *device, uint32_t *host, uint32_t input_height, uint32_t input_width, int size_of_element);

void CopyToHost(uint32_t *host, uint32_t *device, int size);

void cuda_memset(uint32_t *ptr, uint32_t value, uint32_t byte_count);


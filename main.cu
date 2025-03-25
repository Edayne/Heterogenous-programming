#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// Gaussian function
double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Manual bilateral filter
__global__ void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Precompute spatial Gaussian weights
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    // Process image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    double weight_sum[3] = {0.0, 0.0, 0.0};
    double filtered_value[3] = {0.0, 0.0, 0.0};

    // Get center pixel pointer
    unsigned char *center_pixel = src + (y * width + x) * channels;

    // Iterate over local window
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int nx = x + j - radius;
            int ny = y + i - radius;

            // Bounds check to ensure we're within the image
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }

            // Get neighbor pixel pointer
            unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

            for (int c = 0; c < channels; c++) {
                // Compute range weight
                double range_weight = gaussian(abs(neighbor_pixel[c] - center_pixel[c]), sigma_color);
                double weight = spatial_weights[i * d + j] * range_weight;

                // Accumulate weighted sum
                filtered_value[c] += neighbor_pixel[c] * weight;
                weight_sum[c] += weight;
            }
        }
    }

    // Normalize and store result
    unsigned char *output_pixel = dst + (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6)); // Avoid division by zero
    }

    free(spatial_weights);
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    // Ensure that image is not too small for bilateral filter (at least radius of d/2 around edges)
    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    // Allocate memory for output image
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }
    
    unsigned char *d_src, *d_dst;
    cudaMalloc((void**)&d_src, width * height * channels);
    cudaMalloc((void**)&d_dst, width * height * channels);

    cudaMemcpy(d_dst, image, width*height*channels, cudaMemcpyHostToDevice);

    // Apply the bilateral filter
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1)/blockSize.x, (height + blockSize.y - 1)/blockSize.y);
    bilateral_filter<<<gridSize, blockSize>>>(image, filtered_image, width, height, channels, 5, 75.0, 75.0);


    cudaMemcpy(filtered_image, d_dst, width*height*channels, cudaMemcpyDeviceToHost);
    
    // Save the output image
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    // Free memory
    stbi_image_free(image);
    free(filtered_image);
    cudaFree(d_src);
    cudaFree(d_dst);

    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    return 0;
}

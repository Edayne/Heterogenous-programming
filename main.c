#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

// Structure to hold image data
typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;


// Function to read an image
Image read_image(const char *filename) {
    Image img;
    img.data = stbi_load(filename, &img.width, &img.height, &img.channels, 0);
    
    if (!img.data) {
        printf("Failed to load image: %s\n", filename);
    } else {
        printf("Image loaded: %s (Width: %d, Height: %d, Channels: %d)\n",
               filename, img.width, img.height, img.channels);
    }

    return img;
}

// Function to save an image
int save_image(const char *filename, Image img) {
    if (!img.data) {
        printf("No image data to save!\n");
        return 0;
    }

    int success = stbi_write_png(filename, img.width, img.height, img.channels, img.data, img.width * img.channels);
    
    if (success) {
        printf("Image saved: %s\n", filename);
    } else {
        printf("Failed to save image: %s\n", filename);
    }

    return success;
}


int main() {
    // Read image
    Image img = read_image("lena.jpg");
    
    if (img.data) {
        // Save image
        save_image("output.jpg", img);

        // Free memory
        stbi_image_free(img.data);
    }

    return 0;
}


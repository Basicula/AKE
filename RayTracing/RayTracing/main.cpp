 #include <stdio.h>
 #include <iostream>
 
 #include "BMPWriter.h"
 
 int main()
 {
   for(int i = 0; i <4;++i)
     std::cout << int((unsigned char)(257>>(i*8))) << std::endl;
 
   BMPWriter writer(128,128);
   writer.Write("G:\\test.bmp");
 
   system("pause");
   return 0;
 }

//  #include <stdio.h>
//  
//  const int bytesPerPixel = 3; /// red, green, blue
//  const int fileHeaderSize = 14;
//  const int infoHeaderSize = 40;
//  
//  void generateBitmapImage(unsigned char *image, int height, int width, const char* imageFileName);
//  unsigned char* createBitmapFileHeader(int height, int width, int paddingSize);
//  unsigned char* createBitmapInfoHeader(int height, int width);
//  
//  
//  int main() {
//    const int height = 341;
//    const int width = 753;
//    unsigned char image[height][width][bytesPerPixel];
//    const char* imageFileName = "G:\\test.bmp";
//  
//    int i, j;
//    for (i = 0; i < height; i++) {
//      for (j = 0; j < width; j++) {
//        image[i][j][2] = (unsigned char)((double)i / height * 255); ///red
//        image[i][j][1] = (unsigned char)((double)j / width * 255); ///green
//        image[i][j][0] = (unsigned char)(((double)i + j) / (height + width) * 255); ///blue
//      }
//    }
//  
//    generateBitmapImage((unsigned char *)image, height, width, imageFileName);
//    printf("Image generated!!");
//  }
//  
//  
//  void generateBitmapImage(unsigned char *image, int height, int width, const char* imageFileName) {
//  
//    unsigned char padding[3] = { 0, 0, 0 };
//    int paddingSize = (4 - (width*bytesPerPixel) % 4) % 4;
//  
//    unsigned char* fileHeader = createBitmapFileHeader(height, width, paddingSize);
//    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
//  
//    FILE* imageFile = fopen(imageFileName, "wb");
//  
//    fwrite(fileHeader, 1, fileHeaderSize, imageFile);
//    fwrite(infoHeader, 1, infoHeaderSize, imageFile);
//  
//    int i;
//    for (i = 0; i < height; i++) {
//      fwrite(image + (i*width*bytesPerPixel), bytesPerPixel, width, imageFile);
//      fwrite(padding, 1, paddingSize, imageFile);
//    }
//  
//    fclose(imageFile);
//  }
//  
//  unsigned char* createBitmapFileHeader(int height, int width, int paddingSize) {
//    int fileSize = fileHeaderSize + infoHeaderSize + (bytesPerPixel*width + paddingSize) * height;
//  
//    static unsigned char fileHeader[] = {
//        0,0, /// signature
//        0,0,0,0, /// image file size in bytes
//        0,0,0,0, /// reserved
//        0,0,0,0, /// start of pixel array
//    };
//  
//    fileHeader[0] = (unsigned char)('B');
//    fileHeader[1] = (unsigned char)('M');
//    fileHeader[2] = (unsigned char)(fileSize);
//    fileHeader[3] = (unsigned char)(fileSize >> 8);
//    fileHeader[4] = (unsigned char)(fileSize >> 16);
//    fileHeader[5] = (unsigned char)(fileSize >> 24);
//    fileHeader[10] = (unsigned char)(fileHeaderSize + infoHeaderSize);
//  
//    return fileHeader;
//  }
//  
//  unsigned char* createBitmapInfoHeader(int height, int width) {
//    static unsigned char infoHeader[] = {
//        0,0,0,0, /// header size
//        0,0,0,0, /// image width
//        0,0,0,0, /// image height
//        0,0, /// number of color planes
//        0,0, /// bits per pixel
//        0,0,0,0, /// compression
//        0,0,0,0, /// image size
//        0,0,0,0, /// horizontal resolution
//        0,0,0,0, /// vertical resolution
//        0,0,0,0, /// colors in color table
//        0,0,0,0, /// important color count
//    };
//  
//    infoHeader[0] = (unsigned char)(infoHeaderSize);
//    infoHeader[4] = (unsigned char)(width);
//    infoHeader[5] = (unsigned char)(width >> 8);
//    infoHeader[6] = (unsigned char)(width >> 16);
//    infoHeader[7] = (unsigned char)(width >> 24);
//    infoHeader[8] = (unsigned char)(height);
//    infoHeader[9] = (unsigned char)(height >> 8);
//    infoHeader[10] = (unsigned char)(height >> 16);
//    infoHeader[11] = (unsigned char)(height >> 24);
//    infoHeader[12] = (unsigned char)(1);
//    infoHeader[14] = (unsigned char)(bytesPerPixel * 8);
//  
//    return infoHeader;
// }
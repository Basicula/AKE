#include <stdio.h>
#include <iostream>
#include <chrono>

#include <BMPWriter.h>
#include <Vector.h>
#include <Ray.h>
#include <Sphere.h>
#include <IntersectionUtilities.h>

#define mpi 0

Picture TestSphere(int w, int h)
{
  Picture res = Picture(w, h);
  Sphere sphere(Vector3d(0, 0, 10), 10);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
      Ray ray(Vector3d(0, 0, -1), Vector3d(x - w / 2, y - h / 2, 50) - Vector3d(0, 0, -1));
      Vector3d intersection;
      if (IntersectRayWithSphere(intersection, ray, sphere))
        {
          double col = 10*(sphere.GetRadius()+Vector3d(0,0,-1).Distance(sphere.GetCenter()))/Vector3d(0,0,-1).Distance(intersection);
          res[y][x] = Pixel(col, 0, 0);
        }
    }
  return res;
}

int main()
{
  Vector3d vec(1, 2, 3), vec2(2, 3, 4);
  Vector3d res = vec - vec2;
  const size_t width = 800, height = 600;
  auto t1 = std::chrono::system_clock::now();
  auto t2 = std::chrono::system_clock::now();
  Picture mand;
  BMPWriter writer(width, height);
  mand = TestSphere(width, height);
  writer.SetPicture(mand);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\sphere.bmp");
#if mpi == 0
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height, OMP);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
#else
  t1 = std::chrono::system_clock::now();
  mand = MandelbrotSet(width, height, MPI);
  t2 = std::chrono::system_clock::now();
  std::cout << "Default time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
#endif
  writer.SetPicture(mand);
  writer.Write("D:\\Study\\RayTracing\\ResultsOutputs\\test.bmp");

  return 0;
}

// #include <stdio.h>
// 
// const int bytesPerPixel = 3; /// red, green, blue
// const int fileHeaderSize = 14;
// const int infoHeaderSize = 40;
// 
// void generateBitmapImage(unsigned char *image, int height, int width, const char* imageFileName);
// unsigned char* createBitmapFileHeader(int height, int width, int paddingSize);
// unsigned char* createBitmapInfoHeader(int height, int width);
// 
// 
// int main() {
//   const int height = 341;
//   const int width = 753;
//   unsigned char image[height][width][bytesPerPixel];
//   const char* imageFileName = "D:\\Study\\RayTracing\\ResultsOutputs\\test.bmp";
// 
//   int i, j;
//   for (i = 0; i < height; i++) {
//     for (j = 0; j < width; j++) {
//       image[i][j][2] = (unsigned char)((double)i / height * 255); ///red
//       image[i][j][1] = (unsigned char)((double)j / width * 255); ///green
//       image[i][j][0] = (unsigned char)(((double)i + j) / (height + width) * 255); ///blue
//     }
//   }
// 
//   generateBitmapImage((unsigned char *)image, height, width, imageFileName);
//   printf("Image generated!!");
// }
// 
// 
// void generateBitmapImage(unsigned char *image, int height, int width, const char* imageFileName) {
// 
//   unsigned char padding[3] = { 0, 0, 0 };
//   int paddingSize = (4 - (width*bytesPerPixel) % 4) % 4;
// 
//   unsigned char* fileHeader = createBitmapFileHeader(height, width, paddingSize);
//   unsigned char* infoHeader = createBitmapInfoHeader(height, width);
// 
//   FILE* imageFile = fopen(imageFileName, "wb");
// 
//   fwrite(fileHeader, 1, fileHeaderSize, imageFile);
//   fwrite(infoHeader, 1, infoHeaderSize, imageFile);
// 
//   int i;
//   for (i = 0; i < height; i++) {
//     fwrite(image + (i*width*bytesPerPixel), bytesPerPixel, width, imageFile);
//     fwrite(padding, 1, paddingSize, imageFile);
//   }
// 
//   fclose(imageFile);
// }
// 
// unsigned char* createBitmapFileHeader(int height, int width, int paddingSize) {
//   int fileSize = fileHeaderSize + infoHeaderSize + (bytesPerPixel*width + paddingSize) * height;
// 
//   static unsigned char fileHeader[] = {
//       0,0, /// signature
//       0,0,0,0, /// image file size in bytes
//       0,0,0,0, /// reserved
//       0,0,0,0, /// start of pixel array
//   };
// 
//   fileHeader[0] = (unsigned char)('B');
//   fileHeader[1] = (unsigned char)('M');
//   fileHeader[2] = (unsigned char)(fileSize);
//   fileHeader[3] = (unsigned char)(fileSize >> 8);
//   fileHeader[4] = (unsigned char)(fileSize >> 16);
//   fileHeader[5] = (unsigned char)(fileSize >> 24);
//   fileHeader[10] = (unsigned char)(fileHeaderSize + infoHeaderSize);
// 
//   return fileHeader;
// }
// 
// unsigned char* createBitmapInfoHeader(int height, int width) {
//   static unsigned char infoHeader[] = {
//       0,0,0,0, /// header size
//       0,0,0,0, /// image width
//       0,0,0,0, /// image height
//       0,0, /// number of color planes
//       0,0, /// bits per pixel
//       0,0,0,0, /// compression
//       0,0,0,0, /// image size
//       0,0,0,0, /// horizontal resolution
//       0,0,0,0, /// vertical resolution
//       0,0,0,0, /// colors in color table
//       0,0,0,0, /// important color count
//   };
// 
//   infoHeader[0] = (unsigned char)(infoHeaderSize);
//   infoHeader[4] = (unsigned char)(width);
//   infoHeader[5] = (unsigned char)(width >> 8);
//   infoHeader[6] = (unsigned char)(width >> 16);
//   infoHeader[7] = (unsigned char)(width >> 24);
//   infoHeader[8] = (unsigned char)(height);
//   infoHeader[9] = (unsigned char)(height >> 8);
//   infoHeader[10] = (unsigned char)(height >> 16);
//   infoHeader[11] = (unsigned char)(height >> 24);
//   infoHeader[12] = (unsigned char)(1);
//   infoHeader[14] = (unsigned char)(bytesPerPixel * 8);
// 
//   return infoHeader;
//}
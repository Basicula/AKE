import unittest
import time

from engine.Visual import Image, Color

class TestImageConstructor(unittest.TestCase):
    def test_simple_constructor(self):
        print("\nDefault constructor", end = "")
        width = 10
        height = 20
        image = Image(width, height)
        
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        
        default_color = 0xff000000
        for x in range(width):
            for y in range(height):
                self.assertEqual(image.getPixel(x,y), default_color)
    
    def test_full_constructor(self):
        print("\nCustom constructor", end = "")
        width = 12
        height = 32
        image = Image(width, height, 0xffffffff)
        
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        
        custom_color = 0xffffffff
        for x in range(width):
            for y in range(height):
                self.assertEqual(image.getPixel(x,y), custom_color)
                
class TestImageProperties(unittest.TestCase):
    def test_size_properties(self):
        print("\nProperties", end = "")
        width = 123
        height = 321
        image = Image(width, height)
        
        with self.assertRaises(AttributeError):
            image.width = 123
            
        with self.assertRaises(AttributeError):
            image.height = 123
            
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        self.assertEqual(image.size, height * width)
            
    def test_image_data(self):
        print("\nImage data", end = "")
            
        width = 40
        height = 40
        image = Image(width, height, 0xff563412)
            
        image_data = image.data()
        self.assertEqual(image_data[0], 0xff563412)
            
    def test_image_rgb_data(self):
        print("\nImage rgb data", end = "")
            
        width = 40
        height = 40
        image = Image(width, height, 0xff563412)
        
        image_rgb_data = image.rgbData()
        self.assertEqual(image_rgb_data[0], 0x12)
        self.assertEqual(image_rgb_data[1], 0x34)
        self.assertEqual(image_rgb_data[2], 0x56)
        
    def test_image_rgb_str_data(self):
        print("\nImage rgb data as str", end = "")
            
        width = 40
        height = 40
        image = Image(width, height, 0xff563412)
        
        image_rgb_str_data = image.rgbDataStr()
        self.assertEqual(image_rgb_str_data[0], chr(0x12))
        self.assertEqual(image_rgb_str_data[1], chr(0x34))
        self.assertEqual(image_rgb_str_data[2], chr(0x56))
        
        self.assertEqual(image_rgb_str_data, (chr(0x12) + chr(0x34) + chr(0x56)) * image.size)
        
class TestImageFunctionality(unittest.TestCase):
    def test_pixel_manipulation(self):
        print("\nPixel manipulation", end = "")
        width = 22
        height = 33
        image = Image(width, height, 0xff0000ff)
        
        expected_color = 0x00ff00
        image.setPixel(11, 11, expected_color)
        self.assertEqual(image.getPixel(11,11), expected_color)
        
        self.assertEqual(image.getPixel(123,123), 0xff000000)
        
        image.setPixel(123, 1312, 0xff000000)


if __name__ == "__main__":
    print("\n--------------")
    print("...Test Image...")
    unittest.main()
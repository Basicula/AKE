import unittest
import json

from engine.Visual import Color
from engine.Math import Vector as vec

class TestColorConstructors(unittest.TestCase):
    def test_creation(self):
        print("\nConstructors", end = "")
        color = Color()
        self.assertEqual(color.red,   0)
        self.assertEqual(color.green, 0)
        self.assertEqual(color.blue,  0)
        
        color = Color(0xff563412)
        self.assertEqual(color.red,   18)
        self.assertEqual(color.green, 52)
        self.assertEqual(color.blue,  86)
        
        color = Color(123, 231, 32)
        self.assertEqual(color.red,   123)
        self.assertEqual(color.green, 231)
        self.assertEqual(color.blue,  32)
        
class TestColorOperations(unittest.TestCase):
    def test_multuplication_by_factor(self):
        print("\nMultiplication by factor", end = "")
        color = Color(12,32,42)
        
        actual= color * 10
        expected = Color(120,255,255)
        self.assertEqual(actual, expected)
        
        actual= color * -2
        expected = Color()
        self.assertEqual(actual, expected)
        
        actual= color * 100
        expected = Color(255,255,255)
        self.assertEqual(actual, expected)
        
        actual= color * 0.1
        expected = Color(1,3,4)
        self.assertEqual(actual, expected)
        
    def test_multiplication_by_vector(self):
        print("\nMultiplication by vector", end = "")
        color = Color(123,231,42)
        
        common_vec = vec.Vector3d(0.3,0.5,0.7)
        actual = color * common_vec
        expected = Color(36,115,29)
        self.assertEqual(actual, expected)
        
        random_vec = vec.Vector3d(1.2,4.5,0.2)
        actual = color * random_vec
        expected = Color(147,255,8)
        self.assertEqual(actual, expected)
        
        negative_vec = vec.Vector3d(-0.3,0.5,-0.7)
        actual = color * negative_vec
        expected = Color(0,115,0)
        self.assertEqual(actual, expected)
        
    def test_color_sum(self):
        print("\nColor sum", end = "")
        color1 = Color(0xff0000ff)
        color2 = Color(0xff00ff00)
        
        actual = color1 + color2
        expected = Color(0xff00ffff)
        self.assertEqual(actual,expected)
        
        color1 = Color(0xff0000ff)
        color2 = Color(0xff3412ff)
        
        actual = color1 + color2
        expected = Color(0xff3412ff)
        self.assertEqual(actual,expected)
        
class TestColorProperties(unittest.TestCase):
    def test_color_properties(self):
        print("\nProperties", end = "")
        color = Color()
        
        self.assertEqual(color.red,   0)
        self.assertEqual(color.green, 0)
        self.assertEqual(color.blue,  0)
        
        color.red = 18
        color.green = 52
        color.blue = 86
        
        self.assertEqual(color.red,   18)
        self.assertEqual(color.green, 52)
        self.assertEqual(color.blue,  86)
        self.assertEqual(str(color), "0x00563412")
        
        with self.assertRaises(TypeError):
            color.red = 1000
            
        with self.assertRaises(TypeError):
            color.red = -1000
        
class TestColorFunctionality(unittest.TestCase):
    def test_serialization(self):
        print("\nColor serialization", end = "")
        color = Color(0xff563412)
        self.assertEqual(json.loads(repr(color)), {"Color" : 0xff563412})
        
    def test_deserialization(self):
        print("\nColor deserialization", end = "")
        color = Color(0xff563412)
        self.assertEqual(
            Color.fromDict(json.loads(repr(color))), 
            color)
    
if __name__ == "__main__":
    print("\n--------------")
    print("...Test Color...")
    unittest.main()
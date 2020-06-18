import unittest
import json
import math

from engine.Visual import Camera
from engine.Math.Vector import Vector3d

class TestCameraConstructor(unittest.TestCase):
    def test_constructor(self):
        print("\nConstructor", end = "")
        camera = Camera(Vector3d(0,0,0), Vector3d(0,0,1), Vector3d(0,1,0), 60, 4/3, 1)
        
        self.assertEqual(camera.location, Vector3d(0,0,0))
        
class TestCameraFunctionality(unittest.TestCase):
    def test_get_direction(self):
        print("\nGet direction", end = "")
        camera = Camera(Vector3d(0,0,0), Vector3d(0,0,1), Vector3d(0,1,0), 60, 4/3, 1)
        
        width, height = 800, 600
        u, v = 400, 300
        actual = camera.direction(u/width,v/height)
        expected = Vector3d(0,0,1)
        self.assertEqual(actual, expected)
        
        temp = camera.direction(0,v/height)
        self.assertTrue(temp.y == 0)
        self.assertTrue(temp.x != 0)
        
        temp = camera.direction(u/width,0)
        self.assertTrue(temp.y != 0)
        self.assertTrue(temp.x == 0)
        
    def test_serialization(self):
        print("\nCamera serialization", end = "")
        camera = Camera(Vector3d(0,0,0), Vector3d(0,0,1), Vector3d(0,1,0), 60, 4/3, 1)
        
        dict = json.loads(repr(camera))
        cam = dict["Camera"]
        self.assertEqual(cam["Location"], json.loads(repr(Vector3d(0,0,0))))
        self.assertEqual(cam["LookAt"], json.loads(repr(Vector3d(0,0,1))))
        self.assertEqual(cam["Up"], json.loads(repr(Vector3d(0,1,0))))
        self.assertEqual(cam["FoV"], 60)
        self.assertEqual(cam["Aspect"], round(4/3,6))
        self.assertEqual(cam["FocusDistance"], 1)
        
    def test_deserialization(self):
        print("\nCamera deserialization", end = "")
        camera = Camera(Vector3d(0,0,0), Vector3d(0,0,1), Vector3d(0,1,0), 60, 4/3, 1)
        
        dict = json.loads(repr(camera))
        cam = Camera.fromDict(dict)
        self.assertEqual(cam.location, Vector3d(0,0,0))
        self.assertEqual(dict, json.loads(repr(cam)))

if __name__ == "__main__":
    print("\n---------------")
    print("...Test Camera...")
    unittest.main()
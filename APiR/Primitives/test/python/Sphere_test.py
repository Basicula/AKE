import unittest
import math
import json

from engine.Common import BoundingBox, Ray
from engine.Common.Intersection import IntersectionRecord
from engine.Primitives import Sphere
from engine.Math.Vector import Vector3d


class TestSphereConstructor(unittest.TestCase):
    def test_with_default(self):
        print("\nDefault constructor", end = "")
        sphere = Sphere(Vector3d(0,0,0), 10)
        
        self.assertEqual(sphere.center, Vector3d(0,0,0))
        self.assertEqual(sphere.radius, 10)

    def test_full_constructor(self):
        print("\nFull constructor", end = "")
        sphere = Sphere(Vector3d(0,0,0), 10)
        
        self.assertEqual(sphere.center, Vector3d(0,0,0))
        self.assertEqual(sphere.radius, 10)

class TestSphereProperties(unittest.TestCase):
    def test_properties(self):
        print("\nProperties", end = "")
        sphere = Sphere(Vector3d(0,0,0), 10)
        
        sphere.center = Vector3d(1,2,3)
        self.assertEqual(sphere.center, Vector3d(1,2,3))
        
        sphere.radius = 123
        self.assertEqual(sphere.radius, 123)

class TestSphereRayIntersection(unittest.TestCase):
    def test_two_intersections(self):
        print("\nRay intersects Sphere (two intersections)", end = "")
        sphere = Sphere(Vector3d(0,0,30), 10)
        ray = Ray(Vector3d(0,0,0),Vector3d(0,0,1))
        intersection = IntersectionRecord()
        
        hitted = sphere.hitRay(intersection,ray)
        self.assertEqual(hitted, True)
        self.assertEqual(intersection.distance, 20)
        self.assertEqual(intersection.intersection, Vector3d(0,0,20))
        self.assertEqual(intersection.normal, Vector3d(0,0,-1))
        
        intersection = IntersectionRecord()
        sphere = Sphere(Vector3d(0,0,0), 1)
        x = 1/math.sqrt(2)
        ray = Ray(Vector3d(x,0,-5),Vector3d(0,0,1))
        hitted = sphere.hitRay(intersection,ray)
        self.assertEqual(hitted, True)
        self.assertEqual(intersection.distance, 5-x)
        
        self.assertEqual(round(intersection.intersection.x,12), round(x,12))
        self.assertEqual(intersection.intersection.y, 0)
        self.assertEqual(round(intersection.intersection.z,12), round(-x,12))
        
        self.assertEqual(round(intersection.normal.x,12), round(x,12))
        self.assertEqual(intersection.normal.y, 0)
        self.assertEqual(round(intersection.normal.z,12), round(-x,12))

    def test_no_intersection(self):
        print("\nNo intersections", end = "")
        sphere = Sphere(Vector3d(0,0,30), 10)
        ray = Ray(Vector3d(0,0,0),Vector3d(0,0,-1))
        intersection = IntersectionRecord()
        
        hitted = sphere.hitRay(intersection,ray)
        self.assertEqual(hitted, False)
        
        ray = Ray(Vector3d(10.0001,0,0),Vector3d(0,0,1))
        hitted = sphere.hitRay(intersection,ray)
        self.assertEqual(hitted, False)

    def test_tangent(self):
        print("\nTanget ray", end = "")
        sphere = Sphere(Vector3d(0,0,30), 10)
        ray = Ray(Vector3d(10,0,0),Vector3d(0,0,1))
        intersection = IntersectionRecord()
        
        hitted = sphere.hitRay(intersection,ray)
        self.assertEqual(hitted, True)
        self.assertEqual(intersection.distance, 30)
        self.assertEqual(intersection.intersection, Vector3d(10,0,30))
        self.assertEqual(intersection.normal, Vector3d(1,0,0))

class TestSphereFunctionality(unittest.TestCase):
    def test_serialization(self):
        print("\nSphere serialization", end = "")
        sphere = Sphere(Vector3d(0,0,0), 10)
        
        dict = json.loads(repr(sphere))
        obj = dict["Sphere"]
        self.assertEqual(obj["Center"], json.loads(repr(Vector3d(0))))
        self.assertEqual(obj["Radius"], 10)       
    
    def test_deserialization(self):
        print("\nSphere deserialization", end = "")
        sphere = Sphere(Vector3d(0,0,0), 10)
        
        dict = json.loads(repr(sphere))
        obj = Sphere.fromDict(dict)
        self.assertEqual(obj.center, Vector3d(0))
        self.assertEqual(obj.radius, 10)
        
    def test_bounding_box(self):
        print("\nBounding box", end = "")
        sphere = Sphere(Vector3d(0,1,2), 5)
        
        bb = sphere.boundingBox
        
        self.assertEqual(bb.min, Vector3d(-5,-4,-3))
        self.assertEqual(bb.max, Vector3d(5,6,7))
        
if __name__ == "__main__":
    print("\n---------------")
    print("...Test Sphere...")
    unittest.main()
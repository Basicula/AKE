import unittest

from engine.Common.Intersection.Utils import *
from engine.Common import Ray, BoundingBox
from engine.Common.Intersection import RayBoxIntersectionRecord
from engine.Math.Vector import Vector3d

def temp(a):
    a = 2
    return True

class TestRayBoxIntersection(unittest.TestCase):
    def test_ray_intersect_box(self):
        print("\nRay Intersect Box", end = "")
        min_corner = Vector3d(-1)
        max_corner = Vector3d(1)
        box = BoundingBox(min_corner, max_corner)
        ray = Ray(Vector3d(0,0,-3),Vector3d(0,0,1))
        
        self.assertTrue(rayIntersectBox(ray,box))
        
        ray.direction = (max_corner - ray.origin).normalized()
        self.assertTrue(rayIntersectBox(ray,box))
        
        ray.direction = (min_corner - ray.origin).normalized()
        self.assertTrue(rayIntersectBox(ray,box))
        
        near_front_right_top_corner = Vector3d(1+1e-10,1,-1)
        near_front_bottom_corner = Vector3d(-1-1e-10,-1,-1)
        ray.direction = (near_front_right_top_corner - ray.origin).normalized()
        self.assertFalse(rayIntersectBox(ray,box))        
        ray.direction = (near_front_bottom_corner - ray.origin).normalized()
        self.assertFalse(rayIntersectBox(ray,box))

    def test_ray_box_intersection(self):
        print("\nRay Box Intersection", end = "")
        min_corner = Vector3d(-1)
        max_corner = Vector3d(1)
        box = BoundingBox(min_corner, max_corner)
        ray = Ray(Vector3d(0,0,-3),Vector3d(0,0,1))
        
        intersection = RayBoxIntersectionRecord()
        rayBoxIntersection(ray,box,intersection)
        self.assertEqual(2, intersection.tmin)
        self.assertEqual(4, intersection.tmax)
    
if __name__ == "__main__":
    print("\n-----------------------------")
    print("\n...Test Intersection Utils...")
    unittest.main()
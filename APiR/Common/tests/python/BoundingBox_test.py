import unittest
import json

from engine.Math.Vector import Vector3d
from engine.Common import BoundingBox as bbox


class TestBoundingBoxConstructors(unittest.TestCase):
    def test_default_constructor(self):
        print("\nDefault constructor", end = "")
        bb = bbox()
        
        self.assertFalse(bb.isValid())
        
    def test_constructor_with_two_parameters(self):
        print("\nMinMax constructor", end = "")
        bb = bbox(Vector3d(0), Vector3d(1))
        
        self.assertEqual(bb.min, Vector3d(0))
        self.assertEqual(bb.max, Vector3d(1))
        self.assertTrue(bb.isValid())
        
        bb = bbox(Vector3d(1), Vector3d(0))
        self.assertFalse(bb.isValid())

class TestBoundingBoxProperties(unittest.TestCase):
    def test_properties(self):
        print("\nProperties", end = "")
        bb = bbox()
        
        self.assertEqual(bb.min, Vector3d(2147483647))
        self.assertEqual(bb.max, Vector3d(-2147483648))
        self.assertEqual(bb.center, Vector3d(-0.5))
        
        with self.assertRaises(AttributeError):
            bb.min = Vector3d(0)
        
        with self.assertRaises(AttributeError):
            bb.max = Vector3d(0)
            
        with self.assertRaises(AttributeError):
            bb.center = Vector3d(0)
            
        bb = bbox(Vector3d(0,1,2), Vector3d(4,6,8))
        self.assertEqual(bb.deltaX, 4)
        self.assertEqual(bb.deltaY, 5)
        self.assertEqual(bb.deltaZ, 6)
        
        with self.assertRaises(AttributeError):
            bb.deltaX = 0
            
        with self.assertRaises(AttributeError):
            bb.deltaY = 0

        with self.assertRaises(AttributeError):
            bb.deltaZ = 0

class TestBoundingBoxFunctionality(unittest.TestCase):
    def test_adding_point(self):
        print("\nAdd point", end = "")
        bb = bbox()
        
        bb.addPoint(Vector3d(1))
        self.assertFalse(bb.isValid())
        
        bb.addPoint(Vector3d(0))
        self.assertTrue(bb.isValid())
        
        bb.addPoint(Vector3d(0.5))
        
        self.assertEqual(bb.min, Vector3d(0))
        self.assertEqual(bb.max, Vector3d(1))
        
    def test_serialization(self):
        print("\nSerialization", end = "")
        bb = bbox(Vector3d(0), Vector3d(1))
        
        dict = json.loads(repr(bb))
        box = dict["BoundingBox"]
        
        self.assertEqual(box["MinCorner"], json.loads(repr(Vector3d(0))))
        self.assertEqual(box["MaxCorner"], json.loads(repr(Vector3d(1))))
        
    def test_deserialization(self):
        print("\nDeserialization", end = "")
        bb = bbox(Vector3d(0), Vector3d(1))
        
        dict = json.loads(repr(bb))
        box = bbox.fromDict(dict)
        
        self.assertEqual(bb.min, box.min)
        self.assertEqual(bb.max, box.max)
        
    def test_containing_point(self):
        print("\nContaining point", end="")
        bb = bbox(Vector3d(0), Vector3d(5))
        
        self.assertTrue(bb.contains(Vector3d(0)))
        self.assertTrue(bb.contains(Vector3d(5)))
        self.assertTrue(bb.contains(Vector3d(3)))
        self.assertTrue(bb.contains(Vector3d(3,0,0)))
        self.assertTrue(bb.contains(Vector3d(0,3,0)))
        self.assertTrue(bb.contains(Vector3d(0,0,3)))
        
        self.assertFalse(bb.contains(Vector3d(-1)))
        self.assertFalse(bb.contains(Vector3d(6)))
        
    def test_merge(self):
        print("\nMerge of boxes", end = "")
        bb1 = bbox(Vector3d(0), Vector3d(1))
        bb2 = bbox(Vector3d(0.5), Vector3d(1.5))
        
        bb1.merge(bb2)
        self.assertEqual(bb1.min, Vector3d(0))
        self.assertEqual(bb1.max, Vector3d(1.5))

if __name__ == "__main__":
    print("\n----------------------")
    print("\n...Test BoundingBox...")
    unittest.main()
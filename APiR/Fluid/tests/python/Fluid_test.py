import unittest
import json

from engine.Fluid import Fluid
from engine.Common import BoundingBox
from engine.Math.Vector import Vector3d

class TestFluidConstructors(unittest.TestCase):
    def test_default_constructor(self):
        print("\nDefault constructor", end = "")
        with self.assertRaises(TypeError):
            fluid = Fluid()
        
    def test_constructor_with_value(self):
        print("\nConstructor with value", end ="")
        fluid = Fluid(1024)
        
        self.assertEqual(fluid.numOfParticles, 1024)
        self.assertTrue(fluid.boundingBox.isValid())
        
class TestFluidFunctionality(unittest.TestCase):
    def test_serialization(self):
        print("\nSerialization", end = "")
        fluid = Fluid(64)
        
        dict = json.loads(repr(fluid))
        inner = dict["Fluid"]
        
        self.assertEqual(inner["NumOfParticles"], 64)
        
    def test_deserialization(self):
        print("\nDeserialization", end = "")
        fluid = Fluid(1024)
        
        dict = json.loads(repr(fluid))
        fluid_re = Fluid.fromDict(dict)
        
        self.assertEqual(fluid_re.numOfParticles, fluid.numOfParticles)
        self.assertTrue(fluid.boundingBox.isValid())
        self.assertTrue(fluid_re.boundingBox.isValid())

if __name__ == "__main__":
    print("\n--------------")
    print("...Test Fluid...")
    unittest.main()
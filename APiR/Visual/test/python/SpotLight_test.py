import unittest
import json

from engine.Math.Vector import Vector3d
from engine.Visual import Color
from engine.Visual.Light import SpotLight

class TestSpotLightConstructor(unittest.TestCase):
    def test_default_constructor(self):
        print("\nDefault constructor", end = "")
        light = SpotLight(Vector3d(0,0,0))
        
        self.assertEqual(light.location, Vector3d(0,0,0))
        self.assertEqual(light.color, Color(0xffffff))
        self.assertEqual(light.intensity, 1)
        self.assertEqual(light.state, True)
        
    def test_custom_construtor(self):
        print("\nCustom constructor", end = "")
        light = SpotLight(Vector3d(1,2,3),Color(0xff00ff),23, False)
        
        self.assertEqual(light.location, Vector3d(1,2,3))
        self.assertEqual(light.color, Color(0xff00ff))
        self.assertEqual(light.intensity, 23)
        self.assertEqual(light.state, False)
        
class TestSpotLightProperties(unittest.TestCase):
    def test_properties(self):
        print("\nProperties", end = "")
        light = SpotLight(Vector3d(1))
        
        light.location = Vector3d(1,2,3)
        self.assertEqual(light.location, Vector3d(1,2,3))
        
        light.color = Color(0xff00ff)
        self.assertEqual(light.color, Color(0xff00ff))
        
        light.intensity = 23
        self.assertEqual(light.intensity, 23)
        
        light.state = False
        self.assertEqual(light.state, False)
    
class TestSpotLightFunctionality(unittest.TestCase):
    def test_get_functions(self):
        print("\nSpotLight direction at point", end = "")
        light = SpotLight(Vector3d(0))
        
        self.assertEqual(light.direction(Vector3d(1,0,0)), Vector3d(1,0,0))        
        self.assertEqual(light.direction(Vector3d(0,1,0)), Vector3d(0,1,0))        
        self.assertEqual(light.direction(Vector3d(0,0,1)), Vector3d(0,0,1))        
        
    def test_serialization(self):
        print("\nSpotLight serialization", end = "")
        light = SpotLight(Vector3d(0))
        
        dict = json.loads(repr(light))
        splight = dict["SpotLight"]
        self.assertEqual(splight["Location"], json.loads(repr(Vector3d(0))))
        self.assertEqual(splight["Color"], json.loads(repr(Color(0xffffff))))
        self.assertEqual(splight["Intensity"], 1)
        self.assertEqual(splight["State"], True)
        
    def test_deserialization(self):
        print("\nSpotLight deserialization", end = "")
        light = SpotLight(Vector3d(0))
        
        dict = json.loads(repr(light))
        splight = SpotLight.fromDict(dict)
        self.assertEqual(splight.location, Vector3d(0))
        self.assertEqual(splight.color, Color(0xffffff))
        self.assertEqual(splight.intensity, 1)
        self.assertEqual(splight.state, True)
        
        
if __name__ == "__main__":
    print("\n------------------")
    print("...Test SpotLight...")
    unittest.main()
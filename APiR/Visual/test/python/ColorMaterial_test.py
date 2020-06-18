import unittest
import json

from engine.Visual import Color
from engine.Visual.Light import SpotLight
from engine.Visual.Material import ColorMaterial
from engine.Math.Vector import Vector3d

class TestColorMaterialConstructors(unittest.TestCase):
    def test_simple_constructor(self):
        print("\nSimple constructor", end = "")
        cm = ColorMaterial(Color(0x00ff00))
        
        self.assertEqual(cm.color, Color(0x00ff00))
        self.assertEqual(cm.ambient, Vector3d(1,1,1))
        self.assertEqual(cm.diffuse, Vector3d(1,1,1))
        self.assertEqual(cm.specular, Vector3d(1,1,1))
        self.assertEqual(cm.shinines, 1)
        self.assertEqual(cm.reflection, 0)
        self.assertEqual(cm.refraction, 0)
        
    def test_full_constructor(self):
        print("\nFull constructor", end = "")
        ruby_material = ColorMaterial(Color(255, 0, 0)
                                    , Vector3d(0.1745, 0.01175, 0.01175)
                                    , Vector3d(0.61424, 0.04136, 0.04136)
                                    , Vector3d(0.727811, 0.626959, 0.626959)
                                    , 76.8)
        
        self.assertEqual(ruby_material.color, Color(0xff0000))
        self.assertEqual(ruby_material.ambient, Vector3d(0.1745, 0.01175, 0.01175))
        self.assertEqual(ruby_material.diffuse, Vector3d(0.61424, 0.04136, 0.04136))
        self.assertEqual(ruby_material.specular, Vector3d(0.727811, 0.626959, 0.626959))
        self.assertEqual(ruby_material.shinines, 76.8)
        self.assertEqual(ruby_material.reflection, 0)
        self.assertEqual(ruby_material.refraction, 0)

class TestColorMaterialProperties(unittest.TestCase):
    def test_properties(self):
        print("\nProperties", end = "")
        ruby_material = ColorMaterial(Color(255, 0, 0)
                                    , Vector3d(0.1745, 0.01175, 0.01175)
                                    , Vector3d(0.61424, 0.04136, 0.04136)
                                    , Vector3d(0.727811, 0.626959, 0.626959)
                                    , 76.8)
        
        self.assertEqual(ruby_material.color, Color(0xff0000))
        self.assertEqual(ruby_material.ambient, Vector3d(0.1745, 0.01175, 0.01175))
        self.assertEqual(ruby_material.diffuse, Vector3d(0.61424, 0.04136, 0.04136))
        self.assertEqual(ruby_material.specular, Vector3d(0.727811, 0.626959, 0.626959))
        self.assertEqual(ruby_material.shinines, 76.8)
        self.assertEqual(ruby_material.reflection, 0)
        self.assertEqual(ruby_material.refraction, 0)
        
        ruby_material.color = Color(0x00ff00)
        self.assertEqual(ruby_material.color, Color(0x00ff00))
        
        ruby_material.ambient = Vector3d(0,0,0)
        self.assertEqual(ruby_material.ambient, Vector3d(0,0,0))
        self.assertEqual(ruby_material.acolor(), Color(0))
        
        ruby_material.diffuse = Vector3d(0,0,0)
        self.assertEqual(ruby_material.diffuse, Vector3d(0,0,0))
        self.assertEqual(ruby_material.dcolor(), Color(0))
        
        ruby_material.specular = Vector3d(0,0,0)
        self.assertEqual(ruby_material.specular, Vector3d(0,0,0))
        self.assertEqual(ruby_material.scolor(), Color(0))
        
        ruby_material.shinines = 0
        self.assertEqual(ruby_material.shinines, 0)
        
        ruby_material.reflection= 1
        self.assertEqual(ruby_material.reflection, 1)
        
        ruby_material.refraction = 1
        self.assertEqual(ruby_material.refraction, 1)

class TestColorMaterialFunctionality(unittest.TestCase):
    def test_color_get_functions(self):
        print("\nColorMaterial get color functions", end = "")
        ruby_material = ColorMaterial(Color(255, 0, 0)
                                    , Vector3d(0.1745, 0.01175, 0.01175)
                                    , Vector3d(0.61424, 0.04136, 0.04136)
                                    , Vector3d(0.727811, 0.626959, 0.626959)
                                    , 76.8)
        light = SpotLight(Vector3d(2,0,0))
        light1 = SpotLight(Vector3d(4,0,0))
        
        self.assertEqual(
            ruby_material.primitiveColor(), 
            Color(255,0,0) * Vector3d(0.1745, 0.01175, 0.01175))
        self.assertEqual(
            ruby_material.lightInfluence(Vector3d(0),Vector3d(1,0,0),light), 
            Color(255,0,0) * Vector3d(0.61424, 0.04136, 0.04136))
        
    def test_serialization(self):
        print("\nColorMaterial serialization", end = "")
        ruby_material = ColorMaterial(Color(255, 0, 0)
                                    , Vector3d(0.1745, 0.01175, 0.01175)
                                    , Vector3d(0.61424, 0.04136, 0.04136)
                                    , Vector3d(0.727811, 0.626959, 0.626959)
                                    , 76.8)
        
        dict = json.loads(repr(ruby_material))
        mat = dict["ColorMaterial"]
        self.assertEqual(mat["Color"], json.loads(repr(Color(255,0,0))))
        self.assertEqual(mat["Ambient"], json.loads(repr(Vector3d(0.1745, 0.01175, 0.01175))))
        self.assertEqual(mat["Diffuse"], json.loads(repr(Vector3d(0.61424, 0.04136, 0.04136))))
        self.assertEqual(mat["Specular"], json.loads(repr(Vector3d(0.727811, 0.626959, 0.626959))))
        self.assertEqual(mat["Shinines"], 76.8)
        self.assertEqual(mat["Reflection"], 0)
        self.assertEqual(mat["Refraction"], 0)
        
    def test_deserialization(self):
        print("\nColorMaterial deserialization", end = "")
        ruby_material = ColorMaterial(Color(255, 0, 0)
                                    , Vector3d(0.1745, 0.01175, 0.01175)
                                    , Vector3d(0.61424, 0.04136, 0.04136)
                                    , Vector3d(0.727811, 0.626959, 0.626959)
                                    , 76.8)
        
        dict = json.loads(repr(ruby_material))
        material = ColorMaterial.fromDict(dict)
        self.assertEqual(material.color, Color(255,0,0))
        self.assertEqual(material.ambient, Vector3d(0.1745, 0.01175, 0.01175))
        self.assertEqual(material.diffuse, Vector3d(0.61424, 0.04136, 0.04136))
        self.assertEqual(material.specular, Vector3d(0.727811, 0.626959, 0.626959))
        self.assertEqual(material.shinines, 76.8)
        self.assertEqual(material.reflection, 0)
        self.assertEqual(material.refraction, 0)
        
if __name__ == "__main__":
    print("\n----------------------")
    print("...Test ColorMaterial...")
    unittest.main()
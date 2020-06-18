import unittest
import json
import matplotlib.pyplot as plt

from engine import Scene
from engine.Math.Vector import Vector3d
from engine.Visual import *
from engine.Primitives import *
from engine.Visual.Light import *
from engine.Visual.Material import *
from engine.Visual.Renderable import RenderableObject

def create_scene_sample(camera_cnt = 1, obj_cnt = 1, light_cnt = 1):
    scene = Scene("example")
    
    for i in range(camera_cnt):
        scene.addCamera(Camera(Vector3d(2*i+1,3*i+1,4*i+1),Vector3d(0,0,0),Vector3d(0,1,0),60,4/3,1), True)
        
    ruby_material = ColorMaterial(Color(255, 0, 0)
                        , Vector3d(0.1745, 0.01175, 0.01175)
                        , Vector3d(0.61424, 0.04136, 0.04136)
                        , Vector3d(0.727811, 0.626959, 0.626959)
                        , 76.8)
    for i in range(obj_cnt):
        scene.addObject(RenderableObject(Sphere(Vector3d(i*10),i*5), ruby_material))
    
    for i in range(light_cnt):
        scene.addLight(SpotLight(Vector3d(-i*2,i*4,i*5)))
        
    return scene
    
def dump_scene_image(scene):
    width, height = 800, 600
    image = Image(width, height)
    scene.getFrame(image)
    img = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel = Color(image.getPixel(x,y))
            row.append([pixel.red, pixel.green, pixel.blue])
        img.append(row)
    img.reverse()
    plt.imshow(img)
    plt.savefig('test.png')
    
def test(name = "example"):
    width = 640
    height = 480
    scene = Scene(name, width, height)
    
    ruby_material = ColorMaterial(Color(255, 0, 0)
                            , Vector3d(0.1745, 0.01175, 0.01175)
                            , Vector3d(0.61424, 0.04136, 0.04136)
                            , Vector3d(0.727811, 0.626959, 0.626959)
                            , 76.8)
    
    scene.addObject(RenderableObject(Sphere(Vector3d(0), 2), ruby_material))
    #scene.addObject(RenderableObject(Sphere(Vector3d(4,4,-4), 1), ruby_material))
    
    #scene.addLight(SpotLight(Vector3d(10,10,10)))
    scene.addLight(SpotLight(Vector3d(5,6,-6)))
    
    scene.addCamera(Camera(
      Vector3d(0, 7, 7),
      Vector3d(0, 0, 0),
      Vector3d(0, -1, 1).normalized(),
      75,
      width * 1.0 / height,
      2), True)
      
    scene.addCamera(Camera(
      Vector3d(0, 10, 0),
      Vector3d(0, 0, 0),
      Vector3d(0, 0, 1),
      75,
      width * 1.0 / height,
      2), False)
      
    scene.addCamera(Camera(
      Vector3d(10, 0, 0),
      Vector3d(0, 0, 0),
      Vector3d(0, -1, 0),
      75,
      width * 1.0 / height,
      2), False)
      
    scene.addCamera(Camera(
      Vector3d(0, 0, 10),
      Vector3d(0, 0, 0),
      Vector3d(0, -1, 0),
      75,
      width * 1.0 / height,
      2), False)
    
    with open(name+'.json','w') as f:
        json.dump(json.loads(repr(scene)), f)
    
    dump_scene_image(scene)
    

class TestSceneConstructor(unittest.TestCase):
    def test_constructor(self):
        print("\nConstructor without parameters", end = "")
        scene = Scene()
        
        self.assertEqual(scene.name, "unnamed")
        
    def test_custom_constructor(self):
        print("\nConstructor with parameters", end = "")
        scene = Scene("test")
        
        self.assertEqual(scene.name, "test")
        
class TestSceneExample(unittest.TestCase):
    def test_simple_example(self):
        print("\nSimple example", end = "")
        scene = Scene("example")
        
        self.assertEqual(scene.name, "example")
        
        ruby_material = ColorMaterial(Color(255, 0, 0)
                        , Vector3d(0.1745, 0.01175, 0.01175)
                        , Vector3d(0.61424, 0.04136, 0.04136)
                        , Vector3d(0.727811, 0.626959, 0.626959)
                        , 76.8)
        sphere = Sphere(Vector3d(0,0,0), 10)
        object = RenderableObject(sphere, ruby_material)
        light = SpotLight(Vector3d(10,10,10))
        camera = Camera(Vector3d(0,0,-10),Vector3d(0,0,0),Vector3d(0,1,0),60,4/3,1)
        
        scene.addObject(object)
        self.assertEqual(scene.objCnt,1)
        scene.clearObjects()
        self.assertEqual(scene.objCnt,0)
        
        scene.addLight(light)
        self.assertEqual(scene.lightCnt,1)
        self.assertTrue(light.state)
        scene.setOnOffLight(0,False)
        self.assertFalse(light.state)
        scene.clearLights()
        self.assertEqual(scene.lightCnt,0)
        
        scene.addCamera(camera)
        self.assertEqual(scene.camCnt,1)
        scene.clearCameras()
        self.assertEqual(scene.camCnt,0)
        
class TestSceneFunctionality(unittest.TestCase):
    def test_serialization(self):
        print("\nScene serialization", end = "")
        obj_cnt = 12
        light_cnt = 2
        camera_cnt = 1
        scene = create_scene_sample(camera_cnt, obj_cnt, light_cnt)
        
        dict = json.loads(repr(scene))
        scene_dict = dict["Scene"]
        
        self.assertEqual(scene_dict["Name"], "example")
        self.assertEqual(scene_dict["ActiveCamera"], 0)
        self.assertEqual(scene_dict["FrameWidth"], 800)
        self.assertEqual(scene_dict["FrameHeight"], 600)
        self.assertEqual(len(scene_dict["Objects"]), obj_cnt)
        self.assertEqual(len(scene_dict["Lights"]), light_cnt)
        self.assertEqual(len(scene_dict["Cameras"]), camera_cnt)
        
    def test_deserialization(self):
        print("\nScene deserialization", end = "")
        obj_cnt = 10
        light_cnt = 2
        camera_cnt = 1
        scene = create_scene_sample(camera_cnt, obj_cnt, light_cnt)
        
        dict = json.loads(repr(scene))
        s = Scene.fromDict(dict)
        
        self.assertEqual(s.name, "example")
        self.assertEqual(s.frameWidth, 800)
        self.assertEqual(s.frameHeight, 600)
        self.assertEqual(s.objCnt, obj_cnt)
        self.assertEqual(s.lightCnt, light_cnt)
        self.assertEqual(s.camCnt, camera_cnt)
    
if __name__ == "__main__":
    print("\n--------------")
    print("...Test Scene...")
    unittest.main()
import unittest
import math
import json
import engine.Math.Vector as vec

class TestVectorOutput(unittest.TestCase):
    def test_str(self):
        print("\nVector as str", end = "")
        vec2d = vec.Vector2d()
        vec3d = vec.Vector3d()
        vec4d = vec.Vector4d()
        
        self.assertEqual(str(vec2d), "Vector2d")
        self.assertEqual(str(vec3d), "Vector3d")
        self.assertEqual(str(vec4d), "Vector4d")
    
    def test_serialization(self):
        print("\nVector serialization", end = "")
        vec2d = vec.Vector2d()
        vec3d = vec.Vector3d()
        vec4d = vec.Vector4d()
        
        self.assertEqual(json.loads(repr(vec2d)), {"Vector2d" : [0, 0]})
        self.assertEqual(json.loads(repr(vec3d)), {"Vector3d" : [0, 0, 0]})
        self.assertEqual(json.loads(repr(vec4d)), {"Vector4d" : [0, 0, 0, 0]})
        
        vec2d = vec.Vector2d(1)
        vec3d = vec.Vector3d(2)
        vec4d = vec.Vector4d(3)
        
        self.assertEqual(json.loads(repr(vec2d)), {"Vector2d" : [1, 1]})
        self.assertEqual(json.loads(repr(vec3d)), {"Vector3d" : [2, 2, 2]})
        self.assertEqual(json.loads(repr(vec4d)), {"Vector4d" : [3, 3, 3, 3]})
        
        vec2d = vec.Vector2d(1.23, 32.132)
        vec3d = vec.Vector3d(2.2, 312.1231, 2431.123)
        vec4d = vec.Vector4d(3.231, 1231.23, 1231.213, 1231.213)
        
        self.assertEqual(json.loads(repr(vec2d)), {"Vector2d" : [1.23, 32.132]})
        self.assertEqual(json.loads(repr(vec3d)), {"Vector3d" : [2.2, 312.1231, 2431.123]})
        self.assertEqual(json.loads(repr(vec4d)), {"Vector4d" : [3.231, 1231.23, 1231.213, 1231.213]})

    def test_deserialization(self):
        print("\nVector deserialization", end = "")
        vec2d = vec.Vector2d()
        vec3d = vec.Vector3d()
        vec4d = vec.Vector4d()
        
        self.assertEqual(
            vec.Vector2d.fromDict(json.loads(repr(vec2d))), 
            vec2d)
        self.assertEqual(
            vec.Vector3d.fromDict(json.loads(repr(vec3d))),
            vec3d)
        self.assertEqual(
            vec.Vector4d.fromDict(json.loads(repr(vec4d))),
            vec4d)
        
        vec2d = vec.Vector2d(1)
        vec3d = vec.Vector3d(2)
        vec4d = vec.Vector4d(3)
        
        self.assertEqual(
            vec.Vector2d.fromDict(json.loads(repr(vec2d))), 
            vec2d)
        self.assertEqual(
            vec.Vector3d.fromDict(json.loads(repr(vec3d))),
            vec3d)
        self.assertEqual(
            vec.Vector4d.fromDict(json.loads(repr(vec4d))),
            vec4d)
        
        vec2d = vec.Vector2d(1.23, 32.132)
        vec3d = vec.Vector3d(2.2, 312.1231, 2431.123)
        vec4d = vec.Vector4d(3.231, 1231.23, 1231.213, 1231.213)
        
        self.assertEqual(
            vec.Vector2d.fromDict(json.loads(repr(vec2d))), 
            vec2d)
        self.assertEqual(
            vec.Vector3d.fromDict(json.loads(repr(vec3d))),
            vec3d)
        self.assertEqual(
            vec.Vector4d.fromDict(json.loads(repr(vec4d))),
            vec4d)
            
        bad_dict = {"Vect" : [0,0]}
        with self.assertRaises(KeyError):
            vec.Vector2d.fromDict(bad_dict)
    
class TestVectorConstructors(unittest.TestCase):
    def test_vec2_constructors(self):
        print("\nVector2d constructor", end = "")
        vec2d = vec.Vector2d()
        self.assertEqual(vec2d.x, 0.0)
        self.assertEqual(vec2d.y, 0.0)
        
        vec2d = vec.Vector2d(1,2)
        self.assertEqual(vec2d.x, 1.0)
        self.assertEqual(vec2d.y, 2.0)
        
        vec2d = vec.Vector2d(1)
        self.assertEqual(vec2d.x, 1.0)
        self.assertEqual(vec2d.y, 1.0)
        
    def test_vec3_constructors(self):
        print("\nVector3d constructor", end = "")
        vec3d = vec.Vector3d()
        self.assertEqual(vec3d.x, 0.0)
        self.assertEqual(vec3d.y, 0.0)
        self.assertEqual(vec3d.z, 0.0)
        
        vec3d = vec.Vector3d(1)
        self.assertEqual(vec3d.x, 1.0)
        self.assertEqual(vec3d.y, 1.0)
        self.assertEqual(vec3d.z, 1.0)
        
        vec3d = vec.Vector3d(1,2,3)
        self.assertEqual(vec3d.x, 1.0)
        self.assertEqual(vec3d.y, 2.0)
        self.assertEqual(vec3d.z, 3.0)
        
        with self.assertRaises(TypeError):
            vec.Vector3d(1,2)
            
    def test_vec4_constructors(self):
        print("\nVector4d constructor", end = "")
        vec4d = vec.Vector4d()
        self.assertEqual(vec4d.x, 0.0)
        self.assertEqual(vec4d.y, 0.0)
        self.assertEqual(vec4d.z, 0.0)
        self.assertEqual(vec4d.w, 0.0)
        
        vec4d = vec.Vector4d(1,2,3,4)
        self.assertEqual(vec4d.x, 1.0)
        self.assertEqual(vec4d.y, 2.0)
        self.assertEqual(vec4d.z, 3.0)
        self.assertEqual(vec4d.w, 4.0)
        
        vec4d = vec.Vector4d(1)
        self.assertEqual(vec4d.x, 1.0)
        self.assertEqual(vec4d.y, 1.0)
        self.assertEqual(vec4d.z, 1.0)
        self.assertEqual(vec4d.w, 1.0)
        
        with self.assertRaises(TypeError):
            vec.Vector4d(1,2)
            
        with self.assertRaises(TypeError):
            vec.Vector4d(1,2,3)

class TestVectorOperations(unittest.TestCase):
    def test_vec2_simple_operations(self):
        print("\nVector2d simple operations", end = "")
        v1, v2 = vec.Vector2d(1,2), vec.Vector2d(2,1)
        
        actual = v1 + v2
        expected = vec.Vector2d(3)
        self.assertEqual(actual, expected)
        
        v1 += v2
        expected = vec.Vector2d(3)
        self.assertEqual(v1, expected)
        
        v1 -= v2
        expected = vec.Vector2d(1,2)
        self.assertEqual(v1, expected)
        
        actual = v1 - v2
        expected = vec.Vector2d(-1, 1)
        self.assertEqual(actual, expected)
        
        actual = v1 * 10
        expected = vec.Vector2d(10,20)
        self.assertEqual(actual, expected)
        
        v1 *= 10
        expected = vec.Vector2d(10,20)
        self.assertEqual(actual, expected)
        
        v1 /= 10
        expected = vec.Vector2d(1,2)
        self.assertEqual(v1, expected)
        
        actual = v1 / 10
        expected = vec.Vector2d(0.1,0.2)
        self.assertEqual(actual, expected)
        
        actual = -v1
        expected = vec.Vector2d(-1,-2)
        self.assertEqual(actual, expected)
        
    def test_vec3_simple_operations(self):
        print("\nVector3d simple operations", end = "")
        v1, v2 = vec.Vector3d(1,2,3), vec.Vector3d(3,2,1)
        
        actual = v1 + v2
        expected = vec.Vector3d(4)
        self.assertEqual(actual, expected)
        
        v1 += v2
        expected = vec.Vector3d(4)
        self.assertEqual(v1, expected)
        
        v1 -= v2
        expected = vec.Vector3d(1,2,3)
        self.assertEqual(v1, expected)
        
        actual = v1 - v2
        expected = vec.Vector3d(-2, 0, 2)
        self.assertEqual(actual, expected)
        
        actual = v1 * 10
        expected = vec.Vector3d(10,20,30)
        self.assertEqual(actual, expected)
        
        v1 *= 10
        expected = vec.Vector3d(10,20,30)
        self.assertEqual(actual, expected)
        
        v1 /= 10
        expected = vec.Vector3d(1,2,3)
        self.assertEqual(v1, expected)
        
        actual = v1 / 10
        expected = vec.Vector3d(0.1,0.2,0.3)
        self.assertEqual(actual, expected)
        
        actual = -v1
        expected = vec.Vector3d(-1,-2,-3)
        self.assertEqual(actual, expected)
        
    def test_vec4_simple_operations(self):
        print("\nVector4d simple operations", end = "")
        v1, v2 = vec.Vector4d(1,2,3,4), vec.Vector4d(4,3,2,1)
        
        actual = v1 + v2
        expected = vec.Vector4d(5)
        self.assertEqual(actual, expected)
        
        v1 += v2
        expected = vec.Vector4d(5)
        self.assertEqual(v1, expected)
        
        v1 -= v2
        expected = vec.Vector4d(1,2,3,4)
        self.assertEqual(v1, expected)
        
        actual = v1 - v2
        expected = vec.Vector4d(-3, -1, 1, 3)
        self.assertEqual(actual, expected)
        
        actual = v1 * 10
        expected = vec.Vector4d(10,20,30,40)
        self.assertEqual(actual, expected)
        
        v1 *= 10
        expected = vec.Vector4d(10,20,30,40)
        self.assertEqual(actual, expected)
        
        v1 /= 10
        expected = vec.Vector4d(1,2,3,4)
        self.assertEqual(v1, expected)
        
        actual = v1 / 10
        expected = vec.Vector4d(0.1,0.2,0.3,0.4)
        self.assertEqual(actual, expected)
        
        actual = -v1
        expected = vec.Vector4d(-1,-2,-3,-4)
        self.assertEqual(actual, expected)
        
    def test_vec2_vector_operations(self):
        print("\nVector2d vector operations", end = "")
        v1, v2 = vec.Vector2d(1,2),vec.Vector2d(2,1)
        
        actual1 = v1.dot(v2)
        actual2 = v2.dot(v1)
        expected = 4
        self.assertEqual(actual1,actual2)
        self.assertEqual(actual1,expected)
        
        actual = v1.sqrdistance(v2)
        expected = 2
        self.assertEqual(actual,expected)
        
        actual = v1.distance(v2)
        expected = math.sqrt(2)
        self.assertEqual(actual,expected)
        
        v3 = vec.Vector2d(3*math.cos(10), 3*math.sin(10))
        actual = v3.normalized()
        self.assertFalse(actual == v3)
        v3.normalize()
        self.assertTrue(actual == v3)
        expected = vec.Vector2d(math.cos(10),math.sin(10))
        self.assertEqual(round(actual.x,10),round(expected.x,10))
        self.assertEqual(round(actual.y,10),round(expected.y,10))
        
        v3 = vec.Vector2d(3,4)
        actual = v3.length()
        expected = 5
        self.assertEqual(actual, expected)
        
        actual = v3.sqrlength()
        expected = 25
        self.assertEqual(actual, expected)
        
    def test_vec3_vector_operations(self):
        print("\nVector3d vector operations", end = "")
        v1, v2 = vec.Vector3d(1,2,3),vec.Vector3d(3,2,1)
        
        actual1 = v1.dot(v2)
        actual2 = v2.dot(v1)
        expected = 10
        self.assertEqual(actual1,actual2)
        self.assertEqual(actual1,expected)
        
        actual = v1.sqrdistance(v2)
        expected = 8
        self.assertEqual(actual,expected)
        
        actual = v1.distance(v2)
        expected = math.sqrt(8)
        self.assertEqual(actual,expected)
        
        sqrt3 = math.sqrt(3)
        invsqrt3 = 1/sqrt3
        expected = vec.Vector3d(invsqrt3,invsqrt3,invsqrt3)
        v3 = vec.Vector3d(13,13,13)
        actual = v3.normalized()
        self.assertFalse(actual == v3)
        v3.normalize()
        self.assertTrue(actual == v3)
        self.assertEqual(round(actual.x,10),round(expected.x,10))
        self.assertEqual(round(actual.y,10),round(expected.y,10))
        self.assertEqual(round(actual.z,10),round(expected.z,10))
        
        v3 = vec.Vector3d(3,4,5)
        actual = v3.length()
        expected = math.sqrt(50)
        self.assertEqual(actual, expected)
        
        actual = v3.sqrlength()
        expected = 50
        self.assertEqual(actual, expected)
        
        v1,v2 = vec.Vector3d(2,3,4), vec.Vector3d(5,6,7)
        actual1 = v1.cross(v2)
        actual2 = v2.cross(v1)
        self.assertEqual(actual1,-actual2)
        expected = vec.Vector3d(-3,6,-3)
        self.assertEqual(actual1,expected)
        
    def test_vec4_vector_operations(self):
        print("\nVector4d vector operations", end = "")
        v1, v2 = vec.Vector4d(1,2,3,4),vec.Vector4d(4,3,2,1)
        
        actual1 = v1.dot(v2)
        actual2 = v2.dot(v1)
        expected = 20
        self.assertEqual(actual1,actual2)
        self.assertEqual(actual1,expected)
        
        actual = v1.sqrdistance(v2)
        expected = 20
        self.assertEqual(actual,expected)
        
        actual = v1.distance(v2)
        expected = math.sqrt(20)
        self.assertEqual(actual,expected)
        
        v3 = vec.Vector4d(12, 12, 12, 12)
        actual = v3.normalized()
        self.assertFalse(actual == v3)
        v3.normalize()
        self.assertTrue(actual == v3)
        expected = vec.Vector4d(0.5,0.5,0.5,0.5)
        self.assertEqual(round(actual.x,10),round(expected.x,10))
        self.assertEqual(round(actual.y,10),round(expected.y,10))
        self.assertEqual(round(actual.z,10),round(expected.z,10))
        self.assertEqual(round(actual.w,10),round(expected.w,10))
        
        actual = v1.length()
        expected = math.sqrt(30)
        self.assertEqual(actual, expected)
        
        actual = v1.sqrlength()
        expected = 30
        self.assertEqual(actual, expected)
        
    def test_vec2_bool_operations(self):
        print("\nVector2d bool operations", end = "")
        v1, v2 = vec.Vector2d(0.1), vec.Vector2d(0.1)
        
        self.assertTrue(v1 == v2)
        self.assertTrue(v1 <= v2)
        self.assertTrue(v1 >= v2)
        
        v2 = vec.Vector2d(0.101)
        
        self.assertTrue(v1 != v2)
        
        self.assertTrue(v1 < v2)
        self.assertTrue(v1 <= v2)
        
        self.assertTrue(v2 > v1)
        self.assertTrue(v2 >= v1)
        
    def test_vec3_bool_operations(self):
        print("\nVector3d bool operations", end = "")
        v1, v2 = vec.Vector3d(0.1), vec.Vector3d(0.1)
        
        self.assertTrue(v1 == v2)
        self.assertTrue(v1 <= v2)
        self.assertTrue(v1 >= v2)
        
        v2 = vec.Vector3d(0.101)
        
        self.assertTrue(v1 != v2)
        
        self.assertTrue(v1 < v2)
        self.assertTrue(v1 <= v2)
        
        self.assertTrue(v2 > v1)
        self.assertTrue(v2 >= v1)
        
    def test_vec4_bool_operations(self):
        print("\nVector4d bool operations", end = "")
        v1, v2 = vec.Vector4d(0.1), vec.Vector4d(0.1)
        
        self.assertTrue(v1 == v2)
        self.assertTrue(v1 <= v2)
        self.assertTrue(v1 >= v2)
        
        v2 = vec.Vector4d(0.101)
        
        self.assertTrue(v1 != v2)
        
        self.assertTrue(v1 < v2)
        self.assertTrue(v1 <= v2)
        
        self.assertTrue(v2 > v1)
        self.assertTrue(v2 >= v1)

class TestVectorProperties(unittest.TestCase):
    def test_vec2_properties(self):
        print("\nVector2d properties", end = "")
        vec2 = vec.Vector2d(1)
        
        self.assertEqual(vec2.x,1)
        vec2.x = 2
        self.assertEqual(vec2.x,2)
        
        self.assertEqual(vec2.y, 1)
        vec2.y = 3
        self.assertEqual(vec2.y, 3)
        
    def test_vec3_properties(self):
        print("\nVector3d properties", end = "")
        vec3 = vec.Vector3d(1)
        
        self.assertEqual(vec3.x,1)
        vec3.x = 2
        self.assertEqual(vec3.x,2)
        
        self.assertEqual(vec3.y, 1)
        vec3.y = 3
        self.assertEqual(vec3.y, 3)
        
        self.assertEqual(vec3.z, 1)
        vec3.z = 4
        self.assertEqual(vec3.z, 4)
        
    def test_vec4_properties(self):
        print("\nVector4d properties", end = "")
        vec4 = vec.Vector4d(1)
        
        self.assertEqual(vec4.x,1)
        vec4.x = 2
        self.assertEqual(vec4.x,2)
        
        self.assertEqual(vec4.y,1)
        vec4.y = 4
        self.assertEqual(vec4.y,4)
        
        self.assertEqual(vec4.z,1)
        vec4.z = -2
        self.assertEqual(vec4.z,-2)
        
        self.assertEqual(vec4.w,1)
        vec4.w = 154
        self.assertEqual(vec4.w,154)
        
if __name__ == "__main__":
    print("\n-------------------------")
    print("...Test Vector submodule...")
    unittest.main()
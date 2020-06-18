import unittest

from virtual import *

class TestVirtual(unittest.TestCase):
    def test_main(self):
        with self.assertRaises(TypeError):
            iobj = IObject()
            
        obj = Object("kek")
        self.assertEqual(obj.get(),"kek")
        
if __name__ == "__main__":
    unittest.main()
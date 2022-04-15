import time 
import unittest

from engine.Rendering import Image
from engine.Visual import Color

class TestImagePerformance(unittest.TestCase):
    def test_to_string(self):
        width = 800
        height = 600
        image = Image(width, height, Color(0xff0000ff))
        iterations = 1000
        start = time.time()
        for i in range(iterations):
            res = str(image)
        elapsed = time.time() - start
        print("\nImage to string {} iterations per {}s, {}IPS".format(iterations,round(elapsed,4),round(iterations/elapsed,4)), end = "")
    
    def test_raw_data(self):
        width = 800
        height = 600
        image = Image(width, height, Color(0xff0000ff))
        iterations = 1000
        start = time.time()
        for i in range(iterations):
            res = image.rawData
        elapsed = time.time() - start
        print("\nImage raw data {} iterations per {}s, {}IPS".format(iterations,round(elapsed,4),round(iterations/elapsed,4)), end = "")
        
    def test_data(self):
        width = 800
        height = 600
        image = Image(width, height, Color(0xff0000ff))
        iterations = 1000
        start = time.time()
        for i in range(iterations):
            res = image.data
        elapsed = time.time() - start
        print("\nImage data {} iterations per {}s, {}IPS".format(iterations,round(elapsed,4),round(iterations/elapsed,4)), end = "")
        
if __name__ == "__main__":
    print("\n--------------------------")
    print("...Test Image Performance...")
    unittest.main()
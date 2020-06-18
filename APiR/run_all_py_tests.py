import os
import sys
import subprocess
import json

def run(module="", performance = False):
    folder = os.walk(".")
    short_results = []
    for addr, dirs, files in folder:
        for file in files:
            if not module in addr:
                continue
            if not "_test.py" in file:
                continue
            if "performance" in file and not performance:
                continue
            res = file + ": "
            res += "OK" if subprocess.run(["python3", addr+"/"+file]).returncode == 0 else "Failed"
            short_results.append(res)
    
    if len(short_results) > 0:
        print("-------Short results-------")
        for res in short_results:
            print(res)
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            run(sys.argv[1],True)
        else: 
            run(sys.argv[1])
    else:
        run()

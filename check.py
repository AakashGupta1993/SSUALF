import os
import pathlib
result = pathlib.Path('data/data_road/testing').mkdir(parents=True, exist_ok=True) 
print(result)
'''if not os.path.isdir("data"):
  os.makedirs(data)
if not os.path.isdir("data/data_road"):
  os.makedirs(data/data_road)
if not os.path.isdir("data/data_road/testing"):
  os.makedirs(data/data_road/testing)'''
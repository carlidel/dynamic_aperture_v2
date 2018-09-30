import pickle
import numpy as np

print("load data")

data = pickle.load(open("revised_version_hennon.pkl", "rb"))

#%%
print("show data")

for epsilon in data:
    print(epsilon)
    for angle in data[epsilon]:
        print(angle)
        print(data[epsilon][angle])

        
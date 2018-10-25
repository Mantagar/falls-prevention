import os
import sys
import random

tra_factor = float(sys.argv[1])
tes_factor = float(sys.argv[2])
val_factor = float(sys.argv[3])
sum = tra_factor + tes_factor + val_factor
tra_factor /= sum
tes_factor /= sum
val_factor /= sum

loc = "./Processed data/Synkope/"
slist = os.listdir(loc)
slist = [ loc+i for i in slist ]
random.shuffle(slist)
first_idx = int(len(slist)*val_factor)
second_idx = first_idx + 1 + int(len(slist)*tes_factor)
val_data = slist[0:first_idx]
tes_data = slist[first_idx:second_idx]
tra_data = slist[second_idx:]

loc = "./Processed data/Nosynkope/"
nlist = os.listdir(loc)
nlist = [ loc+i for i in nlist ]
random.shuffle(nlist)
first_idx = int(len(nlist)*val_factor)
second_idx = first_idx + 1 + int(len(nlist)*tes_factor)
val_data += nlist[0:first_idx]
tes_data += nlist[first_idx:second_idx]
tra_data += nlist[second_idx:]

random.shuffle(val_data)
random.shuffle(tes_data)
random.shuffle(tra_data)

print("All:\t\t"+str(len(slist)+len(nlist)))
print("Training:\t"+str(len(tra_data)))
print(tra_data)
print("Test:\t\t"+str(len(tes_data)))
print(tes_data)
print("Validation:\t"+str(len(val_data)))
print(val_data)

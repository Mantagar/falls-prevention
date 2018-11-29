import os
import sys
import random

def saveListAsFile(list, path):
  with open(path, 'w') as file:
    for i in list:
      file.write(i+"\n")

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
slen = len(slist)
first_idx = int(slen*val_factor)
second_idx = first_idx + 1 + int(slen*tes_factor)
val_data = slist[0:first_idx]
tes_data = slist[first_idx:second_idx]
tra_data = slist[second_idx:]

loc = "./Processed data/Nosynkope/"
nlist = os.listdir(loc)
nlist = [ loc+i for i in nlist ]
random.shuffle(nlist)
nlen = len(nlist)
first_idx = int(nlen*val_factor)
second_idx = first_idx + 1 + int(nlen*tes_factor)
third_idx = second_idx + 1 + int(nlen*tra_factor)
val_data += nlist[0:first_idx]
tes_data += nlist[first_idx:second_idx]
tra_data += nlist[second_idx:third_idx]

random.shuffle(val_data)
random.shuffle(tes_data)
random.shuffle(tra_data)


tra_path = 'Processed data/training_set.txt'
tes_path = 'Processed data/test_set.txt'
val_path = 'Processed data/validation_set.txt'
print("Training:\t"+str(len(tra_data))+" - saved to '"+tra_path+"'")
saveListAsFile(tra_data, tra_path)
print("Test:\t\t"+str(len(tes_data))+" - saved to '"+tes_path+"'")
saveListAsFile(tes_data, tes_path)
print("Validation:\t"+str(len(val_data))+" - saved to '"+val_path+"'")
saveListAsFile(val_data, val_path)

from models import cnn_md
from utils import inf_md,concat
from keras.preprocessing.image import ImageDataGenerator
from math import floor
from multiprocessing import Pool
from datetime import datetime

import pandas as pd

p=Pool(8)

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--imgdir",dest="imgdir",nargs="?",
                    help="image directory,\
                         beware, it doesn't read\
                          any image form direct sub folder")

parser.add_argument("--weights", dest="weights", nargs='?',
                    help="path to model weights")

parser.add_argument("--log", dest="log",nargs='?',
                    default=None,
                    help="path to log file, \
                    if the path is not set,\
                     it will write a csv log\
                      under current directory",)

parser.add_argument("--ct_log", dest="ct_log",nargs='?',
                    default=None,
                    help="path to count log file, \
                    if the path is not set,\
                     it will write a csv log\
                      under current directory",)

parser.add_argument("--is_remove", dest="is_remove",nargs="?",
                    default="1",
                    help="'1' or '0' if choose '1' \
                    it will remove the \
                    similar images, the remaining one is randomly picked",
                    )

parser.add_argument("--batch_size", dest="batch_size",nargs='?',
                    default=128,
                    help="batch size for processing image",)

args = parser.parse_args()

print("="*60)
print("Constructing Model")
cnn_md = cnn_md(IMAGE_SCALE=108)

print("="*60)
print("Loading weights from %s"%(args.weights))
print("="*60)

cnn_md.load_weights(args.weights)

print(cnn_md.summary())

inf = inf_md(cnn_md)

gen_set = ImageDataGenerator(rotation_range=0,
                           width_shift_range=0,
                           height_shift_range=0
                          )
gen = gen_set.flow_from_directory(args.imgdir,shuffle=False,target_size=(108,108),batch_size=args.batch_size)

fnames = gen.filenames

print("%s files in total"%(len(fnames)))

result = inf.predict_generator(gen,steps=int(floor(len(fnames)/gen.batch_size)+1))
strlist = p.map(concat, result.tolist())

hashdf = pd.DataFrame({"fname":list(gen.directory + i for i in gen.filenames[:len(strlist)]),
                       "hash":strlist},
                      )

nowstamp = datetime.now().strftime("%s")

if not args.log:
    log = "hash_df_%s.csv"%(nowstamp)
else:
    log = args.log

if not args.ct_log:
    ct_log = "hash_count_%s.csv"%(nowstamp)
else:
    ct_log = args.ct_log

hashdf.to_csv(log,index=False)
print("Filename to hashing mapping saved to %s"%(log))

hs_u = list(hashdf["hash"].value_counts().reset_index()["index"])
vc = pd.DataFrame(hashdf["hash"].value_counts().reset_index().rename(columns={"hash":"hexcount","index":"hash"}))
hct = pd.merge(hashdf,vc,on="hash").sort_values(by=["hexcount","hash"],ascending=False)

hct.to_csv(ct_log,index=False)
print("Hash value count saved to %s"%(ct_log))
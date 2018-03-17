import numpy as np
from math import floor
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
import keras.backend as K
from keras.metrics import mae


class sim_gen:
    """
    An image generator for training
    Created to fit the sim_loss function
    """
    def __init__(self, gen_s0, gen_sv, gen_df):
        self.gen_s0 = gen_s0
        self.gen_sv = gen_sv
        self.gen_df = gen_df
        self.batch_size = self.gen_df.batch_size
        self.batch_size_h = self.gen_s0.batch_size
        if self.check_same():
            print("Similarity data file list aligned")
        else:
            print("Warning!: the similarity data file list are not the same")
        self.y = np.zeros(self.batch_size)
        self.s = np.concatenate((np.zeros(self.batch_size_h), np.ones(self.batch_size_h)))

    def check_same(self):
        """
        check if the file list are the same for 2 generators
        :return: Bool value
        """
        return self.gen_s0.filenames == self.gen_sv.filenames

    def __next__(self):
        s0, sv = self.gen_s0.next(), self.gen_sv.next()
        df_ = self.gen_df.next()
        df_l = floor(len(df_) / 2)
        df1, df2 = df_[:df_l], df_[df_l:df_l * 2]
        w1 = np.vstack((s0, df1))
        w2 = np.vstack((sv, df2))
        sample_len = len(w1)
        return dict({"w1": w1, "w2": w2, "s": self.s[:sample_len]}), self.y[:sample_len]


def sim_loss(paramlist):
    """
    Define the loss funciton, response to the switch of is similar or not
    h1, h2: hash1, hash2
    s: 0 for similar image, 1 indicating different image
    """
    [h1, h2, s] = paramlist
    metric = mae(h1, h2)
    return s * (1 - K.log(metric)) + (s - 1) * ((-1) * metric)


def train_md(hash_md,IMG_SCALE=108):
    """
    To wrap a model with the sim_loss function into the inference model.
    :param hash_md: hash model, that we can later use for inference
    :param IMG_SCALE: Same value set for height & width
    :return:tuple, training model and the inference model
    """
    w1_ipt = Input(shape=(IMG_SCALE, IMG_SCALE, 3), name="w1")
    w2_ipt = Input(shape=(IMG_SCALE, IMG_SCALE, 3), name="w2")
    f_ = hash_md()
    s_ipt = Input(shape=(1,), name="s")

    x = Lambda(sim_loss)([f_(w1_ipt), f_(w2_ipt), s_ipt])
    train_md = Model(inputs=[w1_ipt, w2_ipt, s_ipt], outputs=x, name="train_model")
    train_md.compile(loss=lambda y_t, y_p: y_p, optimizer=Adam())
    return train_md, f_
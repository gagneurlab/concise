from concise.preprocessing import encodeDNA
import pandas as pd
import numpy as np
import os

DATA_DIR = "../../data/RBP/"

def data(seq_length=101):

    def load(split="train"):
        dt = pd.read_csv(DATA_DIR + "/PUM2_{0}.csv".format(split))
        # DNA/RNA sequence
        xseq = encodeDNA(dt.seq, maxlen=seq_length, seq_align='center')
        # response variable
        y = dt.binding_site.as_matrix().reshape((-1, 1)).astype("float")
        if split == "train":
            from concise.data import attract
            # add also the pwm_list
            pwm_list = attract.get_pwm_list(["129"])
            return {"seq": xseq}, y, pwm_list
        else:
            return {"seq": xseq}, y

    return load("train"), load("valid"), load("test")

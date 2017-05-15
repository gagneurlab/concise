import h5py
from keras.models import model_from_json
from tests.setup_concise_load_data import load_example_data
from concise.models import single_layer_pos_effect as concise_model
import numpy as np


def get_concise_model():
    param, X_feat, X_seq, y, id_vec = load_example_data(trim_seq_len = 1000)
    dc = concise_model(pooling_layer="sum",
                       init_motifs=["TGCGAT", "TATTTAT"],
                       n_splines=10,
                       n_covariates=0,
                       seq_length=X_seq.shape[1],
                       **param)

    dc.fit([X_seq], y, epochs=1,
           validation_data=([X_seq], y))

    return {"model": dc, "out_annotation": np.array(["output_1"])}

# Requires: A model that can be tested
# An input dataset that can be tested

def get_model():
    import socket
    if not "ebi" in socket.gethostname():
        raise Exception("Only available on EBI cluster!")
    model_path = "/nfs/research2/stegle/users/rkreuzhu/deeplearning/model/results_%d_narrow_%s_allDO" % (921, "hg19")
    preds_fh = h5py.File(model_path + "/pred.hdf5", "r")
    output_labels = preds_fh['out_annotation'].value
    preds_fh.close()
    #
    # k_backend.set_learning_phase(0)
    #
    with open(model_path + "/a.json", "rt") as f:
        model = model_from_json(f.read())
    model.load_weights(model_path + "/best_w.h5")
    return {"model":model, "out_annotation":output_labels}

def get_example_data():
    dataset_path = "./data/sample_hqtl_res.hdf5"
    ifh = h5py.File(dataset_path, "r")
    dataset = {}
    ref = ifh["test_in_ref"].value
    alt = ifh["test_in_alt"].value
    dirs = ifh["test_out"]["seq_direction"].value
    assert(dirs[0] == b"fwd")
    dataset["ref"] = ref[::2,...]
    dataset["alt"] = alt[::2,...]
    dataset["ref_rc"] = ref[1::2,...]
    dataset["alt_rc"] = alt[1::2,...]
    dataset["y"] = ifh["test_out"]["type"].value[::2]
    dataset["mutation_position"] = np.array([500]*dataset["ref"].shape[0])
    ifh.close()
    return dataset




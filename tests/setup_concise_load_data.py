import concise
import pandas as pd
# import os
# dir_root = os.path.dirname(os.path.realpath(__file__)) + "/../../../../"

def load_example_data(trim_seq_len=200):
    param = {}
    # column names
    csv_file_path = "./data/pombe_half-life_UTR3.csv"
    param['features'] = ["UTR3_length", "UTR5_length", "TATTTAT", "ACTAAT", "TTAATGA"]
    param['seq_align'] = "end"
    param['n_motifs'] = 1
    param['trim_seq_len'] = trim_seq_len
    response = "hlt"
    sequence = "seq"
    id_column = "ID"                # unique identifier
    ############################################
    # read the csv + set the index appropriate column (transcript name)
    dt = pd.read_csv(csv_file_path)
    X_feat, X_seq, y, id_vec = concise.prepare_data(dt,
                                                    features=param['features'],
                                                    response=response,
                                                    sequence=sequence,
                                                    id_column=id_column,
                                                    seq_align=param['seq_align'],
                                                    trim_seq_len=param['trim_seq_len']
                                                    )
    return param, X_feat, X_seq, y, id_vec

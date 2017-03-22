from concise.concise_keras import concise_model
from keras.models import model_from_json

def test_serialization():
    c = concise_model(init_motifs=["TAATA", "TGCGAT"],
                      pooling_layer="sum",
                      n_splines=10,
                      )
    js = c.to_json()
    a = model_from_json(js)

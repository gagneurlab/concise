"""
test_concise_keras
----------------------------------

Tests for `concise_keras` module
"""
import keras
from keras.models import model_from_json
from concise.legacy.models import single_layer_pos_effect as concise_model
from concise.eval_metrics import mse
from sklearn.linear_model import LinearRegression

import pytest
from tests.setup_concise_load_data import load_example_data
import numpy as np


def test_serialization():
    c = concise_model(init_motifs=["TAATA", "TGCGAT"],
                      pooling_layer="sum",
                      n_splines=10,
                      )
    js = c.to_json()
    assert isinstance(model_from_json(js), keras.models.Model)


def test_serialization_disk(tmpdir):
    param, X_feat, X_seq, y, id_vec = load_example_data()
    dc = concise_model(pooling_layer="sum",
                       init_motifs=["TGCGAT", "TATTTAT"],
                       n_splines=10,
                       n_covariates=X_feat.shape[1],
                       seq_length=X_seq.shape[1],
                       **param)

    dc.fit([X_seq, X_feat], y, epochs=1,
           validation_data=([X_seq, X_feat], y))

    fn = tmpdir.mkdir('data').join('test_keras.h5')

    dc.save(str(fn))
    dc = keras.models.load_model(str(fn))
    assert isinstance(dc, keras.models.Model)


class TestKerasConciseBasic(object):

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data()
        # pass

    def test_no_error(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data
        dc = concise_model(pooling_layer="max",
                           n_covariates=X_feat.shape[1],
                           seq_length=X_seq.shape[1],
                           **param)
        dc.fit([X_seq, X_feat], y, epochs=1,
               validation_data=([X_seq, X_feat], y))

        y_pred = dc.predict([X_seq, X_feat])
        y_pred

    def test_train_predict_no_X_feat(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data
        dc = concise_model(pooling_layer="max",
                           n_covariates=0,
                           seq_length=X_seq.shape[1],
                           **param)
        dc.fit(X_seq, y, epochs=1,
               validation_data=(X_seq, y))

        y_pred = dc.predict(X_seq)
        y_pred

    @classmethod
    def teardown_class(cls):
        pass


class TestMultiTaskLearning(TestKerasConciseBasic):
    """
    Test multi-task learning
    """

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(num_tasks=3)

class TestConcisePrediction(object):

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(trim_seq_len=1, standardize_features=False)
        cls.data[0]["n_motifs"] = 1
        cls.data[0]["motif_length"] = 1
        cls.data[0]["step_size"] = 0.001
        cls.data[0]["early_stop_patience"] = 3

    def test_non_std(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data

        dc = concise_model(pooling_layer="max",
                           n_covariates=X_feat.shape[1],
                           lambd=0,
                           seq_length=X_seq.shape[1],
                           **param)

        callback = keras.callbacks.EarlyStopping(patience=param["early_stop_patience"])
        dc.fit([X_seq, X_feat], y, epochs=50,
               callbacks=[callback],
               validation_data=([X_seq, X_feat], y))

        dc_coef = dc.layers[-1].get_weights()[0][-X_feat.shape[1]:, 0]
        lm = LinearRegression()
        lm.fit(X_feat, y)

        # np.allclose(lm.coef_, dc_coef, atol=0.02)

        # # weights has to be the same as for linear regression
        # (dc_coef - lm.coef_) / lm.coef_

        # they both have to predict the same
        y_pred = dc.predict([X_seq, X_feat])
        mse_lm = mse(y, lm.predict(X_feat))
        mse_dc = mse(y, y_pred)
        print("mse_lm")
        print(mse_lm)
        print("mse_dc")
        print(mse_dc)
        assert mse_dc < mse_lm + 0.01

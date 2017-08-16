import numpy as np
from hyperopt import fmin, tpe, hp
import data
import model
from concise.hyopt import test_fn, CompileFN, CMongoTrials


hyper_params = {
    "data": {
        "seq_length": hp.choice("d_seq_length", (101, 51, 21))
    },
    "model": {
        "filters": hp.choice("m_filters", (1, 16)),
        "kernel_size": 15,
        "motif_init": hp.choice("m_motif_init", (
            None,
            {"stddev": hp.uniform("m_stddev", 0, 0.2)}
        )),
        "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2))  # 0.0001 - 0.01
    },
    "fit": {
        "epochs": 50,
        "patience": 5,
        "batch_size": 128,
    }
}

objective = CompileFN(db_name="hyopt_example", exp_name="motif_initialization",  # experiment name
                      data_fn=data.data,
                      model_fn=model.model,
                      add_eval_metrics=["auprc", "auc"],  # metrics from concise.eval_metrics, you can also use your own
                      loss_metric="auprc",  # which metric to optimize for
                      loss_metric_mode="max",  # maximum should be searched for
                      valid_split=None,  # use valid from the data function
                      save_model='best',  # checkpoint the best model
                      save_results=True,  # save the results as .json (in addition to mongoDB)
                      save_dir="./saved_models")  # place to store the models

# ---------------------
# Test if the objective is working as expected
test_fn(objective, hyper_params)

# ---------------------
# Run the optimization


# handle to the database
trials = CMongoTrials(db_name="hyopt_example", exp_name="motif_initialization", ip="ouga03")


best = fmin(objective,
            space=hyper_params,
            algo=tpe.suggest,
            trials=trials,
            max_evals=300)

print("Done!")

# Changelog

## 0.1.0 (2016-09-15)

* First release on PyPI.

## 0.1.1 (2016-09-17)

* Minor documentation changes
* Renamed some internal variables

## 0.2.0 (2016-09-21)

* Introduced new feature: regress_out_feat
* Major renaming of variables for concistency

## 0.3.0 (2016-11-30)

* Added L-BFGS optimizer in addition to Adam. Use optimizer="lbfgs" in Concise()

## 0.3.1 (2016-11-30)

* New function: :code:`best_kmers` for motif efficient initialization

## 0.4.0 (2017-02-07)

* refactor: Removed regress_out feature
* feature: multi-task learning

## 0.4.1 (2017-02-09)

* bugfix: multi-task learning

## 0.4.2 (2017-02-09)

* same as 0.4.1 (pypi upload failed for 0.4.1)
  
## 0.4.3 (2017-02-09)

* feat: added early_stop_patience argument

  
## 0.4.4 (2017-02-10)

* fix: When X_feat had 0 columns, loading its weights from file was failing.
* feat: When training the global model in ConciseCV, use the average number of epochs yielding the best validation-set accuracy.

## 0.4.5 SNAPSHOT 

* fix: Update tensorflow function (tf.op_scope -> tf.name_scope, initialize_all_variables -> tf.global_variables_initializer)
  

## 0.6.0 (2017-07-16)

- Complete re-write. Now moved completely to Keras from pure TensorFlow.

## 0.6.1 (2017-07-16)

- fix: required version keras>=2.0.2

## 0.6.2 (2017-08-17)
## 0.6.3 (2017-08-16)

- added more documentation
- ipynb -> docs compilation
- hocomoco motif database added to `concise.data.hocomoco`
- added utility function for position extraction
- fix: hyopt and memoization of data() now works
- implemented seqlogo plot

## 0.6.4 (2017-08-29)

- Added more documentation
- hyopt.CompileFN: loss_metric -> optim_metric
- eval_metrics: y,z -> y_true, y_pred arguments
- speedup encodeSequences by 4x

## 0.6.5 (2018-01-25)

- Added `gtfparse` dependency for parsing the gtf files

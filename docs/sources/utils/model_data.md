### test_len


```python
test_len(train)
```


Test if all the elements in the training have the same shape[0]

----

### split_train_test_idx


```python
split_train_test_idx(train, valid_split=0.2, stratified=False, random_state=None)
```


Return indicies for train-test split

----

### split_KFold_idx


```python
split_KFold_idx(train, cv_n_folds=5, stratified=False, random_state=None)
```


Get k-fold indices generator

----

### subset


```python
subset(train, idx, keep_other=True)
```


Subset the (train, test) data tuple, each of the form:
- list, np.ndarray
- tuple, np.ndarray
- dictionary, np.ndarray
- np.ndarray, np.ndarray

In case there are other data present in the tuple:
(train, test, other1, other2, ...), these get passed on as:
(train_sub, test_sub, other1, other2)

idx = indices to subset the data with

Further fields are ignored

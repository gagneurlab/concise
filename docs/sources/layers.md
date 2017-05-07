<span style="float:right;">[[source]](https://github.com/avsecz/concise/blob/master/concise/layers.py#L56)</span>
### GlobalSumPooling1D

```python
keras.layers.pooling.GlobalSumPooling1D()
```

Global average pooling operation for temporal data.
__Input shape__

3D tensor with shape: `(batch_size, steps, features)`.
__Output shape__

2D tensor with shape:
`(batch_size, channels)`

----

<span style="float:right;">[[source]](https://github.com/avsecz/concise/blob/master/concise/layers.py#L69)</span>
### ConvDNA

```python
concise.layers.ConvDNA(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, seq_length=None, background_probs=None)
```


Convenience wrapper over keras.layers.Conv1D with 2 changes:
- additional argument seq_length specifying input_shape
- restriction in build method: input_shape[-1] needs to be 4

----

<span style="float:right;">[[source]](https://github.com/avsecz/concise/blob/master/concise/layers.py#L186)</span>
### GAMSmooth

```python
concise.layers.GAMSmooth(n_bases=10, spline_order=3, share_splines=False, spline_exp=False, l2_smooth=1e-05, l2=1e-05, use_bias=False, bias_initializer='zeros')
```

----

<span style="float:right;">[[source]](https://github.com/avsecz/concise/blob/master/concise/layers.py#L320)</span>
### ConvDNAQuantitySplines

```python
concise.layers.ConvDNAQuantitySplines(filters, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=<concise.regularizers.GAMRegularizer object at 0x2b38ee396518>, bias_regularizer=None, kernel_constraint=None, bias_constraint=None, activity_regularizer=None)
```


Convenience wrapper over keras.layers.Conv1D with 2 changes:
- additional argument seq_length specifying input_shape (as in ConvDNA)
- restriction in kernel_regularizer - needs to be of class GAMRegularizer
- hard-coded values:
   - kernel_size=1,
   - strides=1,
   - padding='valid',
   - dilation_rate=1,

----

### InputDNA


```python
InputDNA(seq_length, name=None)
```


Convenience wrapper around keras.layers.Input:

Input((seq_length, 4), name=name, **kwargs)

----

### InputRNAStructure


```python
InputRNAStructure(seq_length, name=None)
```


Convenience wrapper around keras.layers.Input:

Input((seq_length, 5), name=name, **kwargs)

----

### InputDNAQuantity


```python
InputDNAQuantity(seq_length, n_features=1, name=None)
```


Convenience wrapper around keras.layers.Input:

Input((seq_length, n_features), name=name, **kwargs)

----

### InputDNAQuantitySplines


```python
InputDNAQuantitySplines(seq_length, n_bases=10, name='DNASmoothPosition')
```


Convenience wrapper around keras.layers.Input:

Input((seq_length, n_bases), name=name, **kwargs)

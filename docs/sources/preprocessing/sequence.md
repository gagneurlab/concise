### encodeDNA


```python
encodeDNA(seq_vec, maxlen=None, seq_align='start')
```



Convert the DNA sequence to 1-hot-encoding numpy array

parameters
----------
- __seq_vec__: list of chars
List of sequences that can have different lengths

- __seq_align__: character; 'end' or 'start'
To which end should we align sequences?

- __maxlen__: int or None,
Should we trim (subset) the resulting sequence. If None don't trim.
Note that trims wrt the align parameter.
It should be smaller than the longest sequence.

returns
-------
3D numpy array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)

Examples
--------
>>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
>>> X_seq = encodeDNA(sequence_vec, align="end", maxlen=8)
>>> X_seq.shape
(2, 8, 4)

>>> print(X_seq)
 [[[ 0.  0.  0.  1.]
   [ 1.  0.  0.  0.]
   [ 0.  1.  0.  0.]
   [ 0.  0.  0.  1.]
   [ 0.  1.  0.  0.]
   [ 1.  0.  0.  0.]
   [ 0.  0.  1.  0.]
   [ 1.  0.  0.  0.]] 


  [[ 0.  0.  0.  0.]
   [ 0.  0.  0.  0.]
   [ 0.  0.  0.  1.]
   [ 0.  1.  0.  0.]
   [ 0.  0.  0.  1.]
   [ 0.  0.  0.  1.]
   [ 0.  0.  0.  1.]
   [ 1.  0.  0.  0.]]]]

----

### encodeRNA


```python
encodeRNA(seq_vec, maxlen=None, seq_align='start')
```

----

### encodeCodon


```python
encodeCodon(seq_vec, ignore_stop_codons=True, maxlen=None, seq_align='start', encode_type='one_hot')
```

----

### encodeAA


```python
encodeAA(seq_vec, maxlen=None, seq_align='start', encode_type='one_hot')
```

----

### pad_sequences


```python
pad_sequences(sequence_vec, maxlen=None, align='end', value='N')
```



See also: https://keras.io/preprocessing/sequence/

1. Pad the sequence with N's or any other sequence element
2. Subset the sequence

Aplicable also for lists of characters

parameters
----------
- __sequence_vec__: list of chars
List of sequences that can have different lengths
- __value__:
Neutral element to pad the sequence with
- __maxlen__: int or None,
Should we trim (subset) the resulting sequence. If None don't trim.
Note that trims wrt the align parameter.
It should be smaller than the longest sequence.
- __align__: character; 'end' or 'start'
To which end should to align the sequences.

Returns
-------
List of sequences of the same class as sequence_vec

Examples
--------
>>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
>>> pad_sequences(sequence_vec, "N", 10, "start")

----

### encodeSequence


```python
encodeSequence(seq_vec, vocab, neutral_vocab, maxlen=None, seq_align='start', pad_value='N', encode_type='one_hot')
```


Convert the sequence to one-hot-encoding.

## Arguments
   - __seq_vec__: list of sequences
   - __vocab__: list of chars: List of "words" to use as the vocabulary. Can be strings of length>0,
but all need to have the same length. For DNA, this is: ["A", "C", "G", "T"]
   - __neutral_vocab__: list of chars: Values used to pad the sequence or represent unknown-values. For DNA, this is: ["N"].
   maxlen, seq_align: see pad_sequences
   - __encode_type__: "one_hot" or "token". "token" represents each vocab element as a positive integer from 1 to len(vocab) + 1.
	  neutral_vocab is represented with 0.

## Returns
   Array with shape for encode_type:
 - "one_hot": (len(seq_vec), maxlen, len(vocab))
 - "token": (len(seq_vec), maxlen)
  If maxlen is None, it gets the value of the longest sequence length from seq_vec.

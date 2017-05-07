### encodeRNAStructure


```python
encodeRNAStructure(seq_vec, maxlen=None, seq_align='start', W=240, L=160, U=1, tmpdir='/tmp/RNAplfold/')
```



- __Arguments__:
   W, Int: span - window length
   L, Int, maxiumm span
   U, Int, size of unpaired region

- __Recomendation__:
- for human, mouse use W, L, u : 240, 160, 1
- for fly, yeast   use W, L, u :  80,  40, 1


### write_fasta


```python
write_fasta(file_path, seq_list, name_list=None)
```


Write fasta to file

----

### iter_fasta


```python
iter_fasta(file_path)
```


Returns an iterator over the fasta file


modified from Brent Pedersen
Correct Way To Parse A Fasta File In Python
given a fasta file. yield tuples of header, sequence

- ____Usage__:__

> fasta = fasta_iter("hg19.fa")
> for header, seq in fasta:
>   print(header)

----

### read_fasta


```python
read_fasta(file_path)
```


Read the fasta file as ordered dictionary

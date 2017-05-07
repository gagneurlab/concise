### get_metadata


```python
get_metadata()
```



Get pandas.DataFrame with metadata about the PWM's.

- __Columns__:
PWM_id (id of the PWM - pass to get_pwm_list() for getting the pwm
Gene_name
Gene_id
Mutated	(if the target gene is mutated)
Organism
Motif (concsensus motif)
Len	(lenght of the motif)
Experiment_description(when available)
Database (Database from where the motifs were extracted PDB: Protein data bank, C: Cisbp-RNA, R:RBPDB, S: Spliceaid-F, AEDB:ASD)
Pubmed (pubmed ID)
Experiment (type of experiment; short description)
Family (domain)
Score (Qscore refer to the paper)

----

### get_pwm_list


```python
get_pwm_list(pwm_id_list, pseudocountProb=0.0001)
```

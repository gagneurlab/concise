This folder contains the modified RNAplfold scripts, please see http://www.tbi.univie.ac.at/RNA/RNAplfold.html for more information on RNAplfold program. The original RNAplfold script only outputs probabilities for unpaired context, we have modified it so that it outputs probabilities for external region, hairpin, internal loop and multiloop seperately. 

Here is an example run to calculate profiles for external region, hairpin, internal loop and multiloop contexts:

 E_RNAplfold -W 240 -L 160 -u 1 <input.fasta >E_profile.txt 
 H_RNAplfold -W 240 -L 160 -u 1 <input.fasta >H_profile.txt 
 I_RNAplfold -W 240 -L 160 -u 1 <input.fasta >I_profile.txt 
 M_RNAplfold -W 240 -L 160 -u 1 <input.fasta >M_profile.txt 

-W is the window length.
-L is the maximum span.
-u is the width.

 We use -W 80 -L 40 for fly and yeast and -W 240 -L 160 for mouse and human. 

Once you run the commands above you have to combine the four output files into one.

We have a python code named combine_letter_profiles.py to do this. Here's how you run it:

python combine_letter_profiles.py E_profile.txt H_profile.txt I_profile.txt M_profile.txt 1 combined_profile.txt 

combined_profile.txt is in the following format:
>seq 1
probabilities for pairedness
probabilities for being in a hairpin loop
probabilities for being in a internal loop
probabilities for being in a multi loop
probabilities for being in a external region
>seq 2
...


After the profiles are calculated, RNAcontext can be run with the alphabet -PHIME.



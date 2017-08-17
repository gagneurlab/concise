"""Test visualization functionality
"""
from deeplift.visualization import viz_sequence
from concise.data import attract, encode
import matplotlib.pyplot as plt
import numpy as np
from concise.utils.plot import seqlogo


# TODO - allways convert things to string
attract.get_metadata()
attract.get_pwm_list([str(1)])[0].plotPWM()

encode.get_metadata()
pwm_elem = encode.get_pwm_list(["AFP_1"])[0]
pwm = pwm_elem.pwm
pssm = pwm_elem.get_pssm()
pwm2 = np.log(pwm / 0.25)
import dragonn

from dragonn import plot

dragonn.plot.plot_bases(pwm)
dragonn.plot.plot_pwm(pwm)
dragonn.plot.plot_bases_on_ax(pwm)
dragonn.plot.plot_pwm(pwm)

plt.figure()
plt.subplot(2, 1, 1)
dragonn.plot.plot_bases(pssm, figsize=(10, 2))


pwm_elem.plotPSSM()
plt.subplot(2, 1, 2)
attract.get_pwm_list([1])[0].plotPSSM()
plt.show()


dragonn.plot.letters_polygons['A']

ax = plt.gca()


from shapely.wkt import loads as load_wkt
from shapely import affinity

dragonn.plot.add_letter_to_axis(ax, 'A',
                                0, 0, 0.1)

import string


ax.show()
plt.show()

letters_polygons = {}
for let in string.ascii_uppercase:
    print("letter")
    try:
        let_std = dragonn.plot.standardize_polygons_str(globals()[let])
        letters_polygons[let] = let_std
    except:
        print("unsuccessfull parsing. letter: " + let)

# A, D, O, P


dragonn.plot.standardize_polygons_str(O)

ax = plt.gca()
add_letter_to_axis(ax, "P", 0, 0, 1)

plt.show()

all_polygons = {l: globals()[l] for l in string.ascii_uppercase}


seqlogo(pwm, "DNA")
plt.show()

seqlogo(pwm, "RNA")
plt.show()

seqlogo(pwm, "AA")
plt.show()


aa_matrix = np.random.uniform(0, 1, (10, len(VOCAB_colors["AA"])))
seqlogo(aa_matrix, "AA")
plt.show()

aa_matrix = np.random.uniform(-1, 1, (10, len(VOCAB_colors["AA"])))
seqlogo(aa_matrix, "AA")
plt.show()


aa_matrix = np.random.uniform(0, 1, (10, len(VOCAB_colors["DNA"])))
seqlogo(aa_matrix, "DNA")
plt.show()

aa_matrix = np.random.uniform(-1, 1, (10, len(VOCAB_colors["DNA"])))
seqlogo(aa_matrix, "DNA")
plt.show()


aa_matrix = np.random.uniform(0, 1, (10, len(VOCAB_colors["RNAStruct"])))
seqlogo(aa_matrix, "RNAStruct")
plt.show()

seqlogo(aa_matrix, "AA")
plt.show()
# TODO - somehow test this code?

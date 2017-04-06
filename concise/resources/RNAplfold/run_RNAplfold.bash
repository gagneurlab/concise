#!/bin/bash

# run RNAplfold for a fasta file

# REQUIREMENTS: python2 needs to be installed on your computer

# TODO implement a reader of this file in my deepcis model and in R
# TODO generate a fasta file of 3' UTR sequence
# 
# input:
# - fasta_file
# - output_file.txt
# -W is the window length. (80)
# -L is the maximum span.
# -u is the width.
#
# output_format - PHIME
# combined_profile.txt is in the following format:
# >seq 1
# probabilities for pairedness (P)
# probabilities for being in a hairpin loop (H)
# probabilities for being in a internal loop (I)
# probabilities for being in a multi loop (M)
# probabilities for being in a external region (E)
# >seq 2
# ...

if [[ $# -eq 2 ]] ; then
    input_fasta=$1
    output_txt_file=$2
    W=240
    L=160
    U=1
    echo "using defalt W = $W, L = $L and u = $U"
elif [[ $# -eq 5 ]] ; then
    input_fasta=$1
    output_txt_file=$2
    W=$3
    L=$4
    U=$5
else
    echo 'Usage: run_RNAplfold.bash <input.fasta> <output_file.txt> [<W-window length(240)> <L-maximum span (160)> <u-width (1)>]'
    echo 'for human, mouse use W, L, u : 240, 160, 1'
    echo 'for fly, yeast   use W, L, u :  80,  40, 1'
    exit 0
fi

src_dir="$(dirname $(readlink -f $0))"
# # if we are running from slurm, it can be problematic:
# # workaround:
# # 1. assumption - we have gagneurlab_shared/bin in our path
# if [[ ! -f $SCRIPTDIR/bam2fastq.bash ]]; then
#     SCRIPTDIR="$(dirname $(readlink -f $(which bam2kallisto_count_table)))"
# fi
# # 2. assumption - we are running the script from gagneurlab_shared/ directory root
# if [[ ! -f $SCRIPTDIR/bam2fastq.bash ]]; then
#     SCRIPTDIR="./bash/isoform_expression_with_kallisto"
# fi

tmpdir=/tmp/RNAplfold_tmp
mkdir -p $tmpdir
echo "$(date +%T) : running E_RNAplfold"
$src_dir/E_RNAplfold -W $W -L $L -u $U <${input_fasta} >$tmpdir/E_profile.txt 
echo "$(date +%T) : running H_RNAplfold"
$src_dir/H_RNAplfold -W $W -L $L -u $U <${input_fasta} >$tmpdir/H_profile.txt 
echo "$(date +%T) : running I_RNAplfold"
$src_dir/I_RNAplfold -W $W -L $L -u $U <${input_fasta} >$tmpdir/I_profile.txt 
echo "$(date +%T) : running M_RNAplfold"
$src_dir/M_RNAplfold -W $W -L $L -u $U <${input_fasta} >$tmpdir/M_profile.txt 

echo "$(date +%T) : running combine_letter_profiles.py"
python2 $src_dir/combine_letter_profiles.py \
       $tmpdir/E_profile.txt $tmpdir/H_profile.txt $tmpdir/I_profile.txt $tmpdir/M_profile.txt 1 ${output_txt_file}

# remove the intermediate files
rm -r /tmp/RNAplfold_tmp

echo "execution successful: $(date +%T)"

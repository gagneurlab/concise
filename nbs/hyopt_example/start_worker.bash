#!/bin/bash
#

#SBATCH --mem=16G
#SBATCH -c 4

DB=hyopt_example

# path to the repository containing the model.py, data.py, train.py scripts
export PYTHONPATH=$PWD

echo "current folder: $PWD"

hyperopt-mongo-worker \
    --mongo=ouga03:1234/$DB \
    --poll-interval=1 \
    --workdir='.' \
    --reserve-timeout=120


# --mongo=ouga03:1234/$DB - IP:PORT connection to MonogoDB and the used MonogDB database
# --poll-interval=1  - check for jobs every second
# --reserve-timeout=120 - terminate if no jobs were assigned after 120 seconds

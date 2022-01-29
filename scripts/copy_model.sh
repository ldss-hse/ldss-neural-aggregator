# make the script verbose
set -x

WHERE_TO_KEEP=~/Downloads/new_model/
SERVER_IP=46.148.228.49
TRAINING_STEP=830000

rm -rf ${WHERE_TO_KEEP}
mkdir -p ${WHERE_TO_KEEP}

scp -i ~/.ssh/id_rsa_ya -r root@${SERVER_IP}:~/NeuralTuringMachine/models/${TRAINING_STEP}/ ${WHERE_TO_KEEP}

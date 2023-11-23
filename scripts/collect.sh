GAME=$1
TASK=$2
SEED=$3
NAME=$4

python train_collect.py \
    --domain_name ${GAME} \
    --task_name ${TASK} \
    --encoder_type identity \
    --decoder_type identity \
    --action_repeat 4 \
    --save_buffer \
    --work_dir log/${NAME} \
    --seed ${SEED} 

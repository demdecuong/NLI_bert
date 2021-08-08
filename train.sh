export lr=3e-5
export c=0.6
export s=100
export tr_bs=16
export dev_bs=16
echo "${lr}"
export MODEL_DIR=JointBERT-CRF_XLMRencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python main.py --token_level word-level \
                  --model_type xlmr \
                  --model_dir $MODEL_DIR \
                  --data_dir data \
                  --seed $s \
                  --do_train \
                  --do_eval_dev \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs 50 \
                  --tuning_metric mean_intent_slot \
                  --gpu_id 0 \
                  --use_r3f \
                  --embedding_type soft \
                  --intent_loss_coef $c \
                  --learning_rate $lr \
                  --train_batch_size $tr_bs \
                  --eval_batch_size $dev_bs

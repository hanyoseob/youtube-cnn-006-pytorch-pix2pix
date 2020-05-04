# !bash ./run_main.sh

#python3 './main.py' \
#        --mode 'train' \
#        --lr 2e-4 \
#        --batch_size 10 \
#        --num_epoch 300 \
#        --ny 256 \
#        --nx 256 \
#        --nch 3 \
#        --nker 64 \
#        --wgt 1e2 \
#        --network 'pix2pix' \
#        --data_dir './../../datasets/facaes' \
#        --ckpt_dir './checkpoint' \
#        --log_dir './log' \
#        --result_dir './result'

python3 '/content/drive/My Drive/YouTube/pytorch-pix2pix/main.py' \
        --mode 'train' \
        --train_continue 'on' \
        --lr 2e-4 \
        --batch_size 10 \
        --num_epoch 300 \
        --ny 256 \
        --nx 256 \
        --nch 3 \
        --nker 64 \
        --wgt 1e2 \
        --opts 'direction' 1 \
        --norm 'bnorm' \
        --network 'pix2pix' \
        --data_dir '/content/drive/My Drive/datasets/facades' \
        --ckpt_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/checkpoint_b2a_bnorm' \
        --log_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/log_b2a_bnorm' \
        --result_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/result_b2a_bnorm'

python3 '/content/drive/My Drive/YouTube/pytorch-pix2pix/main.py' \
        --mode 'train' \
        --train_continue 'on' \
        --lr 2e-4 \
        --batch_size 10 \
        --num_epoch 300 \
        --ny 256 \
        --nx 256 \
        --nch 3 \
        --nker 64 \
        --wgt 1e2 \
        --opts 'direction' 1 \
        --norm 'inorm' \
        --network 'pix2pix' \
        --data_dir '/content/drive/My Drive/datasets/facades' \
        --ckpt_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/checkpoint_b2a_inorm' \
        --log_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/log_b2a_inorm' \
        --result_dir '/content/drive/My Drive/YouTube/pytorch-pix2pix/result_b2a_inorm'
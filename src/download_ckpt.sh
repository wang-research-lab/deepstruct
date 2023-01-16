cd ../ckpt

# download multi-task checkpoints
mkdir MP
cd MP

echo -e 'Downloading 10B multitask checkpoint. This takes a long time, and you can skip it if you only want to have a try on DeepStruct.'
mkdir 10B
cd ./10B
    wget https://huggingface.co/Magolor/deepstruct/resolve/main/hub/MP/10B/mp_rank_00_model_states.pt
cd ..

mkdir 10B_1
cd ./10B_1
    wget https://huggingface.co/Magolor/deepstruct/resolve/main/hub/MP/10B_1/mp_rank_00_model_states.pt
cd ..

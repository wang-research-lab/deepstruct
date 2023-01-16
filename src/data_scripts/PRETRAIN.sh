gdown --fuzzy https://drive.google.com/file/d/1xDb9F8cAEx36gPwxmRmqwzWkezUEPJ-i/view -O ../data/v0.5.4.tar.gz
tar -xzvf ../data/v0.5.4.tar.gz -C ../data
mv ../data/v0.5.4 ../data/cnn_dm_original
rm -f ../data/v0.5.4.tar.gz

wget https://lfs.aminer.cn/misc/cogview/glm-10b-1024.zip
unzip glm-10b-1024.zip -d ../ckpt
mv ../ckpt/glm-10b-1024 ../ckpt/PRETRAIN

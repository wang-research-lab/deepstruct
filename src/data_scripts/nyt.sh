mkdir -p ../data/nyt/
git clone https://github.com/yubowen-ph/JointER.git
rm -r ../data/nyt
mv JointER/dataset/NYT-multi/data ../data/nyt
rm -r JointER
~/.local/bin/gdown --fuzzy https://drive.google.com/file/d/1kguS3pHc7F0NmjJvU-aSmqvKPYAeaAKk/view -O ../data/nyt/schemas.json
cp -rf ../data/nyt/ ../data/nyt_re

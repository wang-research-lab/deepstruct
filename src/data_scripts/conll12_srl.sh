ONTONOTES_PATH=$1
if [ ! -n "$ONTONOTES_PATH" ]; then
        echo -e "\033[31mPlease unzip conll12srl data from https://catalog.ldc.upenn.edu/LDC2013T19 at path ONTONOTES_PATH (absolute path)\033[0m"
        exit 1
fi
echo "ONTONOTES_PATH = ${ONTONOTES_PATH}"

# Prepare conll12 sample ids in each splits
wget -P ../data https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/refs/tags/v12.tar.gz
cd ../data
    tar -xvzf v12.tar.gz
cd ../deepstruct


conda create -n python2 -y python=2.7.18
source activate python2
    # Change format into CoNLL format
    cd ./data_scripts
        bash skeleton2conll.sh -D "${ONTONOTES_PATH}/data/files/data" ../../data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/
    cd ..

    git clone git@github.com:luheng/deep_srl.git
    cd ./deep_srl/
    bash ./scripts/make_conll2012_data.sh ../../data/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/
source activate base
cd ../
cp -rf ./deep_srl/data/srl ../data/
rm -rf ../data/conll12_srl
mv ../data/srl ../data/conll12_srl
rm -rf ./deep_srl
mv ../data/conll12_srl/conll2012.devel.txt ../data/conll12_srl/conll2012.dev.txt

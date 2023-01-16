PTB_PATH=$1
echo "check_certificate = off" >> ~/.wgetrc
if [ ! -n "$PTB_PATH" ]; then
        echo -e "\033[31mPlease untar penn treebank iii data from https://catalog.ldc.upenn.edu/LDC99T42 to path PTB_PATH (absolute path)\033[0m"
        exit 1
fi
git clone https://github.com/luheng/deep_srl.git
cd ./deep_srl/

    # prepare srlconll-1.1.tgz
    SRLPATH="./data/srl"
    if [ ! -d $SRLPATH ]; then
      mkdir -p $SRLPATH
    fi

    # Get srl-conll package.
    wget -O "${SRLPATH}/srlconll-1.1.tgz" --no-check-certificate http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz
    tar xf "${SRLPATH}/srlconll-1.1.tgz" -C "${SRLPATH}"
    rm "${SRLPATH}/srlconll-1.1.tgz"
    sudo apt-get install tcsh

    sudo chmod +x ./scripts/fetch_and_make_conll05_data.sh
    bash ./scripts/fetch_and_make_conll05_data.sh $PTB_PATH
cd ../
cp -rf ./deep_srl/data/srl ../data/
mv ../data/srl ../data/conll05_srl
cp -rf ../data/conll05_srl ../data/conll05_srl_wsj
cp -rf ../data/conll05_srl ../data/conll05_srl_brown
rm -rf ./deep_srl

cp ../data/conll05_srl/conll05.devel.txt ../data/conll05_srl/conll05.dev.txt
cp ../data/conll05_srl_wsj/conll05.test.wsj.txt ../data/conll05_srl_wsj/conll05.test.txt
cp ../data/conll05_srl_brown/conll05.test.brown.txt ../data/conll05_srl_brown/conll05.test.txt

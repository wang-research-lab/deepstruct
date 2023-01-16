mkdir -p ../data/ace2005_joint_er/
ACE05_PATH=$1
if [ ! -n "$ACE05_PATH" ]; then
        echo -e "\033[31mPlease untar ace2005 data from https://catalog.ldc.upenn.edu/LDC2006T06 at path ACE05_PATH (absolute path)\033[0m"
        exit 1
fi
echo "ACE05_PATH = ${ACE05_PATH}"
ACE05_PATH=$1
git clone https://github.com/dwadden/dygiepp.git
cd ./dygiepp
    conda create --name ace-jer-preprocess python=3.7 -y
    source activate ace-jer-preprocess
    pip install -r requirements.txt
    conda develop .
    sudo apt-get update
    sudo apt-get install zsh -y
    sudo apt-get install openjdk-8-jdk -y
    # Please prepare ace2005 data from https://catalog.ldc.upenn.edu/LDC2006T06 at path ACE05_PATH
    bash scripts/data/ace05/get_corenlp.sh
    bash scripts/data/get_ace05.sh ${ACE05_PATH}
    cp -rf ./data/ace05/processed-data/json/* ../../data/ace2005_joint_er/
cd ../
rm -rf dygiepp
cp -r ../data/ace2005_joint_er ../data/ace2005_joint_er_re
source deactivate

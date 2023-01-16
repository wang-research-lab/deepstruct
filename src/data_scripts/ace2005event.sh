mkdir -p ../data/ace2005event_trigger/
mkdir -p ../data/ace2005event_argument/
ACE05_PATH=$1
if [ ! -n "$ACE05_PATH" ]; then
        echo -e "\033[31mPlease untar ace2005 data from https://catalog.ldc.upenn.edu/LDC2006T06 at path ACE05_PATH (absolute path)\033[0m"
        exit 1
fi
echo "ACE05_PATH = ${ACE05_PATH}"

git clone https://github.com/dwadden/dygiepp.git
cd ./dygiepp
    conda create --name ace-event-preprocess -y python=3.7
    source activate ace-event-preprocess
    python -m pip install -r requirements.txt
    python -m pip install -r scripts/data/ace-event/requirements.txt
    python -m spacy download en_core_web_sm
    # Please prepare ace2005 data from https://catalog.ldc.upenn.edu/LDC2006T06 at path ACE05_PATH
    bash ./scripts/data/ace-event/collect_ace_event.sh ${ACE05_PATH}
    python ./scripts/data/ace-event/parse_ace_event.py default-settings
    mkdir -p data/ace-event/collated-data/default-settings/json
    python -m scripts.data.shared.collate \
        data/ace-event/processed-data/default-settings/json \
        data/ace-event/collated-data/default-settings/json \
        --file_extension json
    cp -rf ./data/ace-event/processed-data/default-settings/json/* ../../data/ace2005event_trigger/
    cp -rf ./data/ace-event/processed-data/default-settings/json/* ../../data/ace2005event_argument/
cd ../
rm -rf dygiepp
source deactivate
python data_scripts/jsonl2json.py -i ../data/ace2005event_trigger/train.json -o ../data/ace2005event_trigger/ace2005event_train.json
python data_scripts/jsonl2json.py -i ../data/ace2005event_trigger/dev.json -o ../data/ace2005event_trigger/ace2005event_dev.json
python data_scripts/jsonl2json.py -i ../data/ace2005event_trigger/test.json -o ../data/ace2005event_trigger/ace2005event_test.json
python data_scripts/jsonl2json.py -i ../data/ace2005event_argument/train.json -o ../data/ace2005event_argument/ace2005event_train.json
python data_scripts/jsonl2json.py -i ../data/ace2005event_argument/dev.json -o ../data/ace2005event_argument/ace2005event_dev.json
python data_scripts/jsonl2json.py -i ../data/ace2005event_argument/test.json -o ../data/ace2005event_argument/ace2005event_test.json


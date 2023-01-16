git clone https://github.com/jasonwu0731/trade-dst.git
cp ./data_scripts/multi_woz_create_data.py ./trade-dst/
cd ./trade-dst/
    python multi_woz_create_data.py
cd ../
cp ./trade-dst/utils/fix_label.py ./dataset_processing/preprocess_multiwoz/
cd ./dataset_processing/preprocess_multiwoz/
    python prepare_multi_woz.py --data-dir ../../trade-dst/data
cd ../../
cp -rf ./trade-dst/data/splits ../data/
mv ../data/splits ../data/multi_woz
rm -rf ./trade-dst

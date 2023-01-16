mkdir -p ../data/conll04/
wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/datasets/conll04/ -P ../data/conll04/
cp -rf ../data/conll04/ ../data/conll04_re
mv ../data/conll04_re/conll04_train.json ../data/conll04_re/conll04_re_train.json
mv ../data/conll04_re/conll04_dev.json ../data/conll04_re/conll04_re_dev.json
mv ../data/conll04_re/conll04_test.json ../data/conll04_re/conll04_re_test.json
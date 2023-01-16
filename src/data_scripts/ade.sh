mkdir -p ../data/ade/
wget -r -nH --cut-dirs=100 --reject "index.html*" --no-parent http://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ -P ../data/ade/
cp -rf ../data/ade/ ../data/ade_re
mv ../data/ade_re/ade_full.json ../data/ade_re/ade_re_full.json
mv ../data/ade_re/ade_split_0_train.json ../data/ade_re/ade_re_split_0_train.json
mv ../data/ade_re/ade_split_0_test.json ../data/ade_re/ade_re_split_0_test.json
mv ../data/ade_re/ade_split_1_train.json ../data/ade_re/ade_re_split_1_train.json
mv ../data/ade_re/ade_split_1_test.json ../data/ade_re/ade_re_split_1_test.json
mv ../data/ade_re/ade_split_2_train.json ../data/ade_re/ade_re_split_2_train.json
mv ../data/ade_re/ade_split_2_test.json ../data/ade_re/ade_re_split_2_test.json
mv ../data/ade_re/ade_split_3_train.json ../data/ade_re/ade_re_split_3_train.json
mv ../data/ade_re/ade_split_3_test.json ../data/ade_re/ade_re_split_3_test.json
mv ../data/ade_re/ade_split_4_train.json ../data/ade_re/ade_re_split_4_train.json
mv ../data/ade_re/ade_split_4_test.json ../data/ade_re/ade_re_split_4_test.json
mv ../data/ade_re/ade_split_5_train.json ../data/ade_re/ade_re_split_5_train.json
mv ../data/ade_re/ade_split_5_test.json ../data/ade_re/ade_re_split_5_test.json
mv ../data/ade_re/ade_split_6_train.json ../data/ade_re/ade_re_split_6_train.json
mv ../data/ade_re/ade_split_6_test.json ../data/ade_re/ade_re_split_6_test.json
mv ../data/ade_re/ade_split_7_train.json ../data/ade_re/ade_re_split_7_train.json
mv ../data/ade_re/ade_split_7_test.json ../data/ade_re/ade_re_split_7_test.json
mv ../data/ade_re/ade_split_8_train.json ../data/ade_re/ade_re_split_8_train.json
mv ../data/ade_re/ade_split_8_test.json ../data/ade_re/ade_re_split_8_test.json
mv ../data/ade_re/ade_split_9_train.json ../data/ade_re/ade_re_split_9_train.json
mv ../data/ade_re/ade_split_9_test.json ../data/ade_re/ade_re_split_9_test.json
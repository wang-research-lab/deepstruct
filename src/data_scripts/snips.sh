mkdir -p ../data/snips/
mkdir -p ../data/snips/test/
mkdir -p ../data/snips/train/
mkdir -p ../data/snips/dev/
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/test/label -P ../data/snips/test
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/test/seq.in -P ../data/snips/test
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/test/seq.out -P ../data/snips/test
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/train/label -P ../data/snips/train
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/train/seq.in -P ../data/snips/train
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/train/seq.out -P ../data/snips/train
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/valid/label -P ../data/snips/dev
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/valid/seq.in -P ../data/snips/dev
wget https://raw.githubusercontent.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT/master/data/snips/valid/seq.out -P ../data/snips/dev

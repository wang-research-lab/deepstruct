rm -r ../data/atis
git clone https://github.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT.git
mv joint-intent-classification-and-slot-filling-based-on-BERT/data/atis ../data/atis
mv ../data/atis/valid ../data/atis/dev
rm -r joint-intent-classification-and-slot-filling-based-on-BERT

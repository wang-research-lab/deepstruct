bash ./data_scripts/ace2005event.sh $1
bash ./tasks/mt/ace2005_trigger.sh
cp ../data/ace2005event_trigger/test.target ../data/ace2005event_trigger/test.jsonl.refs
bash ./tasks/mt/ace2005_argument.sh

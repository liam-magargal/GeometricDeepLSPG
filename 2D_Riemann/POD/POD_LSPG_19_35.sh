for i in $(seq 1 10); do 
python FV_RP_LSPG_19_35.py $i;
python RP_POD_comparison_19_35.py $i;
done;
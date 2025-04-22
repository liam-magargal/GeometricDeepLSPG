for i in $(seq 1 10); do 
python FV_RP_LSPG_13_65.py $i;
python RP_POD_comparison_13_65.py $i;
done;
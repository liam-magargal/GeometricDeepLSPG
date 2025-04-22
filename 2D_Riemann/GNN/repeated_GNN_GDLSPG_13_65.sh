for i in $(seq 1 10); do for j in $(seq 1 5); do
python repeated_FV_RP_GNN_GDLSPG_13_65.py $i $j;
python repeated_RP_GNN_comparison_13_65.py $i $j;
done; done;
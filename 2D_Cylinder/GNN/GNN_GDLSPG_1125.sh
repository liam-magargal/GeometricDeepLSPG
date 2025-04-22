for i in $(seq 1 10); do 
python FV_Cyl_GNN_GDLSPG_1125.py $i;
python Cyl_GNN_comparison_1125.py $i;
done; 

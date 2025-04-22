for i in $(seq 1 10); do 
python FV_Burgers_GNN_GD_LSPG_430_021.py $i;
python Burgers_comparison_430_021.py $i;
done; 

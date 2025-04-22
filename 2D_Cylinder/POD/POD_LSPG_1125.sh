for i in $(seq 1 10); do 
python FV_Cyl_LSPG_1125.py $i;
python Cyl_POD_comparison_1125.py $i;
done; 

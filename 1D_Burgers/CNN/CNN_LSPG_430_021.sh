for i in $(seq 1 10); do 
python FV_Burgers_CNN_dLSPG_430_021.py $i;
python Burgers_comparison_430_021.py $i;
done;

for i in $(seq 1 10); do 
python FV_Burgers_CNN_dLSPG_515_0285.py $i;
python Burgers_comparison_515_0285.py $i;
done;

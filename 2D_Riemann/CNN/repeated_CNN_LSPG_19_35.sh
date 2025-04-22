for i in $(seq 1 10); do for j in $(seq 1 5); do
python repeated_FV_RP_CNN_LSPG_19_35.py $i $j;
python repeated_RP_CNN_comparison_19_35.py $i $j;
done; done

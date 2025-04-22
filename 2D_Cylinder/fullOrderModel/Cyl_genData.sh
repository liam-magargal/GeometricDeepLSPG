for i in $(seq 1000 50 1250); do
python3 FV_RP_explicit_unstructured_RHLL_supersonicCylinders_Param.py $i;
done;
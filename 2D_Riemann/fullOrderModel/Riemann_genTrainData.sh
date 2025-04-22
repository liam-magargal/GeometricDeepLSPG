for i in $(seq 0 4); do for j in $(seq 0 4); do 
let u_1=2*i+12;
let v_1=j+3;
python3 2D_Riemann_FOM.py $u_1 $v_1;
done; done;
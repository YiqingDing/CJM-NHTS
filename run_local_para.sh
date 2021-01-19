# Run the main_para file
export PMIX_MCA_gds=hash
mpirun -np 4 python3 raw_dist_para.py
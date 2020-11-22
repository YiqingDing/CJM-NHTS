# Run the main_para file
export PMIX_MCA_gds=hash
mpirun -np 4 python3 main_para.py
//C
gcc -lm -fopenmp seq.c -o seq && ./seq 1024
gcc -lm -fopenmp seq_kahan.c -o seqk && ./seqk 1024

//OpenACC
nvc -acc -gpu=cc70 -O3 acc_col.c -o sac && ./sac 1048576 1024
nvc -acc -gpu=cc70 -O3 acc_col_kahan.c -o sack && ./sack 1048576 1024
nvc -acc -gpu=cc70 -O3 acc_row.c -o sar && ./sar 1024 1024
nvc -acc -gpu=cc70 -O3 acc_row_kahan.c -o sark && ./sark 1024 1024

//CUDA


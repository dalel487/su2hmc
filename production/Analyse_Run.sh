mpirun -n 16 -gtool "advixe-cl -collect survey -mrte-mode=managed -project-dir /home/postgrad/dalel487/intel/advixe/projects/Fortran_Producers --search-dir sym:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir sym:rp=/opt/intel --search-dir bin:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir bin:rp=/opt/intel --search-dir src:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir src:rp=/opt/intel:0-15" /home/postgrad/dalel487/Code/su2hybridcode/production/su2hmc.exe
mpirun -n 16 -gtool "advixe-cl -collect tripcounts -flop -stacks -enable-cache-simulation -project-dir /home/postgrad/dalel487/intel/advixe/projects/Fortran_Producers --search-dir sym:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir sym:rp=/opt/intel --search-dir bin:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir bin:rp=/opt/intel --search-dir src:rp=/home/postgrad/dalel487/Code/su2hybridcode/production --search-dir src:rp=/opt/intel:0-15" /home/postgrad/dalel487/Code/su2hybridcode/production/su2hmc.exe

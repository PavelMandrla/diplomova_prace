set terminal pdfcairo enhanced color font "sans,6" fontscale 1.0 linewidth 1 rounded dashlength 1 background "white" #size 10cm,6cm
set encoding utf8

set datafile separator ","
set key autotitle columnhead
set datafile missing NaN

set xrange [0:7]

file = "results_new.csv"
col_num = 3

set output "err_by_stride_new.pdf"
plot file using 1:(stringcolumn(2) eq "1" ? column(col_num):NaN) title "stride 1"  with lines ls 1 lc rgb "blue",\
       "" using 1:(stringcolumn(2) eq "3" ? column(col_num):NaN) title "stride 3"  with lines ls 1 lc rgb "red",\
       "" using 1:(stringcolumn(2) eq "5" ? column(col_num):NaN) title "stride 5"  with lines ls 1 lc rgb "green"

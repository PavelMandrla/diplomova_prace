set terminal pdfcairo enhanced color font "sans,6" fontscale 1.0 linewidth 1 rounded dashlength 1 background "white" #size 10cm,6cm
set encoding utf8

set datafile separator ","
set key autotitle columnhead
set datafile missing NaN

set xrange [0:7]

file = "results_old_pets.csv"
col_num = 5

set ylabel "Chyba MAE"

set xlabel "hodnota parametru stride"
set output "err_by_len.pdf"
plot file using 2:((stringcolumn(1) eq "1") ? column(col_num):NaN) title "len 1"     w lp ls 1 lc rgb "purple",\
       "" using 2:((stringcolumn(1) eq "2") ? column(col_num):NaN) title "len 2"     w lp ls 1 lc rgb "blue",\
       "" using 2:((stringcolumn(1) eq "3") ? column(col_num):NaN) title "len 3"     w lp ls 1 lc rgb "red",\
       "" using 2:((stringcolumn(1) eq "5") ? column(col_num):NaN) title "len 5"     w lp ls 1 lc rgb "green"

set xlabel "délka vstupní sekvence"
set output "err_by_stride.pdf"
plot file using 1:((stringcolumn(2) eq "1") || (stringcolumn(1) eq "1") ? column(col_num):NaN) title "stride 1"  with lp ls 1 lc rgb "blue",\
       "" using 1:((stringcolumn(2) eq "3") || (stringcolumn(1) eq "1") ? column(col_num):NaN) title "stride 3"  with lp ls 1 lc rgb "red",\
       "" using 1:((stringcolumn(2) eq "5") || (stringcolumn(1) eq "1") ? column(col_num):NaN) title "stride 5"  with lp ls 1 lc rgb "green"

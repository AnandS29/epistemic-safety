for i in 24 ... 25
do
    python3 run_example.py -e $i > log_$i.txt 2>&1 &
done
for i in {24..25}
do
    echo "Running example $i"
    python3 run_example.py -e $i > log_$i.txt 2>&1 &
done
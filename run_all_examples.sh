for (( i=0; i<25; i++ ))
do
    echo "Running example $i"
    python3 run_example.py -e $i > logs/log_$i.txt 2>&1 &
done
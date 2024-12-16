gpus=(0 1 2 3 4 5 6 7)
batch=8 # num gpus
num=50000 # samples

len=$((num / batch + 1))
echo $len

l=0
r=$len
b=()
e=()
for i in `seq 1 $batch`
do
    b+=($l)
    e+=($r)
    l=$((l+len))
    r=$((r+len))
done
echo ${b[@]}
echo ${e[@]}

for i in `seq 0 $((batch-1))`
do
    (
        python judge.py --begin ${b[$i]} \
        --end ${e[$i]} \
        --gpu ${gpus[$i]} \
        --output_path <your-save-path>/vllm_output_$i.json
        echo $i
    )&
done
wait
echo "all weakup"

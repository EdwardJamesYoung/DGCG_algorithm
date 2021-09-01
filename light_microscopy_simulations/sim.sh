# Define array
#$ -t 1-24
# Define working directory
#$ -cwd
# Input and output are the same
#$ -o outputlog
#$ -j y
# Request some node
#$ -l dedicated=24

echo "Task id is $SGE_TASK_ID"

conda activate dgcg
echo "dgcg environment activated"

python simulations.py $SGE_TASK_ID > output.$SGE_TASK_ID
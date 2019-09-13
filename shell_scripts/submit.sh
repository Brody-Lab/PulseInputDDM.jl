#SBATCH -J 'FOF_4_by_cell'
#SBATCH -o ../logs/FOF_4_by_cell.out

region="FOF"
rat_num=4
mode="by_cell"
sbatch --job-name=$region_${rat_num}_$mode.$b.run --output=../logs/FOF_4_by_cell.out sbatch fit_model.sh $1 $2 $3
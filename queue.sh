config=([0]="multilingual-bert-base-russian-100" [1]="multilingual-bert-base-russian-50" [2]="multilingual-bert-base-russian-10")
path=([0]="multilingual-bert-base-fp32/baseline/russian/lr/100" [1]="multilingual-bert-base-fp32/baseline/russian/lr/50" [2]="multilingual-bert-base-fp32/baseline/russian/lr/10")

# for i in {0..2}
# do 
#     sbatch train.sh ${config[i]} ${path[i]}
# done

 sbatch train.sh ${config[0]} ${path[0]}
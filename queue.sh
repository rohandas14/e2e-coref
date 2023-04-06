lang=polish
exp=baseline
config=([0]="multilingual-bert-base-$lang-100" [1]="multilingual-bert-base-$lang-50" [2]="multilingual-bert-base-$lang-10")
path=([0]="multilingual-bert-base-fp32/$exp/$lang/lr/100" [1]="multilingual-bert-base-fp32/$exp/$lang/lr/50" [2]="multilingual-bert-base-fp32/$exp/$lang/lr/10")
branch="bert-coref-baseline"

for i in {0..2}
do 
    sbatch train-alpine.sh ${config[i]} ${path[i]} ${branch}
done

# sbatch train-alpine.sh ${config[0]} ${path[0]} ${branch}

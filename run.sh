models=("convex" "noconvex" "dc")
datasets=("Banknote Authentication" "Ionosphere" "Breast Cancer Wisconsin (Diagnostic)" "Blood Transfusion Service Center" "Tic-Tac-Toe Endgame" "Adult" "Bank Marketing" "MAGIC Gamma Telescope" "Mushroom" "Musk (Version 2)")
lambda0s=(0.0001 0.001 0.002 0.004 0.008 0.016)
lambda1s=(0.0001 0.001 0.002 0.004 0.008 0.016)
pybin="python"

runInstance() {
    model=$1
    results_dir="results"
    dataset=$2
    verbose=0
    seed=10
    lambda0=$3
    lambda1=$4
    
    echo $model $results_dir $dataset $verbose $seed $lambda0 $lambda1
    #"$pybin" -u main.py --model "$model" --results_dir "$results_dir" --dataset "$dataset" --verbose "$verbose" --seed "$seed" --lambda0 "$lambda0"  --lambda0 "$lambda1" 
	        
}
export -f runInstance


for dataset in "${datasets[@]}"
do
	for model in  ${models[@]}
	do
		for lambda0 in ${lambda0s[@]}
		do
			for lambda1 in ${lambda1s[@]}
			do
		    		runInstance "$model" "$dataset" "$lambda0" "$lambda1"
		    	done
	    	done
	done
done


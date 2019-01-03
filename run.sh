function createScript()
{
  cat template.sh | sed -e 's/ARG/'$1'/g'  > script.sh
}

for hidden in 50 75 100 125 150
do
  createScript $hidden
  sbatch script.sh
done

rm script.sh

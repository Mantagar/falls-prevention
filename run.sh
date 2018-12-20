function createScript()
{
  cat template.sh | sed -e 's/STACKS/'$1'/g'  > script.sh
}

for stacks in 2 3 4 5
do
  createScript $stacks
  sbatch script.sh
done

rm script.sh

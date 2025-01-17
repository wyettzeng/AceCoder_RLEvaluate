
mkdir inferenced_output
# creating symbolic links
ln -s /data/code_llm/evalplus/inferenced_output/rl_results/ inferenced_output/evalplus
ln -s /data/code_llm/bigcodebench/bcb_results/ inferenced_output/bcb_results

# Create conda environment
conda create -n codeRLEval python=3.11
conda init
conda activate codeRLEval

pip install -e .
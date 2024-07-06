include .envs/.conda
export

create_conda_env:
	@conda create --name ${CONDA_ENV_NAME} python="3.10"
	@conda activate ${CONDA_ENV_NAME} 
	@conda install pip
	@pip install -r requirements.txt
	@poetry install
	@poetry update

git_push_all:
	@git add .
	@git commit -m "${MESSAGE}"
	@git push

download_data:
	@echo "Not yet"
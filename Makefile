git_push_all:
	@git add .
	@git commit -m "${MESSAGE}"
	@git push

download_data:
	@echo "Not yet"

train_tf:
	@python train.py model_type='tf'

train_torch:
	@python train.py model_type='torch' model='torch/resnet50.yaml' training='torch/trainer.yaml'

inference:
	@streamlit run inference.py
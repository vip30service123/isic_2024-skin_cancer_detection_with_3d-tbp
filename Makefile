git_push_all:
	@git add .
	@git commit -m "${MESSAGE}"
	@git push

download_data:
	@echo "Not yet"

train_tf:
	@python train.py model_type='tf'

train_torch_resnet50:
	@python train.py model_type='torch' model='torch/resnet50.yaml' training='torch/trainer.yaml'

train_torch_efficientnet_b0:
	@python train.py model_type='torch' model='torch/efficientnet_b0.yaml' training='torch/trainer.yaml'

inference:
	@streamlit run inference.py
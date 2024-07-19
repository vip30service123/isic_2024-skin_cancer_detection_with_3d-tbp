# isic_2024-skin_cancer_detection_with_3d-tbp
ISIC 2024 - Skin Cancer Detection with 3D-TBP - Kaggle competition

# Problems
- transform can only be used before batching, model can only use batched data </br>
- One solution: transform is not bound with model creation, it is bound with model config. So more config incomming </br>


# Tasks
- Use mlflow </br> 
- Fix pytorch trainer (it's urgly) </br>
- Check all configs before running the programm </br>
- Modify configs </br>
- Clean code </br>
- Change print into log </br>
- Deploy on GCP </br>
- Inference using streamlit </br>
- Change conda installation </br>
- Fix Docker </br>
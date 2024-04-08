# image-processing-project
Final project for the course Intro to Image Processing (2024/1).

In this work, we aim to investigate the effects of several image processing techniques on the performance of the ICA model for rPPG signal extraction. By applying several image enhancement algorithms on each of the video frames with different parameters we attempt to improve the resulting PPG signal. We use Bayesian optimization techniques to find the parameters that yield the best results.
Our research question is: can image enhancement techniques improve the performance of an rPPG signal extraction model?


# Instructions
1. Navigate to the project folder and install the dependencies using `pip install -r requirements.txt`.
2. In `setup.py` replace the `RAW_DATA_DIR` to the directory containing the videos from  the UBF-rPPG dataset, the `GT_DATA_DIR` to the directory containing the ground truth signals, and the `PREPROCESSED_DATA_DIR` to the desired directory to save the processed data.
3. Run `main.py`.

# Licenses
- UBFC-rPPG Dataset is used for research purposes with permission from the authors. See https://sites.google.com/view/ybenezeth/ubfcrppg.
- Unsupervised rPPG models and post-processing methods are taken from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox/) and are subjected to the [Responsible Artificial Intelligence Source Code License](LICENSE).
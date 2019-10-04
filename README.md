# Semantic-Segmentation-for-RGB-Aerial-Images in collaboration with the Autonomous Systems Lab at ETH Zurich

Semantic Segmentation is crucial for UAVs for finding suitable landing spot at a random outdoor environment, especially for an emergency landing situation. In this work in collaboration with ASL at ETH Zurich, i have tried to address this issue using semantic segmentation for aerial images, taken from a real drone. we use the state-of-art semantic segmentation network Full Resolution Residual Network (FRRN) for semantic segmentation of these images, which could be used to develop an algorithm
to guide the UAV towards suitable landing spot. The proposed framework has been successfully tested on real UAV datasets and in challenging real-world environments.The trained model was also tested on real camera RGB-images of ASL-ETH Zurich drone dataset, which gave very favorable results.

Please do also find the poster (Final_poster.pdf) for this project which gives more detailed answeres for the results and accuracy.

The acceptable dataset format is the similar to Semantic Drone Dataset bu TU Graz. So you can also use that directly.

For running the code please run "train.py". You can tune the parameters and hyperparameters of the networks by editing the values in the training script itself.

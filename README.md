# DimensionalityEstimation
Devoted to experiments on using the Kmeans++ to estimate dimension

## src: Source Directory

* **get_read_videos**	Code to download videos from a youtube list and to then load each frame into a numpy array using open-cv
* **FaceDetection**	Code to transform a video of an announcer (here colbert) into a sequence of `np.array`s containing 300x300 BW/windows
  * detect face locations in using boosted haar (standard in open-cv)
  * Form tracks here the face is moving smoothly from frame to frame
  * estimate the skin color and trim the non-skin parts, translate the image into BW
* **kmeans++**
* **Spark-KMeans++**	

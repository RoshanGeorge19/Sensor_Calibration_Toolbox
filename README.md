# Sensor Calibration Toolbox
A tool to find the relative pose and project the LiDAR point cloud from an ego-vehicle onto a camera embedded in the infrastructure.

By using the shared correspondences in the point cloud and the image, the relative pose between the two sensors is found. The point cloud is then projected onto the image using the camera intrinsics. Depending on your setup, the point cloud can either be an instantaneous point cloud or an accumulated point cloud (generated from the vehicle odometry). 

Easy to follow instructions and tool on how to SLAM on a LiDAR point cloud and generate an accumulated point cloud --> https://gitlab.kitware.com/keu-computervision/slam/-/blob/master/paraview_wrapping/Plugin/doc/How_to_SLAM_with_LidarView.md

# Cloud Info
Header header 

# 记录每条线束的起点所对应的有效点编号(在下面的cloud_deskewed容器中)
int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
# 该帧点云扫描起始时刻的 Lidar的姿态(从IMU的姿态转换到Lidar的姿态)
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Preintegration reset ID
int64 imuPreintegrationResetId

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed   去畸变的原始点云
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "featureExtraction.h"
#include "mapOptmization.h"

/// 自定义的点云数据结构
/// xyz，强度，线束号，时间戳
struct PointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;         // 点的相对时间（相对于扫描起始时刻的时间）
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,  
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (uint16_t, ring, ring) (float, time, time)
)

const int queueLength = 500;

/**
 * @brief The ImageProjection class
 * 点云去畸变，IMU时间戳对齐
 */
class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;          //转换到Lidar坐标系的IMU数据

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;
    
    // imu队列，默认长度500
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];          //相对于某帧扫描起始时刻的IMU积分数据, 任意一帧扫描的起始时刻下，imuRotX[0]=0 ...
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;          // 当前imu数据的idx
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;     //当前帧点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    // odom增量（扫描结束时刻 ---> 扫描起始时刻）
    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;             //当前帧点云的扫描起始时间
    double timeScanNext;            //当前帧点云的扫描结束时间
    std_msgs::Header cloudHeader;


    FeatureExtraction extrator_;
    mapOptimization matcher_;

public:
    // 构造函数
    ImageProjection():
    deskewFlag(0)
    {
        // 订阅
        // 原始IMU？
        subImu        = nh.subscribe<sensor_msgs::Imu>("/imu/data", 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
//        // IMU预积分（这耦合关系。。。）
        //subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic, 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
//        // 原始点云
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布
        //
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam_custom/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam_custom/deskew/cloud_info", 10);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        // pcl::PointCloud<PointXYZIRT>::Ptr 初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    // IMU回调
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        // ROS_INFO("[IMU] IMU Msg");
        // 将机体坐标系的IMU数据转换到Lidar坐标系 （加速度、角速度、姿态）  全局坐标系为ENU
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        // 线程锁
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // ROS_INFO("[Lidar] Point CLoud Msg");
        // 点云字段检查
        if (!cachePointCloud(laserCloudMsg))
            return;

//        // 去畸变，设置位姿估计初始值
//        if (!deskewInfo())
//            return;

        // 将点云投影成深度图（矩阵）
        // x轴负方向对应的列号=0 ， 按顺时针增加
        // x轴帧方向对应的列号=Horizon_SCAN/2
        // 行号即为线束编号
        projectPointCloud();

        cloudExtraction();

        publishClouds();

        extrator_.featureExtra(cloudInfo);


        static Eigen::Affine3f pose=Eigen::Affine3f::Identity();
        static Eigen::Affine3f last_pose = Eigen::Affine3f::Identity();
        static Eigen::Affine3f step = Eigen::Affine3f::Identity();
        static bool inited=false;
        if(!inited){
            Eigen::Matrix3f R_init;
            R_init<<0 , 1 , 0 , -1 , 0 , 0 , 0 , 0 ,1;
            pose.rotate(R_init);
            last_pose=pose;
            inited=true;
        }
        pose= pose*step;
        matcher_.registration(extrator_.cloudInfo,pose);
        //step=last_pose.inverse()*pose;
        //last_pose=pose;

//        std::cout<<pose.translation().transpose()<<std::endl;

        // 一帧点云处理完，参数重置
        resetParameters();
    }

    // 储存并检查点云
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        // 储存原始点云
        cloudQueue.push_back(*laserCloudMsg);

        // 如果是头两帧，储存之后直接返回false
        if (cloudQueue.size() <= 2)
            return false;
        else
        {
            // 取队列中的第一个点云数据，作为currentCloudMsg
            currentCloudMsg = cloudQueue.front();
            cloudQueue.pop_front();

            // 取点云时间戳，传给timeScanCur
            cloudHeader = currentCloudMsg.header;
            timeScanCur = cloudHeader.stamp.toSec();
            // 取下一帧点云时间戳，传给timeScanNext
            timeScanNext = cloudQueue.front().header.stamp.toSec();
        }

        // convert cloud
        // 取currentCloudMsg，转成pcl类型的laserCloudIn
        pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        // 检查点云msg是否有"ring"字段
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }   

        // check point time
        // 检查点云msg是否有"time"字段
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "time")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 如果imu队列为空|| imu首个数据时间戳比点云时间戳晚 || IMU队列最后一个数据时间戳比点云时间戳早
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanNext)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        // 遍历imu队列
        while (!imuQueue.empty())
        {
            // 如果imu队列首个数据时间戳 < 点云时间戳-0.01
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();   //舍弃该imu数据
            else
                break;
        }

        // 如果IMU队列中的数据都不符合，直接返回
        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        // 遍历IMU队列
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            // 取一个imu数据
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            // 取该数据时间戳
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // imu数据 比 点云时间戳早 ， 取该imu数据的 roll, pitch, and yaw （实际上是Lidar的姿态）
            // 储存到cloudInfo结构体内，准备发布
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            // imu数据 比 下一帧点云时间戳晚，并且超过0.01秒，直接返回
            if (currentImuTime > timeScanNext + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 取imu数据角速度，准备下一步积分
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            // 积分旋转量
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];                //当前imu数据-上一个imu数据时间戳
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        // 上面的for循环最后一次，imuPointerCur指向了一个比末尾数据还+1的地方，现在重新指回imu末尾数据
        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        // 设置标志位，表明该点云数据对应的imu信息可用
        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        // 获取扫描起始时刻对应的odom
        nav_msgs::Odometry startOdomMsg;

        // 遍历odomQueue队列
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            // 取一个odom数据
            startOdomMsg = odomQueue[i];

            // 如果该数据比当前帧扫描时间戳早
            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;   //直接遍历下一个odom数据
            else
                break;
        }

        // 取odom消息(IMU预积分)的姿态
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 为mapOptimization模块设置初始估计值
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;
        cloudInfo.imuPreintegrationResetId = round(startOdomMsg.pose.covariance[0]);

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        // 获取扫描结束时刻的odom
        odomDeskewFlag = false;

        // 如果odom队列最后一个数据比下一帧扫描时间戳早（当前帧扫描结束时刻）
        if (odomQueue.back().header.stamp.toSec() < timeScanNext)
            return;

        nav_msgs::Odometry endOdomMsg;

        // 遍历odom队列
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanNext)
                continue;
            else
                break;
        }

        // 如果扫描起始和结束时刻的odom协方差取整后不相等，直接return
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 获取扫描起始时刻的变换
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 获取扫描结束时刻的旋转
        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // 获取扫描结束时刻的变换
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 扫描结束时刻 到 扫描起始时刻的变换
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        // 将这个变换分解
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    // 插值获取旋转量(这个激光点所对应的Lidar姿态)
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            // 当imuPointerFront这个idx对应的imu数据时间戳 > 给定某个点的时间戳，则跳出while
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            // 遍历完所有imu数据，没有找到一个imu数据的时间戳是在这个点之后的，  或者
            // imuPointerFront==0的时候，其时间戳就在这个点之后，
            // 直接取 imuPointerFront 这个id的imu数据作为当前这个点所对应的姿态
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 找到某个imu数据的时间戳在这个激光点的时间戳之后
            // 则去这个imu数据的前一个数据，那么这个数据的时间戳必定在激光点时间戳之前
            // 接下来进行插值
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanNext - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    // 将点变换到扫描起始时刻的激光雷达坐标系(去畸变)
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 判断点云数据是否有时间戳（每个点）  以及 对应的imu数据是否可用
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        // 得到这个点的时间
        double pointTime = timeScanCur + relTime;

        // 插值获取旋转量(这个激光点所对应的Lidar姿态增量，相对于扫描起始时刻的姿态)
        //
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // 将点变换到扫描起始时刻的激光雷达坐标系
        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    // 将点云投影成深度图（矩阵）
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        // 遍历当前帧点云
        for (int i = 0; i < cloudSize; ++i)
        {
            // 取一个点
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // 取该点的线束编号
            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;
            // 看看线束编号
            // ROS_INFO("Point at ring: %d ",rowIdn);

            // 计算该点所对应扫描的角度
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // 水平扫描的分辨率： 0.2度
            float ang_res_x = 360.0/float(Horizon_SCAN);
            // x轴方向所在的线束列编号为 Horizon_SCAN/2
            // x轴负方向所在线束列编号为 0
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;
            // 计算点到激光雷达的距离
            float range = pointDistance(thisPoint);
            
            if (range < 1.0)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // for the amsterdam dataset
            // if (range < 6.0 && rowIdn <= 7 && (columnIdn >= 1600 || columnIdn <= 200))
            //     continue;
            // if (thisPoint.z < -2.0)
            //     continue;

            // 构造矩阵，元素的值为点到雷达的距离
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            int index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        // 遍历每条线束
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 记录每条线束的起点所对应的有效点编号， -1+5 是后面计算粗糙度的时候，需要用到前后5点，一共10个点的信息
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            // 遍历水平扫一圈的数据
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // 从深度矩阵获取距离值
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // 为什么坐标系是base_link？
        // 发布去畸变的点云
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, "base_link");
        // 同时发布对应的点云信息
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "feature_matching_node");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}

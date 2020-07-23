#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

/**
 * @brief The ParamServer class
 * 参数服务器 —— 作为基类，有意思
 */
class ParamServer
{
public:

    ros::NodeHandle nh;
    ros::NodeHandle nh_priv;

    std::string robot_id;

    //topic
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Velodyne Sensor Configuration: Velodyne
    int N_SCAN;                 //线束数
    int Horizon_SCAN;           //水平扫一圈的点数  360*5=1800

    // IMU
    float imuAccNoise;
    float imuGyrNoise;
    float imuAccBiasN;
    float imuGyrBiasN;
    float imuGravity;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;     // IMU到Lidar的旋转变换
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance; 
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold; 
    float surroundingkeyframeAddingAngleThreshold; 
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;
    
    // Loop closure
    bool loopClosureEnableFlag;
    int   surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int   historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    /**
     * @brief ParamServer
     * @brief 构造函数
     * @brief 功能：读取配置参数
     */
    ParamServer()
    {
        nh_priv = ros::NodeHandle("~");
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        nh_priv.param<std::string>("pointCloudTopic", pointCloudTopic, "points_raw");
        nh_priv.param<std::string>("imuTopic", imuTopic, "imu_correct");
        nh_priv.param<std::string>("odomTopic", odomTopic, "odometry/imu");
        nh_priv.param<std::string>("gpsTopic", gpsTopic, "odometry/gps");

        nh_priv.param<bool>("useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh_priv.param<bool>("useGpsElevation", useGpsElevation, false);
        nh_priv.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
        nh_priv.param<float>("poseCovThreshold", poseCovThreshold, 25.0);

        nh_priv.param<bool>("savePCD", savePCD, true);
        nh_priv.param<std::string>("savePCDDirectory", savePCDDirectory, "/home/autoware/shared_dir");

        nh_priv.param<int>("N_SCAN", N_SCAN, 16);
        nh_priv.param<int>("Horizon_SCAN", Horizon_SCAN, 1800);

        nh_priv.param<float>("imuAccNoise", imuAccNoise, 0.01);
        nh_priv.param<float>("imuGyrNoise", imuGyrNoise, 0.001);
        nh_priv.param<float>("imuAccBiasN", imuAccBiasN, 0.0002);
        nh_priv.param<float>("imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh_priv.param<float>("imuGravity", imuGravity, 9.80511);
        nh_priv.param<vector<double>>("extrinsicRot", extRotV, vector<double>());
        nh_priv.param<vector<double>>("extrinsicRPY", extRPYV, vector<double>());
        nh_priv.param<vector<double>>("extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        nh_priv.param<float>("edgeThreshold", edgeThreshold, 0.1);
        nh_priv.param<float>("surfThreshold", surfThreshold, 0.1);
        nh_priv.param<int>("edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh_priv.param<int>("surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh_priv.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh_priv.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh_priv.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh_priv.param<float>("z_tollerance", z_tollerance, FLT_MAX);
        nh_priv.param<float>("rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh_priv.param<int>("numberOfCores", numberOfCores, 2);
        nh_priv.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);

        nh_priv.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh_priv.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh_priv.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh_priv.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh_priv.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh_priv.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh_priv.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh_priv.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh_priv.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh_priv.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh_priv.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh_priv.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh_priv.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
    }

    /**
     * @brief imuConverter
     * @param imu_in    输入的IMU数据
     * @return
     */
    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        // IMU坐标系的加速度，转换到Lidar坐标系
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        // 把姿态转换成Lidar的姿态
        // extQRPY与extRot不一致，是因为，惯导输出的姿态是 车头朝北时，yaw=0, 实际上，在ENU坐标系，车头朝北，yaw=90度才对
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final =extQRPY * q_from;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        // 检查，需要9轴
        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }

    sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
    {
        sensor_msgs::PointCloud2 tempCloud;
        pcl::toROSMsg(*thisCloud, tempCloud);
        tempCloud.header.stamp = thisStamp;
        tempCloud.header.frame_id = thisFrame;
        if (thisPub->getNumSubscribers() != 0)
            thisPub->publish(tempCloud);
        return tempCloud;
    }

    // 获取ros msg的时间戳
    template<typename T>
    double ROS_TIME(T msg)
    {
        return msg->header.stamp.toSec();
    }


    template<typename T>
    void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
    {
        *angular_x = thisImuMsg->angular_velocity.x;
        *angular_y = thisImuMsg->angular_velocity.y;
        *angular_z = thisImuMsg->angular_velocity.z;
    }


    template<typename T>
    void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
    {
        *acc_x = thisImuMsg->linear_acceleration.x;
        *acc_y = thisImuMsg->linear_acceleration.y;
        *acc_z = thisImuMsg->linear_acceleration.z;
    }


    template<typename T>
    void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
    {
        double imuRoll, imuPitch, imuYaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

        *rosRoll = imuRoll;
        *rosPitch = imuPitch;
        *rosYaw = imuYaw;
    }

    // 计算点到原点的距离
    float pointDistance(PointType p)
    {
        return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }


    float pointDistance(PointType p1, PointType p2)
    {
        return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
    }

};

#endif

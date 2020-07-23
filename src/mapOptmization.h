#ifndef MAPOPTMIZATION_H
#define MAPOPTMIZATION_H


#include "utility.h"
#include "feature_matching/cloud_info.h"
#include "tic_toc.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMappedROS;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subGPS;

    std::deque<nav_msgs::Odometry> gpsQueue;
    feature_matching::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;        // 关键帧位姿节点?
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization  降采样后的边缘点特征
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;      // 特征点（边缘点+平面点）
    pcl::PointCloud<PointType>::Ptr coeffSel;           // 与laserCloudOri 对应的系数

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;  // 世界坐标系下的 局部边缘线特征地图
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::PointCloud<PointType>::Ptr corner_GlobalMap;
    pcl::PointCloud<PointType>::Ptr surf_GlobalMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;        // [闭环用]最新帧的特征点点云(已经变换到世界坐标系的)
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;   // [闭环用]闭环关键帧附近合并形成的局部地图(已经变换到世界坐标系的)

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

    // 点云回调，取点云时间戳
    ros::Time timeLaserInfoStamp;
    double timeLaserCloudInfoLast;

    float transformTobeMapped[6];       //(roll,pitch,yaw,x,y,z)    相机位姿

    std::mutex mtx;

    double timeLastProcessing = -1;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    int laserCloudCornerFromMapDSNum = 0;
    int laserCloudSurfFromMapDSNum = 0;
    int laserCloudCornerLastDSNum = 0;              //当前帧点云 边缘点特征
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;       //当前帧点云对应的预积分的位姿估计

    // 构造函数
    mapOptimization()
    {
        // ISAM2
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // 发布
        // 关键帧点云
        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/trajectory", 1);
        // 局部点云地图
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/map_global", 1);
        // 匹配？
        pubOdomAftMappedROS = nh.advertise<nav_msgs::Odometry> ("lio_sam_custom/mapping/odometry", 1);
        // path
        pubPath = nh.advertise<nav_msgs::Path>("lio_sam_custom/mapping/path", 1);

        // 订阅
        // 1. 提取特征的点云以及IMU积分估计位姿
        subLaserCloudInfo = nh.subscribe<feature_matching::cloud_info>("lio_sam_custom/feature/cloud_info", 10, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 2. GPS
        //subGPS = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布
        // 历史关键帧
        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/icp_loop_closure_history_cloud", 1);
        // ICP回环点云
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/icp_loop_closure_corrected_cloud", 1);

        // 最近几帧
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/map_local", 1);
        // 当前帧配准完成后的点云
        pubRecentKeyFrame = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/cloud_registered", 1);
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/mapping/cloud_registered_raw", 1);

        // 降采样参数
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());


        corner_GlobalMap.reset(new pcl::PointCloud<PointType>());
        surf_GlobalMap.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();

        {
            // 读取地图文件
            pcl::io::loadPCDFile(std::getenv("HOME")+savePCDDirectory + "cloudCorner.pcd", *corner_GlobalMap);
            pcl::io::loadPCDFile(std::getenv("HOME")+savePCDDirectory + "cloudSurf.pcd", *surf_GlobalMap);
            /// 降采样
            // Downsample the surrounding corner key frames (or map)
            downSizeFilterCorner.setInputCloud(corner_GlobalMap);
            downSizeFilterCorner.filter(*corner_GlobalMap);
            //laserCloudCornerFromMapDS = corner_GlobalMap;
            //laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
            // Downsample the surrounding surf key frames (or map)
            downSizeFilterSurf.setInputCloud(surf_GlobalMap);
            downSizeFilterSurf.filter(*surf_GlobalMap);
            //laserCloudSurfFromMapDS = surf_GlobalMap;
            //laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
        }
    }

    void registration(const feature_matching::cloud_info& cloud_info_, Eigen::Affine3f& pose_guess_){
        // extract time stamp
        // 取时间戳
        timeLaserInfoStamp = cloud_info_.header.stamp;
        timeLaserCloudInfoLast = cloud_info_.header.stamp.toSec();

        // extract info and feature cloud
        // 分别取角点、平面点
        cloudInfo = cloud_info_;
        pcl::fromROSMsg(cloud_info_.cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(cloud_info_.cloud_surface, *laserCloudSurfLast);

        // 线程锁
        std::lock_guard<std::mutex> lock(mtx);

        // 如果当前帧点云时间戳 - 上一次处理点云的时间戳 > 设定间隔
        if (timeLaserCloudInfoLast - timeLastProcessing >= mappingProcessInterval) {

            timeLastProcessing = timeLaserCloudInfoLast;

            std::cout<<"InPut: "<<pose_guess_.translation().transpose()<<std::endl;
            std::vector<float> origin = {pose_guess_.translation().x(), pose_guess_.translation().y(), pose_guess_.translation().z()};
            std::vector<float> edge;
            std::vector<float> size ={-30.0, 30.0, -30.0, 30.0, -10.0, 10.0};
            // 取局部地图
            pcl::CropBox<PointType> pcl_box_filter_;
            for (size_t i = 0; i < origin.size(); ++i) {
                edge.emplace_back(size.at(2 * i) + origin.at(i));
                edge.emplace_back(size.at(2 * i + 1) + origin.at(i));
            }
            // 根据网格划分边缘，从全地图中裁剪，得到局部地图
            laserCloudCornerFromMapDS->clear();
            pcl_box_filter_.setMin(Eigen::Vector4f(edge.at(0), edge.at(2), edge.at(4), 1.0e-6));
            pcl_box_filter_.setMax(Eigen::Vector4f(edge.at(1), edge.at(3), edge.at(5), 1.0e6));
            // 然后再次滤波
            pcl_box_filter_.setInputCloud(corner_GlobalMap);
            pcl_box_filter_.filter(*laserCloudCornerFromMapDS);
            laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
            laserCloudSurfFromMapDS->clear();
            pcl_box_filter_.setInputCloud(surf_GlobalMap);
            pcl_box_filter_.filter(*laserCloudSurfFromMapDS);
            laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

            //更新初始化位姿估计 transformTobeMapped
            //updateInitialGuess();
            // 分解transFinal，储存到transformTobeMapped
            pcl::getTranslationAndEulerAngles(pose_guess_, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            //对当前帧点云的边缘点、平面点集合进行降采样
            downsampleCurrentScan();

            TicToc tic_;
            //根据边缘点特征、平面点特征，进行匹配，然后LM优化 Lidar位姿
            scan2MapOptimization();
            std::cout<<"LM 耗时: "<<tic_.toc()<<std::endl;

            //判断是否关键帧，然后加入因子，因子图优化
            //saveKeyFramesAndFactor();

            //如果有回环，则更新全局位姿
            //correctPoses();

            pose_guess_=pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            std::cout<<"OutPut: "<<pose_guess_.translation().transpose()<<std::endl;

//            // 匹配之后判断是否需要更新局部地图
//            for (int i = 0; i < 3; i++) {
//                if (fabs(cloud_pose(i, 3) - edge.at(2 * i)) > 50.0 &&
//                    fabs(cloud_pose(i, 3) - edge.at(2 * i + 1)) > 50.0)
//                    continue;
//                ResetLocalMap(cloud_pose(0,3), cloud_pose(1,3), cloud_pose(2,3));
//                break;
//            }

            publishOdometry();

            publishFrames();
        }
    }

    // 点云回调
    void laserCloudInfoHandler(const feature_matching::cloud_infoConstPtr& msgIn)
    {
        // extract time stamp
        // 取时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserCloudInfoLast = msgIn->header.stamp.toSec();

        // extract info and feature cloud
        // 分别取角点、平面点
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        // 线程锁
        std::lock_guard<std::mutex> lock(mtx);

        // 如果当前帧点云时间戳 - 上一次处理点云的时间戳 > 设定间隔
        if (timeLaserCloudInfoLast - timeLastProcessing >= mappingProcessInterval) {

            timeLastProcessing = timeLaserCloudInfoLast;

            //更新初始化位姿估计 transformTobeMapped
            updateInitialGuess();

            //临近关键帧点云合并得到局部边缘点、平面点特征地图
            extractSurroundingKeyFrames();

            //对当前帧点云的边缘点、平面点集合进行降采样
            downsampleCurrentScan();

            //根据边缘点特征、平面点特征，进行匹配，然后LM优化 Lidar位姿
            scan2MapOptimization();

            //判断是否关键帧，然后加入因子，因子图优化
            saveKeyFramesAndFactor();

            //如果有回环，则更新全局位姿
            correctPoses();

            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    // 将激光雷达坐标系的点转换到地图坐标系
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        for (int i = 0; i < cloudSize; ++i){

            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        // (x,y,z,roll,pitch,yaw)
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }
















    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, surf_GlobalMap/*globalMapKeyFramesDS*/, timeLaserInfoStamp, "odom");
    }











    // 回环线程
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(0.2);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
        }
    }

    /**
     * @brief detectLoopClosure 获取回环候选关键帧
     *  1. 位姿节点构建成kd-tree
     *  2. 根据最新关键帧的位置，在kd-tree中查找给定范围内的节点，取时间上最老的节点作为闭环候选
     *  3. 将最新关键帧的特征点点云变换到世界坐标系下，储存到latestKeyFrameCloud
     *  4. 跟据闭环候选关键帧，取前后几帧，合并成局部地图
     *  5. 输出两个id
     * @param latestID      [输出]最新关键帧id
     * @param closestID     [输出]闭环候选关键帧id
     * @return
     */
    bool detectLoopClosure(int *latestID, int *closestID)
    {
        int latestFrameIDLoopCloure;
        int closestHistoryFrameID;

        latestKeyFrameCloud->clear();
        nearHistoryKeyFrameCloud->clear();

        std::lock_guard<std::mutex> lock(mtx);

        // find the closest history key frame
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        // 所有关键帧位置——构建KD tree
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
        // 查找符合范围内的所有节点 (要查找的节点中心，半径，输出，输出)
        kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        closestHistoryFrameID = -1;
        // 遍历查找到的节点
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            // 取对应的索引
            int id = pointSearchIndLoop[i];
            // 检查时间戳是否大于阈值
            // 取最老的闭环
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserCloudInfoLast) > historyKeyframeSearchTimeDiff)
            {
                closestHistoryFrameID = id;
                break;
            }
        }

        if (closestHistoryFrameID == -1)
            return false;

        // 如果找到的刚好是本身
        if ((int)cloudKeyPoses3D->size() - 1 == closestHistoryFrameID)
            return false;

        // save latest key frames
        // 储存最新帧的id
        latestFrameIDLoopCloure = cloudKeyPoses3D->size() - 1;
        // 将最新帧的特征点变换到世界坐标系，然后合并到latestKeyFrameCloud
        *latestKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        *latestKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure],   &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        // save history near key frames
        bool nearFrameAvailable = false;
        // 取闭环候选的前后各historyKeyframeSearchNum帧，合并成局部回环地图
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
        {
            // 如果是前后两头的情况，则跳过
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            // 合并形成 闭环候选的局部地图
            *nearHistoryKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[closestHistoryFrameID+j], &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            *nearHistoryKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID+j],   &cloudKeyPoses6D->points[closestHistoryFrameID+j]);
            nearFrameAvailable = true;
        }

        if (nearFrameAvailable == false)
            return false;

        *latestID = latestFrameIDLoopCloure;        //最新关键帧的id
        *closestID = closestHistoryFrameID;         //闭环候选关键帧id

        return true;
    }

    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        int latestFrameIDLoopCloure;
        int closestHistoryFrameID;
        // kt-tree找临近位姿节点
        if (detectLoopClosure(&latestFrameIDLoopCloure, &closestHistoryFrameID) == false)
            return;


        // ICP Settings
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Downsample map cloud
        // 局部地图将采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearHistoryKeyFrameCloud);
        downSizeFilterICP.filter(*cloud_temp);
        *nearHistoryKeyFrameCloud = *cloud_temp;
        // publish history near key frames
        // 发布闭环候选局部地图
        publishCloud(&pubHistoryKeyFrames, nearHistoryKeyFrameCloud, timeLaserInfoStamp, "odom");

        // Align clouds
        // icp匹配
        icp.setInputSource(latestKeyFrameCloud);
        icp.setInputTarget(nearHistoryKeyFrameCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 如果icp不收敛，或者匹配(距离)>阈值，匹配失败
        // std::cout << "ICP converg flag:" << icp.hasConverged() << ". Fitness score: " << icp.getFitnessScore() << std::endl;
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        // 否则，回环成功，发布 ： 将最新关键帧的点云变换到世界地图上
        if (pubIcpKeyFrames.getNumSubscribers() != 0){
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*latestKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
        }

        // Get pose transformation
        float x, y, z, roll, pitch, yaw;
        // 取回环得到：    [wrong世界坐标系 到 闭环局部地图(true世界坐标系)的变换] (注意:不是得到Lidar在true世界坐标系的位姿)
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        // 取旧的最新关键帧位姿（这里描述为wrong的位姿）
        Eigen::Affine3f tWrong = pclPointToAffine3f(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        // transform from world origin to corrected pose
        // 计算 旧的关键帧Lidar坐标系到 回环得到的Lidar坐标系变换
        // tWrong : 最新关键帧Lidar坐标系 到 wrong世界坐标系的位姿
        // correctionLidarFrame : wrong世界坐标系 到 闭环局部地图(true世界坐标系)的变换
        // tCorrect： 最新关键帧Lidar坐标系在世界坐标系的位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        // 分解
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        // 构造回环因子
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
        // 噪声模型 （取icp匹配分数）
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        //
        std::lock_guard<std::mutex> lock(mtx);
        // 添加到因子图()
        // poseFrom.inv(): 回环得到的Lidar坐标系到旧的关键帧Lidar坐标系的变换
        // poseTo: 闭环关键帧对应的Lidar位姿
        // poseFrom.between(poseTo)： 闭环关键帧到最新关键帧的变换
        gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        gtSAMgraph.resize(0);

        // 取优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        Pose3 latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // 更新当前帧位姿
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // 清除path，需要更新全局位姿
        correctPoses();

        aLoopIsClosed = true;
    }









    /**
     * @brief updateInitialGuess
     * 更新初始化位姿估计
     *  1. 初始化-第一帧：使用9轴IMU数据
     *  2. 预积分数据有效，使用IMU预积分数据
     *  3. 该帧点云对应的IMU数据有效，使用IMU数据
     */
    void updateInitialGuess()
    {
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 如果是初始化，第一帧，则用9轴IMU
        if (cloudKeyPoses3D->points.empty())
        {
            // 记录该帧点云对应的Lidar姿态（在imageProjection模块中根据9轴IMU的数据得到的）
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit/*-M_PI/2*/;

            // 如果不使用IMU的朝向初始化
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // 记录姿态
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // 如果不是初始化，那么就用IMU预积分的结果作为 初始位姿估计
        // use imu pre-integration estimation for pose guess
        // （发生回环校正后,imuPreintegrationResetId会改变）
        if (cloudInfo.odomAvailable == true && cloudInfo.imuPreintegrationResetId == imuPreintegrationResetId)
        {
            transformTobeMapped[0] = cloudInfo.initialGuessRoll;
            transformTobeMapped[1] = cloudInfo.initialGuessPitch;
            transformTobeMapped[2] = cloudInfo.initialGuessYaw;

            transformTobeMapped[3] = cloudInfo.initialGuessX;
            transformTobeMapped[4] = cloudInfo.initialGuessY;
            transformTobeMapped[5] = cloudInfo.initialGuessZ;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }

        // 如果点云msg标记了对应的IMU数据可用，那么使用IMU的数据增量作为初始位姿估计
        // use imu incremental estimation for pose guess (only rotation)
        if (cloudInfo.imuAvailable == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            // 位姿估计增量（当前帧到上一帧的变换）
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            // 根据增量，以及上一次的位姿估计transformTobeMapped，计算当前帧位姿估计transFinal
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            // 分解transFinal，储存到transformTobeMapped
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 将所有关键帧的位置（x，y，z）构建KD-Tree
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        // 查找空间距离最新关键帧最近的节点，把节点id和距离分别储存到pointSearchInd，pointSearchSqDis
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // 最近点队列储存到 surroundingKeyPoses ， 准备进行滤波处理
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        // 降采样
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // also extract some latest key frames in case the robot rotates in one position
        // 同时，提取时间上最近的关键帧，以防只做旋转运动
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            // 把当前帧点云时间戳前10秒内的节点，加入到 surroundingKeyPosesDS
            if (timeLaserCloudInfoLast - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // vector: cloudToExtract有多少个位姿节点，就有多少帧点云数据
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历提取到的位姿节点
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 再次检查两个位姿节点的距离
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;
            // 取id，用来取姿态
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 将提取的边沿点特征集合，变换到世界坐标系W
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            // 将提取的平面点特征集合，变换到世界坐标系W
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // fuse the map
        /// 清空局部地图，再合并
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        // 遍历提取到的位姿节点
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 合并边缘点和平面点特征，构成 世界坐标系W下的局部特征点云地图
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
        }

        /// 降采样
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();
    }

    /**
     * @brief extractSurroundingKeyFrames
     * 关键帧点云合并
     *  1. 提取最新关键帧附近的关键帧
     *  2. 将这些临近关键帧的边缘点和平面点特征转换到世界坐标系
     *  3. 合并
     */
    void extractSurroundingKeyFrames()
    {
        // 如果关键帧容器为空
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (loopClosureEnableFlag == true)
        {
            // 回环提取？
            extractForLoopClosure();
        } else {
            // 从附近提取
            extractNearby();
        }
    }

    /// 对当前帧点云的边缘点、平面点集合进行降采样
    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        // transformTobeMapped： 预积分的位姿估计
        // 转换成Eagen类型 transPointAssociateToMap
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        // 更新关联点？
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            //pcl::PointXYZI
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 取当前帧点云——降采样后的边缘点集合中的一个点
            pointOri = laserCloudCornerLastDS->points[i];
            // 将激光雷达坐标系的点转换到世界坐标系
            pointAssociateToMap(&pointOri, &pointSel);
            // 在世界坐标系下的 局部边缘线特征地图中找到 与 pointSel 最临近的5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // 如果第5个点距离 pointSel < 1m
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                // 遍历这5个点
                for (int j = 0; j < 5; j++) {
                    // 把这5个点的xyz相加后，求均值
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                //求均方差
                //对点云协方差矩阵进行PCA主成分分析：若这五个点分布在直线上，协方差矩阵的特征值包含一个元素显著大于其余两个，
                //与该特征值相关的特征向量表示所处直线的方向；
                //若这五个点分布在平面上，协方差矩阵的特征值存在一个显著小的元素，与该特征值相关的特征向量表示所处平面的法线方向
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                //构建矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                //特征值分解
                cv::eigen(matA1, matD1, matV1);

                //如果最大的特征值大于第二大的特征值三倍以上
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {

                    //p0
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    //根据特征值分解，通过最大特征值对应的特征向量构建直线
                    //根据特征向量的方向构建x1和x2，两者连线代表最近点构成的直线
                    //p1
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    //p2
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 向量v1=[x0-x1,y0-y1,z0-z1] = [l , m , n]
                    // 向量v2=[x0-x2,y0-y2,z0-z2] = [o , p , q]
                    // (v1Xv2)=(mq-np,no-lq,lp-mo)^T
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // (p1-p2)=[x1-x2,y1-y2,z1-z2]
                    // 向量|p1-p2|的模长
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // (p1-p2)X(v1 X v2)
                    // 距离ld2对x0/y0/z0的偏导
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 点p0到边缘线距离
                    float ld2 = a012 / l12;

                    // 权重系数计算
                    float s = 1 - 0.9 * fabs(ld2);

                    // 距离对点(p0)的雅克比J
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    // 距离足够小才使用这个点
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        // 保存系数
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            //pcl::PointXYZI
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 取当前帧点云——降采样后的平面点集合中的一个点
            pointOri = laserCloudSurfLastDS->points[i];
            // 将该点转换到世界坐标系W
            pointAssociateToMap(&pointOri, &pointSel);
            // 在世界坐标系下的 局部平面点特征地图中找到 与 pointSel 最临近的5个点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 如果第5个点距离 该点 < 1m
            if (pointSearchSqDis[4] < 1.0) {
                // 遍历5个点
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                //! matA0                   matX0
                //! | x0 | y0 | z0 |        |X0_1|
                //! | x1 | y1 | z1 |        |X0_2|
                //! | x2 | y2 | z2 |        |X0_3|
                //! | x3 | y3 | z3 |
                //! | x4 | y4 | z4 |

                /// 由于这5个点在一个平面上，直接通过矩阵运算求解5个点构成平面的法向量 matX0
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // 法向量归一化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                bool planeValid = true;
                // 遍历这5个点
                for (int j = 0; j < 5; j++) {
                    // 如果任一个点到拟合出来的平面距离>0.2 直接否定这个平面
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 如果平面有效
                if (planeValid) {
                    // 计算 当前帧的平面点 到 这个平面的距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 权重系数
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    // pa pb pc 同时也是 点到平面距离对(x0,y0,z0)的偏导
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 将边缘点特征和平面点特征的 点以及对应的系数，一同储存在laserCloudOri，coeffSel
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        // 遍历当前帧点云-边缘点特征
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            // 如果这个边缘点特征有效
            if (laserCloudOriCornerFlag[i] == true){
                // 储存这个边缘点(在Lidar坐标系的)
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                // 储存对应的系数（由特征关联得到的），用来构建增量方程用的?
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        // 遍历当前帧点云-平面点特征
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 特征点有效标志位清空，准备下一帧扫描点云的到来
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    // 迭代优化，求解Lidar位姿
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 把激光雷达的位姿转换成相机位姿
        float srx = sin(transformTobeMapped[1]);    // pitch        sin(rx)
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);    // yaw          sin(ry)
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);    // roll         sin(rz)
        float crz = cos(transformTobeMapped[0]);

        // 如果特征点数太少，直接返回
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 遍历特征点
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            // 将特征点，从激光雷达坐标系变换到相机坐标系
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            // 这些系数，本来是 特征距离对 激光雷达坐标系下的当前帧点云中的点（x,y,z）求导
            // 现在，这些点都从激光雷达坐标系变换到相机坐标系中了，因此，求导结果也变了
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            /// qpc : modify： 这实际上就是 特征距离 对 Lidar位姿求导，整这些变换花里胡哨的没用!
            /// rz: roll
            /// rx: pitch
            /// ry: yaw
            /// pointOri.z : point.x
            /// pointOri.x : point.y
            /// pointOri.y : point.z
            /// coeff.x : coeffSel->points[i].y
            /// coeff.y : coeffSel->points[i].z
            /// coeff.z : coeffSel->points[i].x
            /// arx = (cp*sy*sr*pt.y + cp*cr*sy*pt.z - sp*sy*pt.x)* coeffSel->points[i].y +
            ///       (-sp*sr*pt.y - cr*sp*pt.z - cp*pt.x)* coeffSel->points[i].z +
            ///       (cp*cy*sr*pt.y + cp*cy*cr*pt.z - cy*sp*pt.x)* coeffSel->points[i].x
            ///
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            /// arz = ((cr*sp*sy-cy*sr)*pt.y+(-cy*cr-sp*sy*sr)*pt.y)*coeffSel->points[i].y+
            ///       ((cp*cr)*pt.y+(-cp*sr)*pt.z)*coeffSel->points[i].z+
            ///       ((sy*sr+cy*cr*sp)pt.y+(cr*sy-cy*sp*sr)*pt.z)*coeffSel->points[i].x
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera  ===> qpc commit： 这个注释很迷
            matA.at<float>(i, 0) = arz;                     // 特征距离d 对 Lidar位姿 roll求导
            matA.at<float>(i, 1) = arx;                     // 特征距离d 对 Lidar位姿 pitch求导
            matA.at<float>(i, 2) = ary;                     // 特征距离d 对 Lidar位姿 yaw求导
            matA.at<float>(i, 3) = coeff.z;                 // 特征距离d 对 Lidar位姿 tx求导
            matA.at<float>(i, 4) = coeff.x;                 // 特征距离d 对 Lidar位姿 ty求导
            matA.at<float>(i, 5) = coeff.y;                 // 特征距离d 对 Lidar位姿 tz求导
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // 求A^T
        cv::transpose(matA, matAt);
        // A^T*A ===> J^T J
        matAtA = matAt * matA;
        // J^T b
        matAtB = matAt * matB;
        // J^T J = -J^T b
        // QR分解求解: matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 如果
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 特征值分解
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            // TODO: 还没仔细看这个退化情况
            // 如果特征值 < 100 ， 出现退化情况
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 更新待优化变量（Lidar位姿）
        transformTobeMapped[0] += matX.at<float>(0, 0);     //roll
        transformTobeMapped[1] += matX.at<float>(1, 0);     //pitch
        transformTobeMapped[2] += matX.at<float>(2, 0);     //yaw
        transformTobeMapped[3] += matX.at<float>(3, 0);     //x
        transformTobeMapped[4] += matX.at<float>(4, 0);     //y
        transformTobeMapped[5] += matX.at<float>(5, 0);     //z

        // 检查是否收敛
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        // 如果位姿节点容器为空，直接返回
        //if (cloudKeyPoses3D->points.empty())
        //    return;

        // 如果边缘点特征数量 和 平面点特征数量都大于阈值
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 分别构建kd-tree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 遍历30次
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                // 获取与当前帧的边缘点 对应的 局部边缘点匹配
                // 并且计算特征距离，计算偏导
                cornerOptimization();

                // 获取与当前帧的边缘点 对应的 局部平面点匹配
                // 并且计算特征距离，计算偏导
                surfOptimization();

                // 将边缘点特征和平面点特征的 点以及对应的系数，一同储存在laserCloudOri，coeffSel
                combineOptimizationCoeffs();

                // 迭代优化?
                if (LMOptimization(iterCount) == true)
                    break;
            }

            transformUpdate();
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        // 如果该帧点云的IMU数据有效
        if (cloudInfo.imuAvailable == true)
        {
            // 并且IMU 初始化时的Pitch < 1.4 rad?
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = 0.05;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // 取LM优化后的 roll，设置
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                // 取该帧点云扫描起始时刻的 Lidar的roll姿态(从IMU的姿态转换到Lidar的姿态)
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                // slerp：球面线性插值（Spherical linear interpolation），是四元数的一种线性插值运算，主要用于在两个表示旋转的四元数之间平滑差值
                // 由于imuWeight设置为0.05,那么插值得到的应该是 0.95transformQuaternion+0.05imuQuaternion (个人大概认为，还没看这种插值的原理)
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                // 更新roll
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }
        // 检查是否超过限定的值，
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
    }
    // 上界
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;
        // 取上一帧点云对应的Lidar位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 取当前帧点云的LM优化的Lidar位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 得到：当前帧到上一帧的变换
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        // 分解
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 检查是否超过阈值
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    // 添加Lidar里程计因子
    void addOdomFactor()
    {
        // 如果是第一帧
        if (cloudKeyPoses3D->points.empty())
        {
            // 噪声模型 （roll,pitch,yaw,x,y,z）
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 添加先验因子 （roll,pitch,yaw,x,y,z）
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 设定Values （gtsam特有）,表明待优化变量是这个
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // 如果不是第一帧
            // 噪声模型
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 取上一个关键帧 因子图优化后的位姿
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            // 取当前帧点云，Lidar里程计位姿(LM优化的位姿)
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // poseFrom.between(poseTo) 当前帧位姿 到 上一个关键帧 的变换
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 插入到Values， 表明待优化变量是这个
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        // 如果关键帧位姿队列为空，直接返回?
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            // 如果关键帧队列最前的节点与最后的节点，距离<5m，也直接返回
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        // 如果
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // gps数据时间戳比 点云 早0.2秒以上， GPS太旧了
            if (gpsQueue.front().header.stamp.toSec() < timeLaserCloudInfoLast - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserCloudInfoLast + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                // gps时间戳与点云时间戳同步了

                // 取gps数据
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                // 检查协方差
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    // 如果不使用GPS的海拔，那就用Lidar里程计的
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                // 检查距离上一次添加gps因子是否超过 5米
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                // 添加GPS因子
                // 噪声模型
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                // GPS因子
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    //
    void saveKeyFramesAndFactor()
    {
        // 检查是否关键帧
        if (saveFrame() == false)
            return;

        /// 是关键帧，准备因子

        // odom factor
        // Lidar里程计因子(特征匹配LM优化得到的)
        addOdomFactor();

        // gps factor
        // GPS因子
        addGPSFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        // iSAM2算法：优化因子图
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // update multiple-times till converge
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // save key poses
        // 取优化后的数据
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 记录当前帧LM位姿经过优化后的结果
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserCloudInfoLast;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 边缘化，得到协方差矩阵
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // 更新位姿
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 保存当前帧的边缘点特征、平面点特征 两组点云数据(Lidar坐标系)
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        updatePath(thisPose6D);

        // TODO：需要注意的是，这里优化完之后，只对最新关键帧的位姿进行了更新
        //       但是实际上，因子图优化有可能会改变图中的某些节点的位姿，这里没有更新，而是在发生回环之后才更新
    }

    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // 如果发生回环?
        if (aLoopIsClosed == true)
        {
            // clear path
            // 清除path，需要更新全局位姿
            globalPath.poses.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            // 取新的优化结果，更新全局的位姿
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 更新path
                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
            // ID for reseting IMU pre-integration
            ++imuPreintegrationResetId;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        // 将优化后的(当前帧点云对应的位姿)记录到path中
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = timeLaserInfoStamp;
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    // 发布最新的位姿(优化后、回环矫正后)
    void publishOdometry()
    {
        // Publish odometry for ROS
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = "odom";
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        pubOdomAftMappedROS.publish(laserOdometryROS);
    }

    void publishFrames()
    {
//        if (cloudKeyPoses3D->points.empty())
//            return;
        // publish key poses
        // 发布所有关键帧位置(使用点云形式储存发布)
        //publishCloud(&pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, "odom");
        // Publish surrounding key frames
        // 发布世界坐标系下的 局部平面点特征地图
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
        // publish registered key frame
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            // 取最新优化后位姿
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            // 将特征点合并，然后转换到世界坐标系下，发布
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish registered high-res raw cloud
        // 发布原始高分辨率地图
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            // 取原始点云
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            // 取最新优化后位姿
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 转换到世界坐标系下，发布
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
        }
//        // publish path
//        if (pubPath.getNumSubscribers() != 0)
//        {
//            // 发布全局轨迹
//            globalPath.header.stamp = timeLaserInfoStamp;
//            globalPath.header.frame_id = "odom";
//            pubPath.publish(globalPath);
//        }
    }
};
#endif // MAPOPTMIZATION_H

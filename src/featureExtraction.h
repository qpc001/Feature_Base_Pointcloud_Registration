#ifndef FEATUREEXTRACTION_H
#define FEATUREEXTRACTION_H


#include "utility.h"
#include "feature_matching/cloud_info.h"

struct smoothness_t{
    float value;
    size_t ind;
};

struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;         // 去畸变点云
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    feature_matching::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;               //筛选过滤掉的点的id
    int *cloudLabel;    //点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)

    /**
     * @brief FeatureExtraction
     * 构造函数
     */
    FeatureExtraction()
    {
        // 订阅 自定义的点云msg: lio_sam/deskew/cloud_info
        //subLaserCloudInfo = nh.subscribe<feature_matching::cloud_info>("lio_sam_custom/deskew/cloud_info", 100, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布
        // 提取的特征点云lio_sam/feature/cloud_info
        pubLaserCloudInfo = nh.advertise<feature_matching::cloud_info> ("lio_sam_custom/feature/cloud_info", 100);
        // 角点、平面点
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/feature/cloud_corner", 10);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam_custom/feature/cloud_surface", 10);

        // 成员变量初始化
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void featureExtra(const feature_matching::cloud_info &cloud_info_){
        laserCloudInfoHandler(cloud_info_);
    }

    /**
     * @brief laserCloudInfoHandler
     * lio_sam/deskew/cloud_info 消息的回调函数
     * @param msgIn
     */
    void laserCloudInfoHandler(const feature_matching::cloud_info& msgIn)
    {
        cloudInfo = msgIn; // new cloud info
        cloudHeader = msgIn.header; // new cloud header
        pcl::fromROSMsg(msgIn.cloud_deskewed, *extractedCloud); // new cloud for extraction

        //计算点云粗糙度
        calculateSmoothness();

        //挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住
        markOccludedPoints();

        extractFeatures();

        publishFeatureCloud();
    }

    /**
     * @brief calculateSmoothness
     * 计算点云粗糙度
     */
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        // 取前后5个点，计算粗糙度
        for (int i = 5; i < cloudSize - 5; i++) //使用每个点的前后五个点计算曲率，因此前五个与最后五个点跳过
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            //初始化为less flat点
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    //挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住
    void markOccludedPoints()
    {
        //挑选点，排除容易被斜面挡住的点以及离群点，有些点容易被斜面挡住，
        //而离群点可能出现带有偶然性，这些情况都可能导致前后两次扫描不能被同时看到
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i) //与后一个点差值，所以减6
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            // 后一个点的列号与当前点列号差不能大于10个像素
            if (columnDiff < 10){
                // 10 pixel diff in range image
                // 按照两点的深度的比例，将深度较大的点拉回后计算距离
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam（检查深度值是否跳变，是则很可能出现遮挡）
            // 点i与点i-1相对于Lidar的距离差
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            // 点i与点i+1相对于Lidar的距离差
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        //将每条线上的点分入相应的类别：边沿点和平面点
        // 按线束号遍历
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
            //将每个scan的曲率点分成6等份处理,确保周围都有点被选作特征点
            for (int j = 0; j < 6; j++)
            {
                //六等份起点：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                //六等份终点：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                /// 按曲率从小到大排序（这个写法可以）
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                int largestPickedNum = 0;

                //挑选每个分段的曲率很大和比较大的点
                for (int k = ep; k >= sp; k--)
                {
                    //曲率最大点的点序  cloudSortInd: 从小到大排序，k从大到小递减
                    int ind = cloudSmoothness[k].ind;
                    //如果曲率大的点，曲率的确比较大，并且未被筛选过滤掉
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        //挑选曲率最大的前20个点放入sharp点集合
                        if (largestPickedNum <= 20){
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        // 标记这个点已经选过了，进入了 cornerCloud 中了
                        cloudNeighborPicked[ind] = 1;
                        //将曲率比较大的点的前后各5个连续距离比较近的点筛选出去，防止特征点聚集，使得特征点在每个方向上尽量分布均匀
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                /// 挑选每个分段的曲率很小比较小的点
                for (int k = sp; k <= ep; k++)
                {
                    //cloudSortInd: 从小到大排序，k从小到大递增
                    int ind = cloudSmoothness[k].ind;
                    //如果曲率的确比较小，并且未被筛选出
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        //-1代表曲率很小的点
                        cloudLabel[ind] = -1;
                        // 标记这个点已经选过了
                        cloudNeighborPicked[ind] = 1;

                        //同样防止特征点聚集
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // ？？？？ 为什么不放到上面呢？
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            // 挑选的平面点可能有点密，需要滤波稀疏化
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    void freeCloudInfoMemory()
    {
        // 释放掉msg中后面不需要的数据
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    // 发布提取的边沿点、平面点特征
    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner  = publishCloud(&pubCornerPoints,  cornerCloud,  cloudHeader.stamp, "base_link");
        cloudInfo.cloud_surface = publishCloud(&pubSurfacePoints, surfaceCloud, cloudHeader.stamp, "base_link");
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};
#endif // FEATUREEXTRACTION_H

#include <iostream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/segmentation/region_growing.h>
#include <pcl/console/time.h>

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("005/005_from_ply.pcd", *cloud) == -1)
    {
        std::cout << "Cloud reading failed." << std::endl;
        return (-1);
    }

    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(50);
    normal_estimator.compute(*normals);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(10000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(50);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.5);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;
    std::cout << "These are the indices of the points of the initial" << std::endl
              << "cloud that belong to the first cluster:" << std::endl;
    std::size_t counter = 0;
    while (counter < clusters[0].indices.size())
    {
        std::cout << clusters[0].indices[counter] << ", ";
        counter++;
        if (counter % 10 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    // pcl::visualization::CloudViewer viewer("Cluster viewer");
    // viewer.showCloud(colored_cloud);
    // while (!viewer.wasStopped())
    // {
    // }
    pcl::console::TicToc tt;
    std::cerr << "Saving...\n", tt.tic();
    pcl::io::savePCDFile("region_out.pcd", *colored_cloud);
    std::cerr << ">> Done: " << tt.toc() << " ms\n";

    return (0);
}
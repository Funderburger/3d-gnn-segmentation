#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/features/fpfh.h>

#include <boost/filesystem.hpp>                 // for path, exists, ...
#include <boost/algorithm/string/case_conv.hpp> // for to_upper_copy

#include <regex>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
// typedef pcl::PointXYZRGB PointType;

int default_k = 0;
double default_radius = 0.0;
int default_samples = 0;

void printHelp(int, char **argv)
{
  print_error("Syntax is: %s input.pcd output.pcd <options> [optional_arguments]\n", argv[0]);
  print_info("  where options are:\n");
  print_info("                     -radius X = use a radius of Xm around each point to determine the neighborhood (default: ");
  print_value("%f", default_radius);
  print_info(")\n");
  print_info("                     -k X      = use a fixed number of X-nearest neighbors around each point (default: ");
  print_value("%f", default_k);
  print_info(")\n");
  print_info("                     -samples X      = down sample point clouds to a fix number of points (random sample) (default: ");
  print_value("%f", default_samples);
  print_info(")\n");
  print_info(" For organized datasets, an IntegralImageNormalEstimation approach will be used, with the RADIUS given value as SMOOTHING SIZE.\n");
  print_info("\nOptional arguments are:\n");
  print_info("                     -input_dir X  = batch process all PCD files found in input_dir\n");
  print_info("                     -output_dir X = save the processed files from input_dir in this directory\n");
}

bool loadCloud(const std::string &filename, pcl::PCLPointCloud2 &cloud,
               Eigen::Vector4f &translation, Eigen::Quaternionf &orientation)
{
  if (loadPCDFile(filename, cloud, translation, orientation) < 0)
    return (false);

  return (true);
}

void compute(const pcl::PCLPointCloud2::ConstPtr &input, pcl::PCLPointCloud2 &output,
             int k, double radius, int samples)
{
  // Convert data to PointCloud<T>
  // PointCloud<PointXYZ>::Ptr xyz (new PointCloud<PointXYZ>);
  // fromPCLPointCloud2 (*input, *xyz);
  PointCloud<PointXYZRGBL>::Ptr xyz(new PointCloud<PointXYZRGBL>);
  fromPCLPointCloud2(*input, *xyz);

  // ***********************************
  // * sample before normal estimation *
  // ***********************************
  // Test the pcl::PCLPointCloud2 method
  // Randomly sample sample points from cloud
  TicToc ds_tt;
  ds_tt.tic();

  RandomSample<pcl::PCLPointCloud2> random_sample;
  random_sample.setInputCloud(input);
  random_sample.setSample(samples);

  // Indices
  pcl::Indices indices;
  random_sample.filter(indices);

  random_sample.filter(output);
  fromPCLPointCloud2(output, *xyz);

  print_highlight("Computed random down sampling in ");
  print_value("%g", ds_tt.toc());
  print_info(" ms for ");
  print_value("%d", xyz->size());
  print_info(" points.\n");
  // *************************************
  // *************************************

  // * ***************** *
  // * normal estimation *
  // * ***************** *

  TicToc tt;
  tt.tic();

  PointCloud<Normal> normals;

  // Try our luck with organized integral image based normal estimation
  if (xyz->isOrganized())
  {
    IntegralImageNormalEstimation<PointXYZRGBL, Normal> ne;
    ne.setInputCloud(xyz);
    ne.setNormalEstimationMethod(IntegralImageNormalEstimation<PointXYZRGBL, Normal>::COVARIANCE_MATRIX);
    ne.setNormalSmoothingSize(float(radius));
    ne.setDepthDependentSmoothing(true);
    ne.compute(normals);
  }
  else
  {
    NormalEstimation<PointXYZRGBL, Normal> ne;
    ne.setInputCloud(xyz);
    ne.setSearchMethod(search::KdTree<PointXYZRGBL>::Ptr(new search::KdTree<PointXYZRGBL>));
    ne.setKSearch(k);
    ne.setRadiusSearch(radius);
    ne.compute(normals);
  }

  print_highlight("Computed normals in ");
  print_value("%g", tt.toc());
  print_info(" ms for ");
  print_value("%d", normals.width * normals.height);
  print_info(" points.\n");

  // *****************************************
  // *****************************************

  pcl::PCLPointCloud2::Ptr cloudNormal (new pcl::PCLPointCloud2());
  pcl::PCLPointCloud2 output_normals;
  toPCLPointCloud2 (normals, output_normals);
  concatenateFields (output, output_normals, *cloudNormal);

  // ## we chose not to do this ##
  // **********************************
  // * sample after normal estimation *
  // **********************************
  // // Test the pcl::PCLPointCloud2 method
  // // Randomly sample sample points from cloud
  // // toPCLPointCloud2(*xyz, *cloudNormal);
  // concatenateFields (*input, output_normals, *cloudNormal);
  // RandomSample<pcl::PCLPointCloud2> random_sample;
  // random_sample.setInputCloud(cloudNormal);
  // random_sample.setSample(samples);

  // // Indices
  // pcl::Indices indices;
  // random_sample.filter(indices);

  // // Cloud
  
  // random_sample.filter(output);
  // ***********************************
  // ***********************************

  // // Create the FPFH estimation class, and pass the input dataset+normals to it
  // pcl::FPFHEstimation<pcl::PointXYZRGBL, pcl::Normal, pcl::FPFHSignature33> fpfh;
  // fpfh.setInputCloud (xyz);
  // fpfh.setInputNormals (normals);
  // // alternatively, if cloud is of tpe PointNormal, do fpfh.setInputNormals (cloud);

  // // Create an empty kdtree representation, and pass it to the FPFH estimation object.
  // // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  // pcl::search::KdTree<PointXYZ>::Ptr tree (new pcl::search::KdTree<PointXYZ>);

  // fpfh.setSearchMethod (tree);

  // // Output datasets
  // pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // // Use all neighbors in a sphere of radius 5cm
  // // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  // fpfh.setRadiusSearch (0.05);

  // // Compute the features
  // fpfh.compute (*fpfhs);

  // fpfhs->size () should have the same size as the input cloud->size ()*
  
  copyPointCloud(*cloudNormal, output);
  // Convert data back

}

void saveCloud(const std::string &filename, const pcl::PCLPointCloud2 &output,
               const Eigen::Vector4f &translation, const Eigen::Quaternionf &orientation)
{
  PCDWriter w;
  w.writeBinaryCompressed(filename, output, translation, orientation);
}

int batchProcess(const std::vector<std::string> &pcd_files, std::string &output_dir, int k, double radius, int samples)
{
#pragma omp parallel for default(none) \
    shared(k, output_dir, pcd_files, radius)
  for (int i = 0; i < int(pcd_files.size()); ++i)
  {
    // Load the first file
    Eigen::Vector4f translation;
    Eigen::Quaternionf rotation;
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
    if (!loadCloud(pcd_files[i], *cloud, translation, rotation))
      continue;

    // Perform the feature estimation
    pcl::PCLPointCloud2 output;
    compute(cloud, output, k, radius, samples);

    // Prepare output file name
    std::string filename = boost::filesystem::path(pcd_files[i]).filename().string();

    // Save into the second file
    const std::string filepath = output_dir + '/' + filename;
    saveCloud(filepath, output, translation, rotation);
  }
  return (0);
}

/* ---[ */
int main(int argc, char **argv)
{
  print_info("Estimate surface normals using NormalEstimation. For more information, use: %s -h\n", argv[0]);

  if (argc < 3)
  {
    printHelp(argc, argv);
    return (-1);
  }

  bool batch_mode = false;

  // Command line parsing
  int k = default_k;
  double radius = default_radius;
  int samples = default_samples;
  parse_argument(argc, argv, "-k", k);
  parse_argument(argc, argv, "-radius", radius);
  parse_argument(argc, argv, "-samples", samples);
  std::string input_dir, output_dir;
  if (parse_argument(argc, argv, "-input_dir", input_dir) != -1)
  {
    PCL_INFO("Input directory given as %s. Batch process mode on.\n", input_dir.c_str());
    if (parse_argument(argc, argv, "-output_dir", output_dir) == -1)
    {
      PCL_ERROR("Need an output directory! Please use -output_dir to continue.\n");
      return (-1);
    }

    // Both input dir and output dir given, switch into batch processing mode
    batch_mode = true;
  }

  if (!batch_mode)
  {
    // Parse the command line arguments for .pcd files
    std::vector<int> p_file_indices;
    p_file_indices = parse_file_extension_argument(argc, argv, ".pcd");
    if (p_file_indices.size() != 2)
    {
      print_error("Need one input PCD file and one output PCD file to continue.\n");
      return (-1);
    }

    print_info("Estimating normals with a k/radius/smoothing size of: ");
    print_value("%d / %f / %f\n", k, radius, radius);

    // Load the first file
    Eigen::Vector4f translation;
    Eigen::Quaternionf rotation;
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
    if (!loadCloud(argv[p_file_indices[0]], *cloud, translation, rotation))
      return (-1);

    // Load the labeled pcd file
    Eigen::Vector4f translationL;
    Eigen::Quaternionf rotationL;
    pcl::PCLPointCloud2::Ptr cloudL(new pcl::PCLPointCloud2);
    std::string secondFile = std::regex_replace(argv[p_file_indices[0]], std::regex("pcd_rgb_data"), "pcd_label_data");
    if (!loadCloud(secondFile, *cloudL, translationL, rotationL))
      return (-1);

    pcl::PointCloud<PointXYZI> cloudIntensity;
    fromPCLPointCloud2(*cloudL, cloudIntensity);
    pcl::PointCloud<pcl::PointXYZL> cloudLabel; //new pcl::PointCloud<pcl::PointXYZL>);
    cloudLabel.width = cloudIntensity.width;
    cloudLabel.height = cloudIntensity.height;
    cloudLabel.resize(cloudLabel.width * cloudLabel.height);
    cloudLabel.is_dense = true;


    for (size_t i = 0; i < cloudIntensity.points.size(); i++)
    {
      cloudLabel.points[i].x = cloudIntensity.points[i].x;
      cloudLabel.points[i].y = cloudIntensity.points[i].y;
      cloudLabel.points[i].z = cloudIntensity.points[i].z;
      cloudLabel.points[i].label = cloudIntensity.points[i].intensity;
      // cloudLabel->points[i].x = cloudIntensity.points[i].x;
      // cloudLabel->points[i].y = cloudIntensity.points[i].y;
      // cloudLabel->points[i].z = cloudIntensity.points[i].z;
      // cloudLabel->points[i].label = cloudIntensity.points[i].intensity;
    }

    toPCLPointCloud2 (cloudLabel, *cloudL);

    pcl::PCLPointCloud2::Ptr completeCloud(new pcl::PCLPointCloud2);
    // pcl::PCLPointCloud2 intensity;
    // PointCloud<Intensity> intensityLabel (cloudL);
    // toPCLPointCloud2 (intensityLabel, intensity);
    // cloud->concatenate(*cloud,*cloudL); // concatenate(*cloud,&cloudL);
    concatenateFields(*cloud, *cloudL, *completeCloud);

    // *cloud = *cloud + *cloudL;
    // Perform the feature estimation
    pcl::PCLPointCloud2 output;
    compute(completeCloud, output, k, radius, samples);

    // Save into the second file
    saveCloud(argv[p_file_indices[1]], output, translation, rotation);
  }
  else
  {
    if (!input_dir.empty() && boost::filesystem::exists(input_dir))
    {
      std::vector<std::string> pcd_files;
      boost::filesystem::directory_iterator end_itr;
      for (boost::filesystem::directory_iterator itr(input_dir); itr != end_itr; ++itr)
      {
        // Only add PCD files
        if (!is_directory(itr->status()) && boost::algorithm::to_upper_copy(boost::filesystem::extension(itr->path())) == ".PCD")
        {
          pcd_files.push_back(itr->path().string());
          PCL_INFO("[Batch processing mode] Added %s for processing.\n", itr->path().string().c_str());
        }
      }
      batchProcess(pcd_files, output_dir, k, radius, samples);
    }
    else
    {
      PCL_ERROR("Batch processing mode enabled, but invalid input directory (%s) given!\n", input_dir.c_str());
      return (-1);
    }
  }
}

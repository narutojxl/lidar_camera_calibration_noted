#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <ros/package.h>

#include <pcl_ros/point_cloud.h>
#include <boost/foreach.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <velodyne_pointcloud/point_types.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/common/intersections.h>


#include "lidar_camera_calibration/Utils.h"

int iteration_count = 0; //执行多少次回调函数
std::vector< std::vector<cv::Point> > stored_corners;

/**
 * @param scan:  filtered points in rgb_camera_optical_frame 
 */
bool getCorners(cv::Mat img, pcl::PointCloud<pcl::PointXYZ> scan, cv::Mat P, int num_of_markers, int MAX_ITERS, Eigen::Matrix4d T)
{

	//ROS_INFO_STREAM("iteration number: " << iteration_count << "\n");

	/*Masking happens here */
	cv::Mat edge_mask = cv::Mat::zeros(img.size(), CV_8UC1);//单通道
	//edge_mask(cv::Rect(520, 205, 300, 250))=1;
	edge_mask(cv::Rect(0, 0, img.cols, img.rows))=1; //将roi区域置为1
	img.copyTo(edge_mask, edge_mask);
	//pcl::io::savePCDFileASCII ("/home/vishnu/final1.pcd", scan.point_cloud);

	img = edge_mask; //全1像素

	//cv:imwrite("/home/vishnu/marker.png", edge_mask);

	pcl::PointCloud<pcl::PointXYZ> pc = scan;
	//scan = Velodyne::Velodyne(filtered_pc);

	cv::Rect frame(0, 0, img.cols, img.rows);
	
	//pcl::io::savePCDFileASCII("/home/vishnu/final2.pcd", scan.point_cloud);
	
	//向图像投影
	cv::Mat image_edge_laser = project(P, frame, scan, NULL); //投影到camera image上的pixel灰度值为250，没投影到的像素灰度为0
	cv::threshold(image_edge_laser, image_edge_laser, 10, 255, 0); //type = CV_THRESH_BINARY，强度>10,为255；强度<=10,为0
	//阈值化后，投影到image上的像素强度为255，没成像的点的像素值为0


	

	cv::Mat combined_rgb_laser;
	std::vector<cv::Mat> rgb_laser_channels;

	rgb_laser_channels.push_back(image_edge_laser);
	rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1)); //全0像素
	rgb_laser_channels.push_back(img);
			 
	cv::merge(rgb_laser_channels, combined_rgb_laser); //多个单通道图像合成一幅多通道图像。
	/*cv::namedWindow("combined", cv::WINDOW_NORMAL); 
	cv::imshow("combined", combined_rgb_laser);
	cv::waitKey(5);
	*/

	std::map<std::pair<int, int>, std::vector<float> > c2D_to_3D;
	std::vector<float> point_3D;

	/* store correspondences */
	for(pcl::PointCloud<pcl::PointXYZ>::iterator pt = pc.points.begin(); pt < pc.points.end(); pt++)
	{

			// behind the camera
			if (pt->z < 0)
			{
				continue;
			}

			cv::Point xy = project(*pt, P);
			if (xy.inside(frame))
			{
				//create a map of 2D and 3D points
				point_3D.clear();
				point_3D.push_back(pt->x);
				point_3D.push_back(pt->y);
				point_3D.push_back(pt->z);
				c2D_to_3D[std::pair<int, int>(xy.x, xy.y)] = point_3D;
			}
	}

	/* print the correspondences */
	/*for(std::map<std::pair<int, int>, std::vector<float> >::iterator it=c2D_to_3D.begin(); it!=c2D_to_3D.end(); ++it)
	{
		std::cout << it->first.first << "," << it->first.second << " --> " << it->second[0] << "," <<it->second[1] << "," <<it->second[2] << "\n";
	}*/

	/* get region of interest */

	const int QUADS=num_of_markers; //2
	std::vector<int> LINE_SEGMENTS(QUADS, 4); //assuming each has 4 edges and 4 corners

	pcl::PointCloud<pcl::PointXYZ>::Ptr board_corners(new pcl::PointCloud<pcl::PointXYZ>); //
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr marker(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<cv::Point3f> c_3D;
	std::vector<cv::Point2f> c_2D;


	cv::namedWindow("cloud", cv::WINDOW_NORMAL);
	cv::namedWindow("polygon", cv::WINDOW_NORMAL); 
	//cv::namedWindow("combined", cv::WINDOW_NORMAL); 

	std::string pkg_loc = ros::package::getPath("lidar_camera_calibration");
	std::ofstream outfile(pkg_loc + "/conf/points.txt", std::ios_base::trunc);
	outfile << QUADS*4 << "\n";

	for(int q=0; q<QUADS; q++)
	{
		std::cout << "---------Moving on to next marker--------\n";
		std::vector<Eigen::VectorXf> line_model; //4条直线模型
		for(int i=0; i<LINE_SEGMENTS[q]; i++)//对于每一条边
		{
			cv::Point _point_;
			std::vector<cv::Point> polygon;
			int collected;

			// get markings in the first iteration only
			if(iteration_count == 0)
			{
				polygon.clear();
				collected = 0;
				while(collected != LINE_SEGMENTS[q])
				{
					
						cv::setMouseCallback("cloud", onMouse, &_point_); //用鼠标左键在投影的图像上点击的像素(x,y)保存在 _point_ 中
						
						cv::imshow("cloud", image_edge_laser);
						cv::waitKey(0);
						++collected;
						//std::cout << _point_.x << " " << _point_.y << "\n";
						polygon.push_back(_point_);
				}
				stored_corners.push_back(polygon);
			}
			
			polygon = stored_corners[4*q+i];

			cv::Mat polygon_image = cv::Mat::zeros(image_edge_laser.size(), CV_8UC1);
			
			rgb_laser_channels.clear();
			rgb_laser_channels.push_back(image_edge_laser);
			rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1));
			rgb_laser_channels.push_back(cv::Mat::zeros(image_edge_laser.size(), CV_8UC1));
			cv::merge(rgb_laser_channels, combined_rgb_laser);
				
			for( int j = 0; j < 4; j++ )
			{
				cv::line(combined_rgb_laser, polygon[j], polygon[(j+1)%4], cv::Scalar(0, 255, 0)); //绿色直线
			}

			// initialize PointClouds
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr final (new pcl::PointCloud<pcl::PointXYZ>);

			for(std::map<std::pair<int, int>, std::vector<float> >::iterator it=c2D_to_3D.begin(); it!=c2D_to_3D.end(); ++it)
			{
			    //测试一个点是否在多边形中
				if (cv::pointPolygonTest(cv::Mat(polygon), cv::Point(it->first.first, it->first.second), true) > 0)
				{
					cloud->push_back(pcl::PointXYZ(it->second[0],it->second[1],it->second[2]));
					rectangle(combined_rgb_laser, cv::Point(it->first.first, it->first.second), cv::Point(it->first.first, it->first.second), cv::Scalar(0, 0, 255), 3, 8, 0); // RED point
				}
			}
			
			if(cloud->size() < 2){ 
				std::cout<<"Fit a line need at least 2 points.\n";
				return false;
			}

			if(iteration_count ==0){
			     cv::imshow("polygon", combined_rgb_laser);
			     cv::waitKey(4);
			}

			//pcl::io::savePCDFileASCII("/home/vishnu/line_cloud.pcd", *cloud);
			
			std::vector<int> inliers; //内点的index
			Eigen::VectorXf model_coefficients;


		    // TODO we add
		    //transform cloud back into laser frame
			//在直线拟合前， 不论是通过粗糙的变换初值，还是通过精确的变换初值， 点最终转换到了在laser坐标系下的
			//粗糙的初值还是精确的初值只影响在camera下的成的像，即我们是在准确的图像上标记marker的4个边还是在粗糙的图像上标记
			//所以初值的精确程度不影响标定结果的精度
		    Eigen::Transform<double, 3, Eigen::Affine> Affine_T (T);
			pcl::PointCloud<pcl::PointXYZ>::Ptr    new_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::transformPointCloud(*cloud,  *new_cloud,  Affine_T);
		    pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (new_cloud));

			// created RandomSampleConsensus object and compute the appropriated model
			//pcl::SampleConsensusModelLine<pcl::PointXYZ>::Ptr model_l(new pcl::SampleConsensusModelLine<pcl::PointXYZ> (cloud));


			pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_l);
			ransac.setDistanceThreshold (0.01);
			ransac.computeModel();
			ransac.getInliers(inliers);
			ransac.getModelCoefficients(model_coefficients);
			line_model.push_back(model_coefficients); //依次压入top-left边，top-right，down-right, down-left边的模型

			std::cout << "Line coefficients are:" << "\n" << model_coefficients << "\n";
			// copies all inliers of the model computed to another PointCloud
			pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *final); //final: 找到计算该直线模型的inliers points
			//pcl::io::savePCDFileASCII("/home/vishnu/RANSAC_line_cloud.pcd", *final);
			*marker += *final;
		} //end 4条边


		
		/* calculate approximate intersection of lines */
		

		Eigen::Vector4f p1, p2, p_intersect; //最小公垂线的两个垂足, 中点（第4维为0）
		pcl::PointCloud<pcl::PointXYZ>::Ptr corners(new pcl::PointCloud<pcl::PointXYZ>);
		for(int i=0; i<LINE_SEGMENTS[q]; i++)
		{
			pcl::lineToLineSegment(line_model[i], line_model[(i+1)%LINE_SEGMENTS[q]], p1, p2);
			for(int j=0; j<4; j++)
			{
				p_intersect(j) = (p1(j) + p2(j))/2.0;  //依次计算top-left边，top-right边交点； top-right，down-right边交点; etc
			}
			c_3D.push_back(cv::Point3f(p_intersect(0), p_intersect(1), p_intersect(2)));
			corners->push_back(pcl::PointXYZ(p_intersect(0), p_intersect(1), p_intersect(2)));
			//std::cout << "Point of intersection is approximately: \n" << p_intersect << "\n";
			//std::cout << "Distance between the lines: " << (p1 - p2).squaredNorm () << "\n";
			//std::cout << p_intersect(0) << " " << p_intersect(1) << " " << p_intersect(2) <<  "\n";
			outfile << p_intersect(0) << " " << p_intersect(1) << " " << p_intersect(2) <<  "\n"; 
			//标定板的4个角，in rgb_camera_optical_frame, 写入到文件中。

		}
		
		*board_corners += *corners; //标定板的4个角，in rgb_camera_optical_frame.

		std::cout << "Distance between the corners:\n";
		for(int i=0; i<4; i++)
		{
			std::cout << sqrt(
						  pow(c_3D[4*q+i].x - c_3D[4*q+(i+1)%4].x, 2)
						+ pow(c_3D[4*q+i].y - c_3D[4*q+(i+1)%4].y, 2)
						+ pow(c_3D[4*q+i].z - c_3D[4*q+(i+1)%4].z, 2)
						)<< std::endl;
		}
	}
	outfile.close();

	if(iteration_count ==0){
	    cv::destroyWindow("cloud");
	    cv::destroyWindow("polygon");
	}

	if(iteration_count == MAX_ITERS)
	{
		ros::shutdown();
	}
    
	iteration_count++;
	return true;
	/* store point cloud with intersection points */
	// pcl::io::savePCDFileASCII("/home/jxl/RANSAC_marker.pcd", *marker);
	// pcl::io::savePCDFileASCII("/home/jxl/RANSAC_corners.pcd", *board_corners);
}

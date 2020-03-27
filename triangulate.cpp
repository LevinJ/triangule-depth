//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
using namespace std;

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};

int main_execute(double &pos_err, double &ratio, bool add_noise = false, double pixel_noise = 2,int poseNums = 10, bool tracing = true){


	double radius = 8;
//    double fx = 1.;
//    double fy = 1.;
	std::vector<Pose> camera_pose;
	for(int n = 0; n < poseNums; ++n ) {
		double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
		// 绕 z轴 旋转
		Eigen::Matrix3d R;
		R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
		Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
		camera_pose.push_back(Pose(R,t));
	}

	// 随机数生成 1 个 三维特征点
	std::default_random_engine generator;
	std::uniform_real_distribution<double> xy_rand(-4, 4.0);
	std::uniform_real_distribution<double> z_rand(8., 10.);
	double tx = xy_rand(generator);
	double ty = xy_rand(generator);
	double tz = z_rand(generator);

	Eigen::Vector3d Pw(tx, ty, tz);
	// 这个特征从第三帧相机开始被观测，i=3
	int start_frame_id = 3;
	int end_frame_id = poseNums;

	double fx = 2000;
	double uv_noise = pixel_noise * 1. / fx;
	std::normal_distribution<double> noise_pdf(0., uv_noise);
	std::uniform_real_distribution<double> uv_rand(8., 10.);
	for (int i = start_frame_id; i < end_frame_id; ++i) {
		Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
		Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);

		Pc = Pc / Pc.z();  // 归一化图像平面
		if(add_noise){
			Pc[0] += noise_pdf(generator);
			Pc[1] += noise_pdf(generator);
		}

		camera_pose[i].uv = Eigen::Vector2d(Pc[0],Pc[1]);
	}

	/// TODO::homework; 请完成三角化估计深度的代码
	// 遍历所有的观测数据，并三角化
	Eigen::Vector3d P_est;           // 结果保存到这个变量
	P_est.setZero();

	int num_frames = end_frame_id - start_frame_id;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D(num_frames*2, 4);
	/* your code begin */
	//fill in the D matrix
	for (int i = start_frame_id; i < end_frame_id; ++i) {
		Eigen::Matrix<double, 3, 4> Pk;
		auto Rcw = camera_pose[i].Rwc.transpose();
		auto tcw = -Rcw *camera_pose[i].twc ;
		Pk.block(0, 0, 3, 3) = Rcw;
		Pk.col(3) = tcw;
		double uk = camera_pose[i].uv(0);
		double vk = camera_pose[i].uv(1);
		Eigen::Matrix<double, 2, 4> dk;
		dk.row(0).noalias()=uk * Pk.row(2) - Pk.row(0);
		dk.row(1).noalias()=vk * Pk.row(2) - Pk.row(1);
		D.block((i-start_frame_id)*2,0,2,4).noalias() = dk;
	}
	//solve the Dy=0 problem
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(D.transpose() * D, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Vector4d lambda = svd.singularValues();

	ratio = lambda(3)/lambda(2);

	if( ratio > 1e-2){
		cout<<"failure"<<endl;
		return -1;
	}

	Eigen::Vector4d u4 = svd.matrixU().block(0,3, 4,1);
	if(u4(3)!=0 && u4(2)/u4(3) > 0){
		P_est(0) = u4(0)/u4(3);
		P_est(1) = u4(1)/u4(3);
		P_est(2) = u4(2)/u4(3);
	}

 
	/* your code end */
	pos_err = (Pw - P_est).norm();
	if(tracing){
		cout<<"sigular values="<<lambda.transpose()<<endl;
		cout<<"sigular value ratio = "<<ratio<<endl;
		std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
		std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;
		std::cout<<"error=" <<pos_err<<endl;
	}

	// TODO:: 请如课程讲解中提到的判断三角化结果好坏的方式，绘制奇异值比值变化曲线
}
void test_uv_nosie(){
	double pos_err;
	double ratio;
	bool add_noise = true;
	int poseNums = 10;
	bool tracing = false;
	cout<<"pixel_noise,ratio,pos_err"<<endl;
	for(int pixel_noise=0; pixel_noise< 30; pixel_noise++){
		main_execute(pos_err, ratio, add_noise, pixel_noise,poseNums,tracing);
		cout<<pixel_noise<<","<<ratio<<","<<pos_err<<endl;
	}
}

void test_frame_num(){
	double pos_err;
	double ratio;
	bool add_noise = true;
	double pixel_noise = 5;
	bool tracing = false;
	cout<<"poseNums,ratio,pos_err"<<endl;
	for(int poseNums=5; poseNums< 30; poseNums++){
		main_execute(pos_err, ratio, add_noise, pixel_noise,poseNums,tracing);
		cout<<poseNums-2<<","<<ratio<<","<<pos_err<<endl;
	}
}
int main()
{
	double pos_err;
	double ratio;
	main_execute(pos_err, ratio);
	test_uv_nosie();
	test_frame_num();

    return 0;
}

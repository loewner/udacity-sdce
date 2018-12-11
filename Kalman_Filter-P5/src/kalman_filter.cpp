#include "kalman_filter.h"
#include <math.h>
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	//prepare
	MatrixXd K;
	K = MatrixXd(4, 4);

	// do update
	K = P_* H_.transpose() * ((H_ * P_ * H_.transpose() + R_).inverse());
	x_ = x_ + K * (z- H_*x_);
	P_ = P_ - ( K * H_ * P_ );

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

	// prepare measurement
	double rho = sqrt(pow(x_(0),2) + pow(x_(1),2));
	if (rho < 0.00001){
		rho = 0.00001;
	}
	double phi = atan2(x_(1),x_(0));
	double rhodot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
	VectorXd y;
	MatrixXd K;
	y = VectorXd(3);
	y << rho, phi, rhodot;
	y = z- y;
	double pi = 4*atan(1);

	VectorXd y_normalized;
	y_normalized = VectorXd(3);
	y_normalized << y(0), y(1) - 2*pi * round(y(1)/(2*pi)), y(2);

	// do update
	K = MatrixXd(4, 4);
	K = P_* H_.transpose() * ((H_ * P_ * H_.transpose() + R_).inverse());
	x_ = x_ + K * y_normalized;
	P_ = P_ - ( K * H_ * P_ );
}

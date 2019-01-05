#include "PID.h"
#include <iostream>
#include <math.h>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
	this->Kp = Kp;
	this->Ki = Ki;
	this->Kd = Kd;
	this->i_error = 0;
	this->total_squared_error = 0;
	this->cnt = 0;
}

void PID::UpdateError(double cte) {
	this->total_squared_error += pow(cte,2);
	if (this->firstTime == true){
		this->firstTime = false;
		this->d_error = 0;
	}
	else{
		this->d_error = cte - this->p_error;
	}
	this->i_error += cte;
	this->p_error = cte;
}

double PID::TotalError() {
	double totalErr = -this->Kp * this->p_error - this->Ki * this->i_error - this->Kd * this->d_error;
	return totalErr;
}


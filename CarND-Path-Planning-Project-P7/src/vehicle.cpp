
#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "../Eigen-3.3/Eigen/Core"
#include "../Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "road.h"
#include "vehicle.h"

/**
 * Initializes ego vehicle
 */
Vehicle::Vehicle(Road road){
	this->road = road;
	this->parameter_JMP.insert(make_pair(make_tuple(0,1), this->compute_paramteters_JMP({2,0,0},{6,0,0}, 50)));
	this->parameter_JMP.insert(make_pair(make_tuple(1,2), this->compute_paramteters_JMP({6,0,0},{9.8,0,0}, 50)));
	this->parameter_JMP.insert(make_pair(make_tuple(1,0), this->compute_paramteters_JMP({6,0,0},{2,0,0}, 50)));
	this->parameter_JMP.insert(make_pair(make_tuple(2,1), this->compute_paramteters_JMP({9.8,0,0},{6,0,0}, 50)));
}
/**
 * Initializes target vehicle
 */
Vehicle::Vehicle(int id, double x, double y, double vx, double vy, double s, double d)
{
	this->id = id;
	this->x = x;
	this->y = y;
	this->speed = sqrt(vx*vx + vy*vy);
	this->s = s;
	this->d = d;
}

Vehicle::~Vehicle() {}

tuple<vector<double>,vector<int>> Vehicle::closest_dist_in_lane(int lane){
	double dist_before = 1e20;
	double dist_behind = 1e20;
	int before_id = -1;
	int behind_id = -1;
	for (int i =0; i< this->other_veh.size(); i++){
		if(this->other_veh[i].get_lane() == lane ){
			double tempA = this->s - other_veh[i].s;
			double tempB = this->road.max_s - abs(tempA);
			if (tempA > 0){
				if(abs(tempA) > tempB){ // case  |--TARGET-------------------------EGO--|
					if(tempB < dist_before){
						dist_before = tempB;
						before_id = other_veh[i].id;
					}
				}
				else {
					if(abs(tempA) < dist_behind){ // case  |--------------------TARGET-------EGO--|
						dist_behind = abs(tempA);
						behind_id = other_veh[i].id;
					}
				}
			}
			else {
				if(abs(tempA) > tempB){
					if(tempB < dist_behind){ // case  |---EGO------------------------TARGET--|
						dist_behind = tempB;
						behind_id = other_veh[i].id;
					}
				}
				else{
					if(abs(tempA) < dist_before){ // case  |-----------------------EGO----TARGET--|
						dist_before = abs(tempA);
						before_id = other_veh[i].id;
					}
				}
			}
		}
	}
	vector<double> distances = {dist_before, dist_behind};
	vector<int> ids = {before_id, behind_id};
	return make_tuple(distances, ids);
}

void Vehicle::set_lane_speeds(){
	vector<int> ids;
	vector<double> distances;
	for( int l = 0; l< this->road.lanes.size(); l++){
		tuple<vector<double>,vector<int>> laneInfo = this->closest_dist_in_lane(l);
		ids = get<1>(laneInfo);
		distances = get<0>(laneInfo);
		if(distances[0] > 100){ // unbounded lane
			this->lane_speeds[l] = this->road.MAX_SPEED;
		}
		else{ // otherwise set lane speed to target veh speed
			this->lane_speeds[l] = this->other_veh[ids[0]].speed;
		}

	}
}


void Vehicle::set_target_speed(){

	this->set_begin_path_s();
	vector<double> distances = get<0>(this->closest_dist_in_lane(this->get_lane()));

	if(distances[0] < 45){
		this->target_speed -= 0.15;
	}
	if(distances[0] < 30){
		this->target_speed -= (30-distances[0])/60;
	}
	if(distances[0] < 20){
		this->target_speed -= (20-distances[0])/40;
	}
	if(distances[0] < 10){
			this->target_speed -= (10-distances[0])/25;
	}
	else {
		if(this->target_speed < 49){
			this->target_speed += abs(target_speed - 49) /60;
		}
	}

//	double curv = 0;
//
//	for (int i = 0; i < 5; i++){
//		double current_s = this->begin_path_s + double(i) /2.5 * double(target_speed)/45;
//		curv += this->curvature(current_s);
//	}
//	if (abs(curv/5) > 0.001 & target_speed > 45){
//		this->target_speed = 44;
//	}



}

double Vehicle::curvature(double s){
	double norm = sqrt(pow(this->road.sx.deriv(1,s),2) + pow(this->road.sy.deriv(1,s),2));

	double curv1 = road.sdx.deriv(1,s) / (this->road.sx.deriv(1,s) /norm);
	double curv2 = road.sdy.deriv(1,s) / (this->road.sy.deriv(1,s) /norm);

	return (curv1 + curv2)/2;
}

void Vehicle::follow(){

	vector<double> s;
	vector<double> d;
	double current_s;

	for (int i = 0; i < 100; i++){
		current_s = this->begin_path_s + double(i) /2.5 * double(target_speed)/45;
		s.push_back(current_s);
		double lane;
		if (2+this->get_lane()*4 <10){
			lane = 2+this->get_lane()*4;
		}
		else{
			lane = 9.8;
		}
		d.push_back(lane);
	}
	this->drive2(s, d);
}

void Vehicle::change_lane(string next_state){
	// initialize lane change
	if ( this->during_lane_change == false){
		this->during_lane_change = true;
		this->lange_change_start_s = this->begin_path_s;
		this->lane_to_change = this->get_lane();
		if (next_state == "LC"){
			this->lange_change_state = "LC";
			this->lange_change_orientation = -1;
		}
		else {
			this->lange_change_state = "RC";
			this->lange_change_orientation = 1;
		}
	}
	vector<double> s;
	vector<double> d;
	for (int i = 0; i < 100; i++){
		double current_s = this->begin_path_s + i /2.5 * double(this->target_speed)/45;
		if ((current_s > this->lange_change_start_s) & (current_s <= this->lange_change_start_s + this->lange_change_length)){
			d.push_back(this->eval_JMP(
					current_s - this->lange_change_start_s,
					this->parameter_JMP[make_tuple(this->lane_to_change, this->lane_to_change+this->lange_change_orientation)]));
		}
		if (current_s > this->lange_change_start_s + this->lange_change_length){
			if (2 + 4* (this->lane_to_change + this->lange_change_orientation) < 10){
				d.push_back( 2 + 4* (this->lane_to_change + this->lange_change_orientation));
			}
			else {
				d.push_back(9.8);
			}

		}
		s.push_back(current_s);
	}
	this->drive2(s, d);
}

vector<string> Vehicle::get_successor_states(){
	vector<string> successor_states;
	for (int j = 0; j< this->states.size(); j++){
		int lane = this->get_lane();
		if( states[j] == "K"){
			successor_states.push_back(states[j]);
		}
		if( (states[j] == "RC") & (lane <= 1) ){
			tuple<vector<double>,vector<int>> laneInfo = this->closest_dist_in_lane(lane+1);
			vector<double> distances = get<0>(laneInfo);
			if ((distances[0] > 30) & (distances[1] > 20)){
				successor_states.push_back(states[j]);
			}

		}
		if( (states[j] == "LC") & (lane >=1) ){
			tuple<vector<double>,vector<int>> laneInfo = this->closest_dist_in_lane(lane-1);
			vector<double> distances = get<0>(laneInfo);
			if ((distances[0] > 30) & (distances[1] > 20)){
				successor_states.push_back(states[j]);
			}
		}
	}

	return successor_states;
}


void Vehicle::set_target_state(vector<string> states){

	// set current lane speeds
	this->set_lane_speeds();

	// compute cost functions
	vector<double> costs_ineff = this->inefficiency_cost();
	vector<double> costs_safe = this->safety_cost();
	vector<double> costs;

	// calc total costs
	for (int j = 0; j< costs_ineff.size(); j++){
		costs.push_back(costs_ineff[j] + costs_safe[j]);
	}

	// calc the costs for each state
	vector<double> costs_at_states;
	for (int j = 0; j<states.size(); j++){
		if (states[j] == "K"){
			costs_at_states.push_back(costs[this->get_lane()]);
		}
		if (states[j] == "LC"){
			costs_at_states.push_back(costs[this->get_lane() - 1]);
		}
		if (states[j] == "RC"){
			costs_at_states.push_back(costs[this->get_lane() + 1]);
		}
	}
//	cout << "\n===\n";
//	for (int j = 0; j< costs_at_states.size(); j++){
//		cout << " - " << costs_at_states[j];
//	}
//	cout << "\n===\n";

	// find state with lowest cost
	int minimum_cost_state = -1;
	double minimum_cost = 2;
	for (int l = 0; l< states.size(); l++){
		if (costs_at_states[l]<minimum_cost){
			minimum_cost = costs_at_states[l];
			minimum_cost_state = l;
		}
	}

	// set target state
	this->target_state = states[minimum_cost_state];
//	cout << "minimum cost state: " << this->target_state << "\n";
}


void Vehicle::start_piloted_driving(){

	// set current s
	this->set_begin_path_s();

	// get all reachable next states
	vector<string> successor_states = this->get_successor_states();
//	cout << "\n";
//	for ( int j = 0; j< successor_states.size(); j++ ){
//		cout << " - " << successor_states[j];
//	}
//	cout << "\n-----------------------\n";

	// choose best of these states by global minimum cost
	this->set_target_state(successor_states);

	// set next state
	string next_state = this->target_state;
//	cout << "during lc: " << this->during_lane_change << "\n";
//	cout << "lc start: " << this->lange_change_start_s << "\n";
//	cout << "begin path s: " << this->begin_path_s << "\n";
	if (this->during_lane_change == true){
		if (this->begin_path_s - this->lange_change_start_s > this->lange_change_length){
			this->during_lane_change = false;
			next_state = "K";
		}
		else{
			next_state = this->lange_change_state;
		}
	}
//	cout << next_state << "\n successor states:";

	// set target speed
	this->set_target_speed();

	if (next_state == "K"){
		this->follow();
	}
	if ( (next_state == "LC") | (next_state == "RC") ){
		this->change_lane(next_state);
	}

}

void Vehicle::set_begin_path_s(){
	// calc current_s based on previous_path (this->s does not fit to good)
	vector<double> temp_s = {this->s - 3 ,this->s +3};
	vector<double> temp_d = {this->d, this->d};

	double lambda = 0.5;
	if(previous_path.size() > 0){
		vector<waypoint> temp_res = this->road.getXY_spline(temp_s, temp_d);
		if (abs(temp_res[1].x - temp_res[0].x) > abs(temp_res[1].y - temp_res[0].y)){
			lambda = (this->previous_path[0].x - temp_res[0].x)/(temp_res[1].x - temp_res[0].x);
		}
		else{
			lambda = (this->previous_path[0].y - temp_res[0].y)/(temp_res[1].y - temp_res[0].y);
		}
	}
	this->begin_path_s = this->s -3 + lambda * 6;
}


void Vehicle::drive2(vector<double> s, vector<double> d){

	// do homotopy between new and previous path
	vector<waypoint> result;
	result = this->road.getXY_spline(s, d);

	int N = this->previous_path.size();
	double control;

	for (int i = 0; i < N; i++){
		result[i].x = this->previous_path[i].x * (1- double(i)/(N)) + result[i].x * double(i)/(N);
		result[i].y = this->previous_path[i].y * (1- double(i)/(N)) + result[i].y * double(i)/(N);
		if (i > 0){
			control = this->road.distance(result[i].x,result[i].y,result[i-1].x,result[i-1].y);
			if (control > 47/45/2.5){
				result[i].x = result[i-1].x + (result[i].x - result[i-1].x) * 47/45/2.5 / control;
				result[i].y = result[i-1].y + (result[i].y - result[i-1].y) * 47/45/2.5 / control;
			}
		}
	}

	// return result
	this->current_path = result;
}


vector<double> Vehicle::compute_paramteters_JMP(vector <double> start, vector <double> end, double T)
{
    /*
    Calculate the Jerk Minimizing Trajectory that connects the initial state
    to the final state in time T.

    INPUTS

    start - the vehicles start location given as a length three array
        corresponding to initial values of [s, s_dot, s_double_dot]

    end   - the desired end state for vehicle. Like "start" this is a
        length three array.

    T     - The duration, in seconds, over which this maneuver should occur.

    OUTPUT
    an array of length 6, each value corresponding to a coefficent in the polynomial
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

    EXAMPLE

    > JMT( [0, 10, 0], [10, 10, 0], 1)
    [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    */
    Eigen::MatrixXd matrix(3,3);
    matrix << pow(T,3), pow(T,4), pow(T,5), 3*pow(T,2), 4*pow(T,3), 5*pow(T,4), 6*T, 12*pow(T,2), 20*pow(T,3);

    Eigen::VectorXd vec(3);
    vec << end[0]-(start[0]+start[1]*T + start[2]*pow(T,2)/2), end[1]-(start[1]+start[2]*T), end[2]-start[2];

    Eigen::VectorXd x(3);
    x = matrix.colPivHouseholderQr().solve(vec);

    return {start[0],start[1],start[2]/2, x[0],x[1],x[2]};
}

double Vehicle::eval_JMP( double x, vector<double> params){
	return params[0] + params[1]*x + params[2]*pow(x,2) + params[3]*pow(x,3) + params[4]*pow(x,4) + params[5]*pow(x,5);
}


int Vehicle::get_lane(){
	return floor(this->d / 4);
}


vector<double> Vehicle::inefficiency_cost(){
	vector<double> costs = {2,2,2};
	for (int l = 0; l< this->road.lanes.size(); l++){
		if (abs(this->get_lane() -l)<=1){
			costs[l] = exp( - this->lane_speeds[l] /this->road.MAX_SPEED );
		}
	}
	return costs;
}

vector<double> Vehicle::safety_cost(){
	vector<double> costs = {0,0,0};
	for (int l = 0; l< this->road.lanes.size(); l++){
		if (this->get_lane() != l){
			costs[l] = 0.001;
		}
	}
	return costs;
}


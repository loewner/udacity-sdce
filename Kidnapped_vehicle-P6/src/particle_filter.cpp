/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 500;
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// Set standard deviations for x, y, and theta
	 std_x = std[0];
	 std_y = std[1];
	 std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// iterate over all particles
	for (int i = 0; i < num_particles; ++i) {
		// Sample from these normal distrubtions like this:
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1;

		// append particle to particles vector
		particles.push_back(particle);

	}
	is_initialized = true;
	std::cout << "initialized all particles";

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// iterate over all particles

	// Set standard deviations for x, y, and theta
	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// This line creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	// iterate over all particles
	for (int i = 0; i < num_particles; ++i) {
		//get address of i-th particle
		Particle* p = &(particles[i]);
		if (yaw_rate < 1E-15){
			// use rule of L'Hospital (when yaw_rate tends to 0)
			p->x += velocity * cos(p->theta) * delta_t + dist_x(gen);
			p->y +=	velocity * sin(p->theta) * delta_t + dist_y(gen);
		}
		else{
			p->x += velocity/yaw_rate * (sin(p->theta+yaw_rate*delta_t)-sin(p->theta)) + dist_x(gen);
			p->y += velocity/yaw_rate * (cos(p->theta)-cos(p->theta+yaw_rate*delta_t)) + dist_y(gen);
		}
		p->theta += yaw_rate * delta_t + dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int num_obs = observations.size();
	int num_landmarks = predicted.size();
	// iterate over observed measurement
	for (int i = 0; i < num_obs; i++){
		double temp_err = std::numeric_limits<double>::max();
		double temp_x;
		double temp_y;
		double temp_id;
		// iterate over static landmarks
		for (int j = 0; j < num_landmarks; j++){
			double dx2 = (observations[i].x - predicted[j].x)*(observations[i].x - predicted[j].x);
			double dy2 = (observations[i].y - predicted[j].y)*(observations[i].y - predicted[j].y);
			if(dx2 + dy2 < temp_err){
				temp_x = predicted[j].x;
				temp_y = predicted[j].y;
				temp_id = predicted[j].id;
				temp_err = dx2+dy2;
			}
		}
		// set minimum distance landmark
		observations[i].x = temp_x;
		observations[i].y = temp_y;
		observations[i].id = temp_id;
	}

}

void ParticleFilter::normalizeWeights(){
	// normalize particle weights such that they add up to 1
	double sum = 0;
	for (int i=0; i < num_particles; i++){
		sum += particles[i].weight;
	}
	for (int i=0; i < num_particles; i++){
		if (sum < 1E-30) {
			// re-normalize weights if sum is to small
			particles[i].weight = 1 / num_particles;
		}
		else {
			particles[i].weight /= sum;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Set standard deviations for x, y, and theta
	default_random_engine gen;
	double std_x, std_y; // Standard deviations for x and y
	std_x = std_landmark[0];
	std_y = std_landmark[1];

	// transform landmarks with struct single_landmark_s to LandmarkObs (float -> double)
	std::vector<LandmarkObs> predicted;
	for (int i = 0; i < map_landmarks.landmark_list.size(); i++){
		LandmarkObs l;
		l.id = map_landmarks.landmark_list[i].id_i;
		l.x = (double) map_landmarks.landmark_list[i].x_f;
		l.y = (double) map_landmarks.landmark_list[i].y_f;
		predicted.push_back(l);
	}



	// find relevant observations
	std::vector<LandmarkObs> relevant_observations;
	for (int j =0; j < observations.size(); j++){
		if (observations[j].x * observations[j].x + observations[j].y * observations[j].y < sensor_range * sensor_range ){
			relevant_observations.push_back(observations[j]);
		}
	}
	// number of relevant observations
	int num_obs = observations.size();



	// iterate over all particles
	for (int i = 0; i < num_particles; ++i) {
		//get address of i-th particle
		Particle* p = &(particles[i]);

		//initialize predicted vector
		std::vector<LandmarkObs> observed;

		// iterate over observations
		for (int j=0; j < num_obs; j++){
			// transform observation to map's coordinate system
			LandmarkObs tranf_obs;
			tranf_obs.id = relevant_observations[j].id;
			tranf_obs.x =  relevant_observations[j].x * cos(p->theta) - relevant_observations[j].y * sin(p->theta) + p->x;
			tranf_obs.y =  relevant_observations[j].y * cos(p->theta) + relevant_observations[j].x * sin(p->theta) + p->y;
			observed.push_back(tranf_obs);
		}

		// Find the predicted measurement that is closest to each observed measurement
		std::vector<LandmarkObs> pred_observed = observed;
		dataAssociation(predicted, pred_observed);

		// iterate over observations
		double prob = 1;
		for (int j=0; j < num_obs; j++){
			// compute pdf's
			double probx = normal_pdf(pred_observed[j].x, observed[j].x, std_x);
			double proby = normal_pdf(pred_observed[j].y, observed[j].y, std_y);
			prob *= probx*proby;
		}
		p->weight=prob;
	}
	// normalize particle weights
	normalizeWeights();
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// get random index in 0 to N-1

	// write weights to vector
	vector<double> weights;
	weights.clear();
	vector<Particle> new_Particles;
	for (int i=0; i < num_particles; i++){
			weights.push_back(particles[i].weight);
	}
	default_random_engine gen;
	discrete_distribution<> dist(weights.begin(), weights.end());

	for (int i=0; i < num_particles; i++){
		int index = dist(gen);
		new_Particles.push_back(particles[index]);
	}

	// set new particles
	particles = new_Particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

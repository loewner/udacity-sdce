# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---
This project implements a basic PID controller for driving a vehicle around a track based on the following simulator:
https://github.com/udacity/self-driving-car-sim/releases

Moreover, to find optimal parameters for the pid control we implemented a coordinate ascent method.

## Getting started
### how to build and start the PID controller

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake ../src && make`
4. Run it: `./pid(tau_p, tau_i, tau_d, N)`
    * tau_p: p control parameter
    * tau_i: i control parameter
    * tau_d: d control parameter
    * N: Number of timesteps to go

    Example: `./pid(0.05, 0, 1.8, 1000)`

### how to start the coordinate ascent method (Linux Plattform only)

This is implemented in the following Jupyer notebook: `coordinateAscent.ipynb`

1. start the simulator: "./term2_sim.x86_64" (maybe you need `chmod a+x` before)
2. run the Jupyter notebook: `coordinateAscent.ipynb`

As a result, this gives you good parameters tau_p, tau_i and tau_d

Please note that this script uses `xdotool` (takes control over your mouse) to give commands to the simulator. Therefore, we recommend to run this in an encapsulated environment.


## model description and implementation

### pid controller
the pid controller is implemented in the class `PID`:

* p stands for the proportional component: It helps to find the center of the lane.
* i for the integrated component: This value is useful to balance a bias.
* d for the derivative component: This value is useful to reduce overshooting

Each cross track error (cte) leads to an update of p, i and d values. This is done by the method `Update_error`.

the control value is than given as the sum of these values. This is done by the method `TotalError`.

In theory, the usage of pid controllers is very simple. The challenge due to find good parameters (p, i, d).

In order to do so, we used coordinate ascent:

### coordinate ascent

This is implemented in the Jupyter notebook `coordinateAscent.ipynb`. Note that the notebook can only be used in combination with linux platforms. The reason for this is, that we used `xdotool`, a simple unix tool for giving click and key commands. Unfortunately, the simulator can not be uses from command line, so xdotool is used to start and stop the simulator by using the gui. 
There are "click"-tools like `xdotool` also available for Mac and Windows as well, so it should not be hard to carry this solution over to other platforms.

One disadvantage of coordinate ascent is, that is needs to perform the simulation quite often, so using the script might take several hours.
Moreover coordinate ascent might get stuck to local minima.

In order to circumvent the first problem, we used weights in order to update the coefficients p, i and d.
Therefore, we use a weight of 1/10¹ for p, 1/10³ for i and 1 for d. 
This means, p is first updated by +- 1/10¹, while i is updated first by +- 1/10³
The reason for this, is that the coefficients are not comparable, so adding a constant value to i will have much more impact than adding the same constant value to d. They weights will somehow balance this a little bit.

### simulation results

In our simulation we used a constant throttle value of 0.3.
This allows the car to go around 35 mph in max.

We used the twiddle method win coordinateAscent.ipynb with the following standard parameter:
```
twiddle(start_p = [0.1, 0.001, 1.0], 
            start_dp = [1,1,1],
            standardization=[10,1000,1], 
            tol=0.2): 
```

this leads us to the following parameters:

* p=
* i=
* d=

TODO: VIDEO


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.




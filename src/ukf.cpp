#define DEBUG false
#define NIS false

#include "ukf.h"
#include "Eigen/Dense"
#include "iostream"
#include "measurement_package.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;

    // State dimension
    n_x_ = 5;

    // Augmented state dimension
    n_aug_ = n_x_ + 2;

    // radar measurement dimension
    n_rdr_ = 3;

    // radar measurement dimension
    n_ldr_ = 2;

    // Number of sigma points
    n_sigma_ = 2 * n_aug_ + 1;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);

    // Weights of sigma points
    weights_ = VectorXd(n_sigma_);

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.8;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 2.8;

    /**
     * DO NOT MODIFY measurement noise values below.
     * These are provided by the sensor manufacturer.
     */

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
     * End DO NOT MODIFY section for measurement noise values
     */

    /**
     * TODO: Complete the initialization. See ukf.h for other member properties.
     * Hint: one or more values initialized above might be wildly off...
     */

    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;

    // Weights of sigma points
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < weights_.size(); i++) weights_(i) = 0.5 / (lambda_ + n_aug_);
}

UKF::~UKF() {}

void UKF::printNIS() {
    std::cout << "\nLaser NIS values \n";
    for (auto &nis : nis_laser) std::cout << nis << ",";

    std::cout << "\nRadar NIS values \n";
    for (auto &nis : nis_radar) std::cout << nis << ",";
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Make sure you switch between lidar and radar
     * measurements.
     */

    // Forum answer https://knowledge.udacity.com/questions/759764 help me correct my understanding of where
    // to initialize x_ and P_. Previously, I initialized both the state estimate and the state covariance P_ in
    // the constructor and not here in the ProcessMeasurement function. I combined lidar and radar standard deviations
    // when initializing _P:
    //             P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
    //                    0, std_laspy_ * std_laspx_, 0, 0, 0,
    //                    0, 0, std_radrd_ * std_radrd_, 0, 0,
    //                    0, 0, 0, std_radphi_ * std_radphi_, 0,
    //                    0, 0, 0, 0, std_radphi_ * std_radphi_;

    if (!is_initialized_) {
        std::cout << "Kalman Filter Initialization " << std::endl;

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
            P_ << std_laspx_ * std_laspx_, 0, 0, 0, 0,
                    0, std_laspy_ * std_laspy_, 0, 0, 0,
                    0, 0, 1, 0, 0,
                    0, 0, 0, 1, 0,
                    0, 0, 0, 0, 1;
        } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

            const double rho = meas_package.raw_measurements_(0);
            const double phi = meas_package.raw_measurements_(1);
            const double rho_dot = meas_package.raw_measurements_(2);
            const double x = rho * cos(phi);
            const double y = rho * sin(phi);
            const double vx = rho_dot * cos(phi);
            const double vy = rho_dot * sin(phi);
            const double v = sqrt(vx * vx + vy * vy);
            x_ << x, y, v, rho, rho_dot;
            P_ << std_radr_ * std_radr_, 0, 0, 0, 0,
                    0, std_radr_ * std_radr_, 0, 0, 0,
                    0, 0, std_radrd_ * std_radrd_, 0, 0,
                    0, 0, 0, std_radphi_ * std_radphi_, 0,
                    0, 0, 0, 0, std_radphi_ * std_radphi_;
        } else {
            std::cout << "NO MEASUREMENT RECORDED" << std::endl;
        }
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    else {
        // calculate the time between the previous and current measurements
        double dt = (double) (meas_package.timestamp_ - time_us_) / 1000000.0;

        // input current time as the previous time
        time_us_ = meas_package.timestamp_;

        // predict
        Prediction(dt);

        // measurement update step
        if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) UpdateLidar(meas_package);
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) UpdateRadar(meas_package);
    }
}

void UKF::Prediction(const double delta_t) {
    /**
     * TODO: Complete this function! Estimate the object's location.
     * Modify the state vector, x_. Predict sigma points, the state,
     * and the state covariance matrix.
     */

    // 1. Generate Sigma Points

    // create the augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug << x_, 0, 0;

    // create augmented state covariance
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug.bottomRightCorner(2, 2) << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

    // create square root P_aug matrix AND augmented sigma points matrix
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sigma_);
    MatrixXd A = (3 * P_aug).llt().matrixL();
    Xsig_aug.colwise() += x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) += A.col(i);
        Xsig_aug.col(i + 1 + n_aug_) -= A.col(i);
    }

    if (DEBUG) {
        std::cout << "Xsig_aug = " << Xsig_aug << std::endl;
    }

    // 2. Predict Sigma Points
    Xsig_pred_ = Xsig_aug.topRows(n_x_);

    for (int idx = 0; idx < n_sigma_; idx++) {
        double v        = Xsig_aug(2, idx);
        double psi      = Xsig_aug(3, idx);
        double psi_d    = Xsig_aug(4, idx);
        double nu_a     = Xsig_aug(5, idx);
        double nu_yawdd = Xsig_aug(6, idx);

        if (fabs(psi_d) > 0.001) {
            Xsig_pred_(0, idx) += (v / psi_d) * (sin(psi + delta_t * psi_d) - sin(psi));
            Xsig_pred_(1, idx) += (v / psi_d) * (cos(psi) - cos(psi + delta_t * psi_d));
            Xsig_pred_(3, idx) += delta_t * psi_d;
        } else {
            Xsig_pred_(0, idx) += v * delta_t * cos(psi);
            Xsig_pred_(1, idx) += v * delta_t * sin(psi);
        }

        // add noise
        Xsig_pred_(0, idx) += 0.5 * delta_t * delta_t * cos(psi) * nu_a;
        Xsig_pred_(1, idx) += 0.5 * delta_t * delta_t * sin(psi) * nu_a;
        Xsig_pred_(2, idx) += delta_t * nu_a;
        Xsig_pred_(3, idx) += 0.5 * delta_t * delta_t * nu_yawdd;
        Xsig_pred_(4, idx) += delta_t * nu_yawdd;
    }

    // 3. Predict State Mean & State Covariance Matrix
    x_ = Xsig_pred_ * weights_;
    P_.setZero();
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
        P_ += weights_(i) * x_diff * x_diff.transpose();
    }

    if (DEBUG) {
        std::cout << "Predicted state" << std::endl;
        std::cout << x_ << std::endl;
        std::cout << "Predicted covariance matrix: " << std::endl;
        std::cout << P_ << std::endl;
    }
}

void UKF::UpdateStateAndCovariance(MatrixXd &S, MatrixXd &Z, const MeasurementPackage &meas_package) {

    // mean predicted measurement
    VectorXd z_pred = Z * weights_;

    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, S.cols());

    // predict state covariance matrix & calculate cross correlation matrix
    Z.colwise() -= z_pred;
    for (int i = 0; i < n_sigma_; i++) {
        S  += weights_(i) * Z.col(i) * Z.col(i).transpose();
        Tc += weights_(i) * (Xsig_pred_.col(i) - x_) * Z.col(i).transpose();
    }

    // compute Kalman gain
    MatrixXd K = Tc * S.inverse();

    // update state mean and covariance matrix
    auto err = meas_package.raw_measurements_ - z_pred;
    x_ += K * err;
    P_ -= K * S * K.transpose();

    // Compute NIS (Mahalanobis distance).  NIS ~ χ2 ; threshold is p = 0.95.
    if (NIS) {
        double nis = err.transpose() * S.inverse() * err;
        if (meas_package.sensor_type_ == MeasurementPackage::LASER) nis_laser.push_back(nis);
        else nis_radar.push_back(nis);
    }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use lidar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the lidar NIS, if desired.
     */

    // Predict Measurement

    //  apply 'h' measurement function to sigma points; the first two rows have position [px, py]
    MatrixXd Zsig = Xsig_pred_.topRows(n_ldr_);

    // create measurement covariance matrix S; then initialize with noise covariance. R is a diagonal matrix.
    MatrixXd S = MatrixXd(n_ldr_, n_ldr_);
    S << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

    // Update state mean and covariance matrix
    UpdateStateAndCovariance(S, Zsig, meas_package);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
     * TODO: Complete this function! Use radar data to update the belief
     * about the object's position. Modify the state vector, x_, and
     * covariance, P_.
     * You can also calculate the radar NIS, if desired.
     */

    // Predict Measurement

    // create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_rdr_, n_sigma_);

    // create empty measurement covariance matrix S, then initialize it with noise covariance R. R is a diagonal matrix.
    MatrixXd S = MatrixXd(n_rdr_, n_rdr_);
    S << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_ * std_radrd_;

    // fill Zsig and predicted
    for (int j = 0; j < weights_.size(); j++) {
        double px = Xsig_pred_(0, j);
        double py = Xsig_pred_(1, j);
        double v = Xsig_pred_(2, j);
        double psi = Xsig_pred_(3, j);

        // apply 'h' measurement function
        Zsig(0, j) = sqrt(px * px + py * py);
        Zsig(1, j) = atan2(py, px);
        Zsig(2, j) = v * (px * cos(psi) + py * sin(psi)) / Zsig(0, j);
    }

    // Update state mean and covariance matrix
    UpdateStateAndCovariance( S, Zsig, meas_package);
}

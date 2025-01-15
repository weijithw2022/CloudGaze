import numpy as np

class KalmanFilter:
    """
    Kalman Filter for tracking position and velocity in a 2D plane.
    
    Attributes:
        - dt: Time step between state updates
        - process_noise (Q): Process noise covariance matrix
        - measurement_noise (R): Measurement noise covariance matrix
        - initial_state (x): Initial state vector [position_x, position_y, velocity_x, velocity_y]
        - initial_covariance (P): Initial estimation error covariance matrix

    Methods:
        - predict(): Predicts the next state and updates covariance based on system dynamics.
        - update(z): Updates the state and covariance with a new measurement z.
        - get_state(): Returns the current state estimate.
    """

    def __init__(self, dt, process_noise, measurement_noise, initial_state, initial_covariance):
        """
        Initialize the Kalman Filter.

        Parameters:
        - dt (float): Time step between updates
        - process_noise (np.ndarray): Process noise covariance matrix (Q)
        - measurement_noise (np.ndarray): Measurement noise covariance matrix (R)
        - initial_state (np.ndarray): Initial state vector [pos_x, pos_y, vel_x, vel_y]
        - initial_covariance (np.ndarray): Initial state covariance matrix (P)
        """
        self.dt = dt
        
        # State transition matrix A (models how state evolves over time)
        # [ pos_x_new ]   = [ 1 0 dt 0 ] [ pos_x ]
        # [ pos_y_new ]     [ 0 1 0 dt ] [ pos_y ]
        # [ vel_x_new ]     [ 0 0 1  0 ] [ vel_x ]
        # [ vel_y_new ]     [ 0 0 0  1 ] [ vel_y ]
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix H (maps state to observed measurements)
        # Only position measurements are available:
        # [ pos_x ]
        # [ pos_y ]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance matrix (Q)
        self.Q = process_noise
        
        # Measurement noise covariance matrix (R)
        self.R = measurement_noise
        
        # Initial state vector (x)
        self.x = initial_state
        
        # Initial state covariance matrix (P)
        self.P = initial_covariance

    def predict(self):
        """
        Predict the next state and update the state covariance matrix.
        State prediction: Xk+1|k = A * Xk|k
        Covariance prediction: Pk+1|k = A * Pk * A^T + Q
        """
        # State prediction
        self.x = self.A @ self.x
        
        # Covariance prediction
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        """
        Update the state estimate and covariance with the measurement z.
        
        Parameters:
        - z (np.ndarray): New measurement [pos_x, pos_y]
        """
        # Kalman Gain calculation
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        
        # State update: Xk+1|k+1 = Xk+1|k + K * (z - H * Xk+1|k)
        self.x = self.x + K @ (z - self.H @ self.x)
        
        # Covariance update: Pk+1|k+1 = (I - K * H) * Pk+1|k
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P

    def get_state(self):
        """
        Return the current state estimate.
        
        Returns:
        - np.ndarray: Current state vector [pos_x, pos_y, vel_x, vel_y]
        """
        return self.x

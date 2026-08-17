# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ros2 python
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from lifecycle_msgs.srv import GetState

# libraries
import numpy as np
import range_libc
import time
from threading import Lock
from particle_filter import utils as Utils

# TF
# import tf.transformations
# import tf
from tf2_ros import TransformBroadcaster
import tf_transformations

# messages
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap

'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFiler(Node):
    '''
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    '''

    def __init__(self):
        super().__init__('particle_filter')

        # declare parameters
        self.declare_parameter('angle_step', 18)
        self.declare_parameter('max_particles', 4000)
        self.declare_parameter('max_viz_particles', 60)
        self.declare_parameter('squash_factor', 2.2)
        self.declare_parameter('max_range', 10)
        self.declare_parameter('theta_discretization', 112)
        self.declare_parameter('range_method', 'rmgpu')
        self.declare_parameter('rangelib_variant', 2)
        self.declare_parameter('fine_timing', 0)
        self.declare_parameter('publish_odom', 1)
        self.declare_parameter('viz', 1)
        self.declare_parameter('z_short', 0.01)
        self.declare_parameter('z_max', 0.07)
        self.declare_parameter('z_rand', 0.12)
        self.declare_parameter('z_hit', 0.75)
        self.declare_parameter('sigma_hit', 8.0)
        self.declare_parameter('motion_dispersion_x', 0.05)
        self.declare_parameter('motion_dispersion_y', 0.025)
        self.declare_parameter('motion_dispersion_theta', 0.25)
        self.declare_parameter('motion_model', 'arc')
        self.declare_parameter('motion_dispersion_arc_x', 0.05)
        self.declare_parameter('motion_dispersion_arc_y', 0.025)
        self.declare_parameter('motion_dispersion_arc_theta', 0.25)
        self.declare_parameter('motion_dispersion_arc_xy', 0.0)
        self.declare_parameter('motion_dispersion_arc_x_min', 0.01)
        self.declare_parameter('motion_dispersion_arc_y_min', 0.01)
        self.declare_parameter('motion_dispersion_arc_y_max', 0.01)
        self.declare_parameter('motion_dispersion_arc_theta_min', 0.01)
        self.declare_parameter('motion_dispersion_arc_xy_min_x', 0.01)
        self.declare_parameter('enable_recovery', True)
        self.declare_parameter('recovery_alpha_slow', 0.001)
        self.declare_parameter('recovery_alpha_fast', 0.1)
        self.declare_parameter('recovery_max_fraction', 0.05)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odometry_topic', '/early_fusion/odom')
        self.declare_parameter('ess_threshold_ratio', 0.5)
        self.declare_parameter('out_of_map_penalty', 0.01)
        self.declare_parameter('global_frame_id', 'map')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('laser_frame_id', 'laser')
        self.declare_parameter('laser_offset_x', 0.165)
        self.declare_parameter('laser_offset_y', 0.0)
        self.declare_parameter('pose_estimation_method', 'mean')

        # parameters
        self.ANGLE_STEP           = self.get_parameter('angle_step').value
        self.MAX_PARTICLES        = self.get_parameter('max_particles').value
        self.MAX_VIZ_PARTICLES    = self.get_parameter('max_viz_particles').value
        self.INV_SQUASH_FACTOR    = 1.0 / self.get_parameter('squash_factor').value
        self.MAX_RANGE_METERS     = self.get_parameter('max_range').value
        self.THETA_DISCRETIZATION = self.get_parameter('theta_discretization').value
        self.WHICH_RM             = self.get_parameter('range_method').value
        self.RANGELIB_VAR         = self.get_parameter('rangelib_variant').value
        self.SHOW_FINE_TIMING     = self.get_parameter('fine_timing').value
        self.PUBLISH_ODOM         = self.get_parameter('publish_odom').value
        self.DO_VIZ               = self.get_parameter('viz').value
        self.POSE_ESTIMATION_METHOD = self.get_parameter('pose_estimation_method').value

        # sensor model constants
        self.Z_SHORT   = self.get_parameter('z_short').value
        self.Z_MAX     = self.get_parameter('z_max').value
        self.Z_RAND    = self.get_parameter('z_rand').value
        self.Z_HIT     = self.get_parameter('z_hit').value
        self.SIGMA_HIT = self.get_parameter('sigma_hit').value

        # motion model constants
        self.MOTION_MODEL            = self.get_parameter('motion_model').value
        self.MOTION_DISPERSION_X     = self.get_parameter('motion_dispersion_x').value
        self.MOTION_DISPERSION_Y     = self.get_parameter('motion_dispersion_y').value
        self.MOTION_DISPERSION_THETA = self.get_parameter('motion_dispersion_theta').value
        self.MOTION_DISPERSION_ARC_X = self.get_parameter('motion_dispersion_arc_x').value
        self.MOTION_DISPERSION_ARC_Y = self.get_parameter('motion_dispersion_arc_y').value
        self.MOTION_DISPERSION_ARC_THETA = self.get_parameter('motion_dispersion_arc_theta').value
        self.MOTION_DISPERSION_ARC_XY = self.get_parameter('motion_dispersion_arc_xy').value
        self.MOTION_DISPERSION_ARC_X_MIN = self.get_parameter('motion_dispersion_arc_x_min').value
        self.MOTION_DISPERSION_ARC_Y_MIN = self.get_parameter('motion_dispersion_arc_y_min').value
        self.MOTION_DISPERSION_ARC_Y_MAX = self.get_parameter('motion_dispersion_arc_y_max').value
        self.MOTION_DISPERSION_ARC_THETA_MIN = self.get_parameter('motion_dispersion_arc_theta_min').value
        self.MOTION_DISPERSION_ARC_XY_MIN_X = self.get_parameter('motion_dispersion_arc_xy_min_x').value

        # recovery system constants
        self.ENABLE_RECOVERY = self.get_parameter('enable_recovery').value
        self.RECOVERY_ALPHA_SLOW = self.get_parameter('recovery_alpha_slow').value
        self.RECOVERY_ALPHA_FAST = self.get_parameter('recovery_alpha_fast').value
        self.RECOVERY_MAX_FRACTION = self.get_parameter('recovery_max_fraction').value
        self.ESS_THRESHOLD_RATIO = self.get_parameter('ess_threshold_ratio').value
        self.OUT_OF_MAP_PENALTY = self.get_parameter('out_of_map_penalty').value
        self.GLOBAL_FRAME_ID = self.get_parameter('global_frame_id').value
        self.ODOM_FRAME_ID = self.get_parameter('odom_frame_id').value
        self.BASE_FRAME_ID = self.get_parameter('base_frame_id').value
        self.LASER_FRAME_ID = self.get_parameter('laser_frame_id').value
        self.laser_offset_x = self.get_parameter('laser_offset_x').value
        self.laser_offset_y = self.get_parameter('laser_offset_y').value
        self.w_slow = 0.0
        self.w_fast = 0.0
        
        # various data containers used in the MCL algorithm
        self.MAX_RANGE_PX = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_stamp = None
        self.first_sensor_update = True
        self.state_lock = Lock()
        self.new_lidar_data = False

        # cache working buffers to avoid allocation in critical loop
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float64)
        self.proposal_distribution = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float64)
        self.queries_pose = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)
        self.motion_dx = np.zeros(self.MAX_PARTICLES, dtype=np.float64)
        self.motion_dy = np.zeros(self.MAX_PARTICLES, dtype=np.float64)
        self.motion_dtheta = np.zeros(self.MAX_PARTICLES, dtype=np.float64)

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float64)
        self.weights = np.ones(self.MAX_PARTICLES, dtype=np.float64) / float(self.MAX_PARTICLES)

        # initialize the state
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)
        self.pose_reset = False
        # map service client
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_omap()
        self.precompute_sensor_model()
        self.initialize_global()

        # parameter callback for dynamic tuning
        self.add_on_set_parameters_callback(self.parameters_callback)

        # keep track of speed from input odom
        self.current_speed = 0.0

        # Pub Subs
        # these topics are for visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, '/pf/viz/poly1', 1)

        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, '/pf/pose/odom', 1)

        # these topics are for coordinate space things
        self.pub_tf = TransformBroadcaster(self)

        # Laser offset parameters are loaded in parameters section

        # these topics are to receive data from the racecar
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.lidarCB,
            1)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odometry_topic').value,
            self.odomCB,
            1)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.clicked_pose,
            1)
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_pose,
            1)

        self.get_logger().info('Finished initializing, waiting on messages...')

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''

        # Wait for map_server lifecycle node to reach ACTIVE state (id == 3)
        state_client = self.create_client(GetState, '/map_server/get_state')
        self.get_logger().info('Waiting for map_server lifecycle node to reach ACTIVE state...')
        while True:
            if state_client.wait_for_service(timeout_sec=1.0):
                req = GetState.Request()
                future = state_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                if future.result() is not None:
                    curr_state = future.result().current_state
                    if curr_state.id == 3:  # 3 = ACTIVE
                        self.get_logger().info('map_server is ACTIVE! Requesting map...')
                        break
            self.get_logger().info('Waiting for map_server to activate...')
            time.sleep(0.5)

        map_msg = None
        while map_msg is None:
            while not self.map_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Get map service not available, waiting...')
            req = GetMap.Request()
            future = self.map_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                msg = future.result().map
                if msg.info.resolution > 1e-6 and msg.info.width > 0 and msg.info.height > 0 and len(msg.data) > 0:
                    map_msg = msg
                    break
            time.sleep(0.5)

        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        self.get_logger().info('Done loading map')

         # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1
        self.map_initialized = True

    def publish_tf(self, pose, stamp=None):
        ''' Publish a tf for the car. This tells ROS where the car is with respect to the map. '''
        stamp = self.get_clock().now().to_msg()

        if self.last_pose is None:
            return

        # Compute map -> odom transform
        x_m, y_m, theta_m = pose
        x_o, y_o, theta_o = self.last_pose

        theta_mo = theta_m - theta_o
        x_mo = x_m - x_o * np.cos(theta_mo) + y_o * np.sin(theta_mo)
        y_mo = y_m - x_o * np.sin(theta_mo) - y_o * np.cos(theta_mo)

        t = TransformStamped()
        # header
        t.header.stamp = stamp
        t.header.frame_id = self.GLOBAL_FRAME_ID
        t.child_frame_id = self.ODOM_FRAME_ID
        # translation
        t.transform.translation.x = x_mo
        t.transform.translation.y = y_mo
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0., 0., theta_mo)
        # rotation
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.pub_tf.sendTransform(t)
        # also publish odometry to facilitate getting the localization pose
        if self.PUBLISH_ODOM:
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = self.GLOBAL_FRAME_ID
            odom.child_frame_id = self.BASE_FRAME_ID
            odom.pose.pose.position.x = pose[0]
            odom.pose.pose.position.y = pose[1]
            odom.pose.pose.orientation = Utils.angle_to_quaternion(pose[2])
            # Calculate 3x3 covariance matrix (x, y, theta) using circular statistics for yaw
            spread = self.particles - self.inferred_pose
            spread[:, 2] = np.arctan2(np.sin(spread[:, 2]), np.cos(spread[:, 2]))
            spread_weighted = spread * self.weights[:, np.newaxis]
            cov_mat = np.dot(spread_weighted.T, spread)
            
            # Map 3D state covariance to 6D state covariance (6x6 matrix)
            ros_cov = np.zeros(36)
            
            # Row 0 (X): indices 0 (X), 1 (Y), 5 (Yaw)
            ros_cov[0] = cov_mat[0, 0] # var(x)
            ros_cov[1] = cov_mat[0, 1] # cov(x, y)
            ros_cov[5] = cov_mat[0, 2] # cov(x, yaw)
            
            # Row 1 (Y): indices 6 (X), 7 (Y), 11 (Yaw)
            ros_cov[6] = cov_mat[1, 0] # cov(y, x)
            ros_cov[7] = cov_mat[1, 1] # var(y)
            ros_cov[11] = cov_mat[1, 2] # cov(y, yaw)
            
            # Row 5 (Yaw): indices 30 (X), 31 (Y), 35 (Yaw)
            ros_cov[30] = cov_mat[2, 0] # cov(yaw, x)
            ros_cov[31] = cov_mat[2, 1] # cov(yaw, y)
            ros_cov[35] = cov_mat[2, 2] # var(yaw)
            
            odom.pose.covariance = ros_cov
            odom.twist.twist.linear.x = self.current_speed
            self.odom_pub.publish(odom)
        
        return

    def visualize(self):
        '''
        Publish various visualization messages.
        '''
        if not self.DO_VIZ:
            return

        if self.pose_pub.get_subscription_count() > 0 and isinstance(self.inferred_pose, np.ndarray):
            # Publish the inferred pose for visualization
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.GLOBAL_FRAME_ID
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            # publish a downsampled version of the particle distribution to avoid a lot of latency
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                # randomly downsample particles
                proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
                # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
                self.publish_particles(self.particles[proposal_indices,:])
            else:
                self.publish_particles(self.particles)

        if self.pub_fake_scan.get_subscription_count() > 0 and isinstance(self.ranges, np.ndarray):
            # generate the scan from the point of view of the inferred position for visualization
            self.viz_queries[:,0] = self.inferred_pose[0]
            self.viz_queries[:,1] = self.inferred_pose[1]
            self.viz_queries[:,2] = self.downsampled_angles + self.inferred_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(self.downsampled_angles, self.viz_ranges)

    def publish_particles(self, particles):
        # publish the given particles as a PoseArray object
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.GLOBAL_FRAME_ID
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        # publish the given angels and ranges as a laser scan message
        ls = LaserScan()
        ls.header.stamp = self.last_stamp
        ls.header.frame_id = self.LASER_FRAME_ID
        ls.angle_min = np.min(angles)
        ls.angle_max = np.max(angles)
        ls.angle_increment = np.abs(angles[0] - angles[1])
        ls.range_min = 0
        ls.range_max = np.max(ranges)
        ls.ranges = ranges
        self.pub_fake_scan.publish(ls)

    def lidarCB(self, msg):
        '''
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        '''
        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info('...Received first LiDAR message')
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = np.copy(self.laser_angles[0::self.ANGLE_STEP]).astype(np.float32)
            self.viz_queries = np.zeros((self.downsampled_angles.shape[0],3), dtype=np.float32)
            self.viz_ranges = np.zeros(self.downsampled_angles.shape[0], dtype=np.float32)
            self.get_logger().info(str(self.downsampled_angles.shape[0]))

        # store the necessary scanner information for later processing
        raw_ranges = np.array(msg.ranges[::self.ANGLE_STEP], dtype=np.float32)
        # Replace NaN and Inf with max range
        raw_ranges[np.isnan(raw_ranges)] = float(self.MAX_RANGE_METERS)
        raw_ranges[np.isinf(raw_ranges)] = float(self.MAX_RANGE_METERS)
        # Clip to [0, max_range]
        self.downsampled_ranges = np.clip(raw_ranges, 0.0, float(self.MAX_RANGE_METERS))
        self.lidar_initialized = True
        self.new_lidar_data = True
        # self.update()

    def odomCB(self, msg):
        '''
        Store deltas between consecutive odometry messages in the coordinate space of the car.

        Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
        '''
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot*delta).transpose()
            
            dtheta = orientation - self.last_pose[2]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

            # Reject physically impossible EKF jumps (e.g. EKF resets, telemetry glitches)
            displacement = np.sqrt(local_delta[0,0]**2 + local_delta[0,1]**2)
            if displacement > 0.5 or abs(dtheta) > 1.0:
                self.get_logger().warn(f"Ignoring EKF jump: dist={displacement:.3f}m, dtheta={dtheta:.3f}rad")
                self.last_pose = pose
                return

            self.odometry_data += np.array([local_delta[0,0], local_delta[0,1], dtheta])
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
        else:
            self.get_logger().info('...Received first Odometry message')
            self.last_pose = pose

        # this topic is slower than lidar, so update every time we receive a message
        self.update()

    def clicked_pose(self, msg):
        '''
        Receive pose messages from RViz and initialize the particle distribution in response.
        '''
        self.get_logger().info(f"clicked_pose callback triggered! Received message of type: {type(msg)}")
        if isinstance(msg, PointStamped):
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            self.initialize_particles_pose(msg.pose.pose, msg.pose.covariance)

    def initialize_particles_pose(self, pose, covariance=None):
        '''
        Initialize particles in the general region of the provided pose.
        '''
        self.get_logger().info('SETTING POSE')
        self.get_logger().info(str([pose.position.x, pose.position.y]))
        
        # Default standard deviations (fallbacks if covariance is zero/empty)
        x_std = 0.5
        y_std = 0.5
        yaw_std = 0.4
        
        if covariance is not None and len(covariance) == 36:
            x_var = covariance[0]
            y_var = covariance[7]
            yaw_var = covariance[35]
            if x_var > 1e-4:
                x_std = np.sqrt(x_var)
            if y_var > 1e-4:
                y_std = np.sqrt(y_var)
            if yaw_var > 1e-4:
                yaw_std = np.sqrt(yaw_var)

        with self.state_lock:
            self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
            self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0, scale=x_std, size=self.MAX_PARTICLES)
            self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0, scale=y_std, size=self.MAX_PARTICLES)
            self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0, scale=yaw_std, size=self.MAX_PARTICLES)
            self.pose_reset = True
            self.last_pose = None

    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.
        '''
        self.get_logger().info('GLOBAL INITIALIZATION')
        # randomize over grid coordinate space
        with self.state_lock:
            permissible_x, permissible_y = np.where(self.permissible_region == 1)
            indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

            permissible_states = np.zeros((self.MAX_PARTICLES,3))
            permissible_states[:,0] = permissible_y[indices]
            permissible_states[:,1] = permissible_x[indices]
            permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

            Utils.map_to_world(permissible_states, self.map_info)
            self.particles = permissible_states
            self.weights[:] = 1.0 / self.MAX_PARTICLES
            self.pose_reset = True
            self.last_pose = None

    def precompute_sensor_model(self):
        '''
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        self.get_logger().info('Precomputing sensor model')
        # sensor model constants
        z_short = self.Z_SHORT
        z_max   = self.Z_MAX
        z_rand  = self.Z_RAND
        z_hit   = self.Z_HIT
        sigma_hit = self.SIGMA_HIT
        
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width,table_width))

        t = time.time()
        # d is the computed range from RangeLibc
        for d in range(table_width):
            norm = 0.0
            sum_unkown = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(r-d)
                # reflects from the intended object
                prob += z_hit * np.exp(-(z*z)/(2.0*sigma_hit*sigma_hit)) / (sigma_hit * np.sqrt(2.0*np.pi))

                # observed range is less than the predicted range - short reading
                if r < d:
                    prob += 2.0 * z_short * (d - r) / float(d)

                # erroneous max range measurement
                if int(r) == int(self.MAX_RANGE_PX):
                    prob += z_max

                # random measurement
                if r < int(self.MAX_RANGE_PX):
                    prob += z_rand * 1.0/float(self.MAX_RANGE_PX)

                norm += prob
                self.sensor_model_table[int(r),int(d)] = prob

            # normalize
            self.sensor_model_table[:,int(d)] /= norm

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'motion_model':
                self.MOTION_MODEL = param.value
            elif param.name == 'motion_dispersion_x':
                self.MOTION_DISPERSION_X = param.value
            elif param.name == 'motion_dispersion_y':
                self.MOTION_DISPERSION_Y = param.value
            elif param.name == 'motion_dispersion_theta':
                self.MOTION_DISPERSION_THETA = param.value
            elif param.name == 'motion_dispersion_arc_x':
                self.MOTION_DISPERSION_ARC_X = param.value
            elif param.name == 'motion_dispersion_arc_y':
                self.MOTION_DISPERSION_ARC_Y = param.value
            elif param.name == 'motion_dispersion_arc_theta':
                self.MOTION_DISPERSION_ARC_THETA = param.value
            elif param.name == 'motion_dispersion_arc_xy':
                self.MOTION_DISPERSION_ARC_XY = param.value
            elif param.name == 'motion_dispersion_arc_x_min':
                self.MOTION_DISPERSION_ARC_X_MIN = param.value
            elif param.name == 'motion_dispersion_arc_y_min':
                self.MOTION_DISPERSION_ARC_Y_MIN = param.value
            elif param.name == 'motion_dispersion_arc_y_max':
                self.MOTION_DISPERSION_ARC_Y_MAX = param.value
            elif param.name == 'motion_dispersion_arc_theta_min':
                self.MOTION_DISPERSION_ARC_THETA_MIN = param.value
            elif param.name == 'motion_dispersion_arc_xy_min_x':
                self.MOTION_DISPERSION_ARC_XY_MIN_X = param.value
            elif param.name == 'enable_recovery':
                self.ENABLE_RECOVERY = param.value
            elif param.name == 'recovery_alpha_slow':
                self.RECOVERY_ALPHA_SLOW = param.value
            elif param.name == 'recovery_alpha_fast':
                self.RECOVERY_ALPHA_FAST = param.value
            elif param.name == 'recovery_max_fraction':
                self.RECOVERY_MAX_FRACTION = param.value
            elif param.name == 'ess_threshold_ratio':
                self.ESS_THRESHOLD_RATIO = param.value
            elif param.name == 'out_of_map_penalty':
                self.OUT_OF_MAP_PENALTY = param.value
            elif param.name == 'pose_estimation_method':
                self.POSE_ESTIMATION_METHOD = param.value
        return SetParametersResult(successful=True, reason="Parameters updated successfully")

    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution.
        Supports both the standard community model (default) and the SynPF arc-based model.
        '''
        if self.MOTION_MODEL == 'arc':
            dx_, dy_, dtheta_ = action

            scale_x = max(self.MOTION_DISPERSION_ARC_X_MIN, abs(dx_) * self.MOTION_DISPERSION_ARC_X)
            scale_y = abs(dy_) * self.MOTION_DISPERSION_ARC_Y
            if abs(dx_) > self.MOTION_DISPERSION_ARC_XY_MIN_X:
                scale_y += (abs(dx_) - self.MOTION_DISPERSION_ARC_XY_MIN_X) * self.MOTION_DISPERSION_ARC_XY
            scale_y = min(self.MOTION_DISPERSION_ARC_Y_MAX, max(self.MOTION_DISPERSION_ARC_Y_MIN, scale_y))
            scale_th = max(self.MOTION_DISPERSION_ARC_THETA_MIN, abs(dtheta_) * self.MOTION_DISPERSION_ARC_THETA)

            np.random.normal(loc=dx_, scale=scale_x, size=self.MAX_PARTICLES, out=self.motion_dx)
            np.random.normal(loc=dy_, scale=scale_y, size=self.MAX_PARTICLES, out=self.motion_dy)
            np.random.normal(loc=dtheta_, scale=scale_th, size=self.MAX_PARTICLES, out=self.motion_dtheta)

            abs_dth = np.abs(self.motion_dtheta)
            straight_mask = abs_dth < 1e-6
            curved_mask = ~straight_mask

            theta = proposal_dist[:, 2]
            theta_next = theta + self.motion_dtheta

            if np.any(curved_mask):
                th_c = theta[curved_mask]
                dth_c = self.motion_dtheta[curved_mask]
                r = self.motion_dx[curved_mask] / dth_c
                cx = proposal_dist[curved_mask, 0] - r * np.sin(th_c)
                cy = proposal_dist[curved_mask, 1] + r * np.cos(th_c)
                proposal_dist[curved_mask, 0] = cx + r * np.sin(th_c + dth_c)
                proposal_dist[curved_mask, 1] = cy - r * np.cos(th_c + dth_c)

            if np.any(straight_mask):
                th_s = theta[straight_mask]
                proposal_dist[straight_mask, 0] += self.motion_dx[straight_mask] * np.cos(th_s)
                proposal_dist[straight_mask, 1] += self.motion_dx[straight_mask] * np.sin(th_s)

            lat_angle = theta_next + (np.pi / 2.0)
            proposal_dist[:, 0] += self.motion_dy * np.cos(lat_angle)
            proposal_dist[:, 1] += self.motion_dy * np.sin(lat_angle)
            proposal_dist[:, 2] = np.arctan2(np.sin(theta_next), np.cos(theta_next))

        else:
            cosines = np.cos(proposal_dist[:, 2])
            sines = np.sin(proposal_dist[:, 2])

            self.local_deltas[:, 0] = cosines * action[0] - sines * action[1]
            self.local_deltas[:, 1] = sines * action[0] + cosines * action[1]
            self.local_deltas[:, 2] = action[2]

            proposal_dist += self.local_deltas
            proposal_dist[:, 0] += np.random.normal(loc=0.0, scale=self.MOTION_DISPERSION_X, size=self.MAX_PARTICLES)
            proposal_dist[:, 1] += np.random.normal(loc=0.0, scale=self.MOTION_DISPERSION_Y, size=self.MAX_PARTICLES)
            proposal_dist[:, 2] += np.random.normal(loc=0.0, scale=self.MOTION_DISPERSION_THETA, size=self.MAX_PARTICLES)
            proposal_dist[:, 2] = np.arctan2(np.sin(proposal_dist[:, 2]), np.cos(proposal_dist[:, 2]))

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probabilistic weight for each particle in the proposal distribution.
        '''
        num_rays = self.downsampled_angles.shape[0]
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros((num_rays * self.MAX_PARTICLES, 3), dtype=np.float32)
            else:
                self.queries = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)

            self.ranges = np.zeros(num_rays * self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        # Offset the raycasting queries to the actual LiDAR sensor frame origin into preallocated self.queries_pose
        theta = proposal_dist[:, 2]
        self.queries_pose[:, 0] = proposal_dist[:, 0] + self.laser_offset_x * np.cos(theta) - self.laser_offset_y * np.sin(theta)
        self.queries_pose[:, 1] = proposal_dist[:, 1] + self.laser_offset_x * np.sin(theta) + self.laser_offset_y * np.cos(theta)
        self.queries_pose[:, 2] = theta

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if 'cddt' in self.WHICH_RM:
                self.queries[:, :] = self.queries_pose[:, :]
                self.range_method.calc_range_many_radial_optimized(num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                self.get_logger().info('Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2')
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:, :] = self.queries_pose[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(self.queries, self.downsampled_angles, obs, self.weights)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            self.queries[:, :] = self.queries_pose[:, :]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(self.queries, self.downsampled_angles, self.ranges)
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # evaluate the sensor model
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(str(['sensor_model: init: ', np.round((t_init-t_start)/t_total, 2), 'range:', np.round((t_range-t_init)/t_total, 2), \
                      'eval:', np.round((t_eval-t_range)/t_total, 2), 'squash:', np.round((t_squash-t_eval)/t_total, 2)]))
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:,0] = np.repeat(queries_pose[:,0], num_rays)
            self.queries[:,1] = np.repeat(queries_pose[:,1], num_rays)
            self.queries[:,2] = np.repeat(queries_pose[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:,0] = np.repeat(queries_pose[:,0], num_rays)
            self.queries[:,1] = np.repeat(queries_pose[:,1], num_rays)
            self.queries[:,2] = np.repeat(queries_pose[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(self.sensor_model_table[intobs,intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info('PLEASE SET rangelib_variant PARAM to 0-4')

    def MCL(self, a, o):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        '''
        if self.SHOW_FINE_TIMING:
            t = time.time()

        # Determine if we should inject random particles for recovery
        p_inject = 0.0
        if self.ENABLE_RECOVERY and self.w_slow > 0.0:
            # Ratio of short-term to long-term average weight
            p_inject = max(0.0, 1.0 - self.w_fast / self.w_slow)

        num_injected = 0
        if p_inject > 0.0:
            # Scale injection count by the probability and the maximum allowed fraction
            num_injected = int(p_inject * self.RECOVERY_MAX_FRACTION * self.MAX_PARTICLES)

        num_resampled = self.MAX_PARTICLES - num_injected

        # Calculate Effective Sample Size (ESS)
        ess = 1.0 / max(1e-9, np.sum(self.weights ** 2))
        resampled = False
        proposal_distribution = self.proposal_distribution

        # Only resample if ESS is below threshold or if we are injecting recovery particles
        if num_resampled > 0:
            if ess < self.MAX_PARTICLES * self.ESS_THRESHOLD_RATIO or num_injected > 0:
                resampled = True
                # Vectorized Low Variance Resampling via cumulative distribution search
                positions = (np.random.random() + np.arange(num_resampled)) / num_resampled
                cumulative_sum = np.cumsum(self.weights)
                
                proposal_indices = np.searchsorted(cumulative_sum, positions)
                np.clip(proposal_indices, 0, self.MAX_PARTICLES - 1, out=proposal_indices)
                        
                proposal_distribution[:num_resampled, :] = self.particles[proposal_indices, :]
                
                # Reset weights for resampled particles
                self.weights.fill(1.0 / float(self.MAX_PARTICLES))
            else:
                # ESS is high enough, keep particles and skip resampling
                proposal_distribution[:num_resampled, :] = self.particles[:num_resampled, :]

        # Inject random recovery particles
        if num_injected > 0:
            permissible_x, permissible_y = np.where(self.permissible_region == 1)
            indices = np.random.randint(0, len(permissible_x), size=num_injected)

            random_states = np.zeros((num_injected, 3), dtype=np.float64)
            random_states[:, 0] = permissible_y[indices]
            random_states[:, 1] = permissible_x[indices]
            random_states[:, 2] = np.random.random(num_injected) * np.pi * 2.0

            Utils.map_to_world(random_states, self.map_info)
            proposal_distribution[num_resampled:, :] = random_states

        if self.SHOW_FINE_TIMING:
            t_propose = time.time()

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, a)
        if self.SHOW_FINE_TIMING:
            t_motion = time.time()

        # compute the sensor model
        old_weights = np.copy(self.weights) if not resampled else None
        
        self.sensor_model(proposal_distribution, o, self.weights)
        
        if not resampled:
            self.weights *= old_weights

        # Check for permissible region and downscale weight if a particle goes out of track
        particles_in_map = np.copy(proposal_distribution)
        Utils.world_to_map(particles_in_map, self.map_info)
        limit = self.permissible_region.shape
        x_coords = np.clip(particles_in_map[:, 0].astype(np.int32), 0, limit[1] - 1)
        y_coords = np.clip(particles_in_map[:, 1].astype(np.int32), 0, limit[0] - 1)
        valid_particles = self.permissible_region[y_coords, x_coords]
        self.weights = np.where(valid_particles == 1, self.weights, self.OUT_OF_MAP_PENALTY * self.weights)
        if self.SHOW_FINE_TIMING:
            t_sensor = time.time()

        # Calculate sum of weights before normalization
        weight_sum = np.sum(self.weights)

        # Update weight running averages for recovery tracking
        if self.ENABLE_RECOVERY:
            w_avg = weight_sum / float(self.MAX_PARTICLES)
            if self.w_slow == 0.0:
                self.w_slow = w_avg
                self.w_fast = w_avg
            else:
                self.w_slow += self.RECOVERY_ALPHA_SLOW * (w_avg - self.w_slow)
                self.w_fast += self.RECOVERY_ALPHA_FAST * (w_avg - self.w_fast)

        # normalize importance weights safely
        if weight_sum < 1e-9:
            self.weights.fill(1.0 / float(self.MAX_PARTICLES))
        else:
            self.weights /= weight_sum

        if self.SHOW_FINE_TIMING:
            t_norm = time.time()
            t_total = (t_norm - t)/100.0

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(str(['MCL: propose: ', np.round((t_propose-t)/t_total, 2), 'motion:', np.round((t_motion-t_propose)/t_total, 2), \
                  'sensor:', np.round((t_sensor-t_motion)/t_total, 2), 'norm:', np.round((t_norm-t_sensor)/t_total, 2)]))

        # save the particles
        self.particles = proposal_distribution
    
    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        if self.POSE_ESTIMATION_METHOD == 'map':
            # Maximum A Posteriori (single highest-weight particle)
            max_idx = np.argmax(self.weights)
            return np.copy(self.particles[max_idx, :])

        elif self.POSE_ESTIMATION_METHOD == 'local_mean':
            # Local mean around the MAP particle (refines MAP, avoids multi-modal averages)
            max_idx = np.argmax(self.weights)
            map_particle = self.particles[max_idx, :]

            # Find particles within 0.5 meters of the MAP particle
            dx = self.particles[:, 0] - map_particle[0]
            dy = self.particles[:, 1] - map_particle[1]
            distances_sq = dx**2 + dy**2
            in_cluster = distances_sq < 0.5**2

            cluster_weights = self.weights[in_cluster]
            weight_sum = np.sum(cluster_weights)
            if weight_sum > 1e-9:
                cluster_weights /= weight_sum
                x = np.dot(self.particles[in_cluster, 0], cluster_weights)
                y = np.dot(self.particles[in_cluster, 1], cluster_weights)
                sin_theta = np.dot(np.sin(self.particles[in_cluster, 2]), cluster_weights)
                cos_theta = np.dot(np.cos(self.particles[in_cluster, 2]), cluster_weights)
                theta = np.arctan2(sin_theta, cos_theta)
                return np.array([x, y, theta])
            else:
                return np.copy(map_particle)

        else: # 'mean' (Weighted average)
            # Use circular mean for the heading to avoid mathematical errors near +/- pi
            x = np.dot(self.particles[:, 0], self.weights)
            y = np.dot(self.particles[:, 1], self.weights)

            sin_theta = np.dot(np.sin(self.particles[:, 2]), self.weights)
            cos_theta = np.dot(np.cos(self.particles[:, 2]), self.weights)
            theta = np.arctan2(sin_theta, cos_theta)

            return np.array([x, y, theta])

    def update(self):
        '''
        Apply the MCL function to update particle filter state. 

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            if not self.state_lock.acquire(blocking=False):
                # self.get_logger().info('Concurrency error avoided')
                return
            try:
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
 
                # run the MCL update algorithm only if the car has actually moved to prevent stationary drift
                dx, dy, dtheta = action
                moved = (np.sqrt(dx**2 + dy**2) > 1e-4) or (abs(dtheta) > 1e-4)
 
                if moved or self.inferred_pose is None or self.pose_reset:
                    # Reset the accumulated odometry now that we are applying it
                    self.odometry_data = np.zeros(3)

                    if self.pose_reset:
                        # Recalculate pose immediately from new particles when clicked/reset
                        self.inferred_pose = self.expected_pose()
                        self.pose_reset = False
                    else:
                        if self.new_lidar_data:
                            self.MCL(action, observation)
                            self.new_lidar_data = False
                        else:
                            self.motion_model(self.particles, action)
 
                        # compute the expected value of the robot pose
                        self.inferred_pose = self.expected_pose()
            finally:
                self.state_lock.release()
            t2 = time.time()

            # publish transformation frame based on inferred pose
            self.publish_tf(self.inferred_pose, self.last_stamp)

            # this is for tracking particle filter speed
            ips = 1.0 / (t2 - t1)
            self.smoothing.append(ips)
            if self.iters % 10 == 0:
                self.get_logger().info(str(['iters per sec:', int(self.timer.fps()), ' possible:', int(self.smoothing.mean())]))

            self.visualize()

# import argparse
# import sys
# parser = argparse.ArgumentParser(description='Particle filter.')
# parser.add_argument('--config', help='Path to yaml file containing config parameters. Helpful for calling node directly with Python for profiling.')

# def load_params_from_yaml(fp):
#     from yaml import load
#     with open(fp, 'r') as infile:
#         yaml_data = load(infile)
#         for param in yaml_data:
#             print 'param:', param, ':', yaml_data[param]
#             rospy.set_param('~'+param, yaml_data[param])

# # this function can be used to generate flame graphs easily
# def make_flamegraph(filterx=None):
#     import flamegraph, os
#     perf_log_path = os.path.join(os.path.dirname(__file__), '../tmp/perf.log')
#     flamegraph.start_profile_thread(fd=open(perf_log_path, 'w'),
#                                     filter=filterx,
#                                     interval=0.001)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    rclpy.spin(pf)

if __name__ == '__main__':
    main()

# if __name__=='__main__':
#     rospy.init_node('particle_filter')

#     args,_ = parser.parse_known_args()
#     if args.config:
#         load_params_from_yaml(args.config)

#     # make_flamegraph(r'update')

#     pf = ParticleFiler()
#     rospy.spin()

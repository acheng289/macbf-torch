import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import macbf_torch.core as core
import macbf_torch.config as config
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from dynus_interfaces.msg import State, Goal
from ament_index_python import get_package_share_directory
import os
from pathlib import Path
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

class MACSimNode(Node):
    """
    A ROS 2 node that simulates the MACBF-Torch model for a single agent in a multi-agent system.
    This node initializes the state of the agent, subscribes to goals and other agents' states,
    and publishes the agent's state at a fixed rate. It uses a neural network model to compute
    the agent's action based on the current state and the reference state derived from the goal.
    """
    def __init__(self):
        super().__init__('mac_sim_node')

        self.get_logger().info('Initializing MACSimNode...')
        
        # Declare parameters
        self.declare_parameter("start_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("start_yaw", 0.0)
        self.declare_parameter("all_agents", ['NX01'])
        self.declare_parameter("self_agent", 'NX01')
        self.declare_parameter("model_weight_file", 'macbf_torch_weights.pt')

        self.start_pose = self.get_parameter("start_pose").get_parameter_value().double_array_value
        self.start_yaw = self.get_parameter("start_yaw").get_parameter_value().double_value
        self.self_agent = self.get_parameter("self_agent").get_parameter_value().string_value
        self.all_agents = self.get_parameter("all_agents").get_parameter_value().string_array_value
        self.model_weight_file = self.get_parameter("model_weight_file").get_parameter_value().string_value

        # Initalize callback groups
        # No need because by default, the default callback group is MutuallyExclusiveCallbackGroup
        # Create one for the state publisher to execute in parallel
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # Initialize state matrix
        self.state = State()
        self.state.pos.x = float(self.start_pose[0])
        self.state.pos.y = float(self.start_pose[1])
        self.state.pos.z = float(self.start_pose[2])
        roll, pitch = 0.0, 0.0
        # Returns in the order x, y, z, w
        rotQuat = R.from_euler('xyz', [roll, pitch, self.start_yaw], degrees=True).as_quat()
        self.state.quat.x = rotQuat[0]
        self.state.quat.y = rotQuat[1]
        self.state.quat.z = rotQuat[2]
        self.state.quat.w = rotQuat[3]

        self.current_state_vector = np.array([
            self.state.pos.x,
            self.state.pos.y,
            self.state.pos.z,
            self.state.vel.x,
            self.state.vel.y,
            self.state.vel.z,
            self.state.quat.x,
            self.state.quat.y,
            self.state.quat.z,
            self.state.quat.w
        ], dtype=np.float32)

        # Create (NxS) state matrix, where N is the number of agents and S is the state size
        self.state_matrix = np.zeros((len(self.all_agents), 10), dtype=np.float32)
        self.state_matrix[0] = self.current_state_vector

        # Create timer that publishes state at 10Hz, executes in parallel with the other ME callbacks
        self.timer = self.create_timer(0.1, self.publish_state, callback_group=self.me_cb_group)

        # Initialize model
        self.model_weight = (
            Path(get_package_share_directory('macbf_torch')) /
            'models' /
            self.model_weight_file
        ).as_posix()
        self.get_logger().info(f'Using model weight file: {self.model_weight}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_net = core.NetworkAction()
        checkpoint = torch.load(self.model_weight, map_location=self.device)
        self.action_net.load_state_dict(checkpoint['action_net_state_dict'])
        self.action_net.to(self.device)
        self.action_net.eval()
        self.get_logger().info('Model loaded successfully')

        # Initialize publishers and subscribers
        # Same as original fake sim, with reliable and volatile QoS
        self.state_publisher = self.create_publisher(State, f'/{self.self_agent}/state', QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        ))

        # Subscribe to own goal
        self.goal_subscriber = self.create_subscription(Goal, f'/{self.self_agent}/goal', self.goal_callback, 10)

        # Subscribe to other agents' states
        agent_index = 0
        for agent in self.all_agents:
            if agent != self.self_agent:
                agent_index += 1
                self.create_subscription(
                    State,
                    f'/{agent}/state',
                    self.create_agent_state_callback(agent_index),
                    QoSProfile(
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE,
                        depth=10
                    ) # Documentation says that this is actually the default QoS
                )
        
        self.get_logger().info(f'MACSimNode for {self.self_agent} initialized')

    # On state callback, update the state matrix    
    def create_agent_state_callback(self, agent_index):
        def agent_state_callback(msg):
            # Update the state matrix with the received state
            self.state_matrix[agent_index] = np.array([
                msg.pos.x,
                msg.pos.y,
                msg.pos.z,
                msg.vel.x,
                msg.vel.y,
                msg.vel.z,
                msg.quat.x,
                msg.quat.y,
                msg.quat.z,
                msg.quat.w
            ], dtype=np.float32)
            self.get_logger().info(f'Updated state for agent {agent_index}: {msg}')
        
        return agent_state_callback
    
    # On goal callback, compute the updated state for self agent, update state matrix, put state matrix through model to get action
    # Compute next actual state and publish
    def goal_callback(self, goal):
        # Extract thrust
        thrust = np.array([
            goal.a.x,
            goal.a.y,
            goal.a.z + 9.81
        ])

        # Normalize thrust and extract values
        thrust_normalized = thrust / np.linalg.norm(thrust)
        a = thrust_normalized[0]
        b = thrust_normalized[1]
        c = thrust_normalized[2]

        tmp = 1 / np.sqrt(2 * (1 + c))

        # Compute the quaternion values, convention is x, y, z, w
        qabc_values = [-b * tmp, a * tmp, 0.0, tmp * (1 + c)]
        qabc = R.from_quat(qabc_values)

        qpsi_values = [0.0, 0.0, np.sin(goal.yaw / 2), np.cos(goal.yaw / 2)]
        qpsi = R.from_quat(qpsi_values)

        # Quaternion multiplication
        w_q_b = qabc * qpsi

        ref_state_vector = np.array([
            goal.p.x,
            goal.p.y,
            goal.p.z,
            goal.v.x,
            goal.v.y,
            goal.v.z,
            w_q_b.as_quat()[0],
            w_q_b.as_quat()[1],
            w_q_b.as_quat()[2],
            w_q_b.as_quat()[3]
        ])

        ref_state_matrix = self.state_matrix.copy()
        ref_state_matrix[0] = ref_state_vector

        # Convert to torch tensor
        ref_state_tensor = torch.tensor(ref_state_matrix, device=self.device, dtype=torch.float32)
        state_matrix_tensor = torch.tensor(self.state_matrix, device=self.device, dtype=torch.float32)

        # Compute action using the model
        with torch.no_grad():
            action = self.action_net(state_matrix_tensor, ref_state_tensor)
        
        # Compute the next state based on the action
        dsdt = action.cpu().numpy()
        next_state_vector = self.state_matrix[0] + dsdt * config.TIME_STEP
        self.get_logger().info(f'Computed next state vector: {next_state_vector}')

        # Update the state matrix with the new state
        self.state_matrix[0] = next_state_vector

        # Publishing of new state handled by timer

    def publish_state(self):
        current_state_vector = self.state_matrix[0]
        self.state.pos.x = current_state_vector[0].item()
        self.state.pos.y = current_state_vector[1].item()
        self.state.pos.z = current_state_vector[2].item()
        self.state.vel.x = current_state_vector[3].item()
        self.state.vel.y = current_state_vector[4].item()
        self.state.vel.z = current_state_vector[5].item()
        self.state.quat.x = current_state_vector[6].item()
        self.state.quat.y = current_state_vector[7].item()
        self.state.quat.z = current_state_vector[8].item()
        self.state.quat.w = current_state_vector[9].item()
        self.get_logger().info(f'Publishing state: {self.state}')
        self.state_publisher.publish(self.state)

def main(args=None):
    rclpy.init(args=args)
    node = MACSimNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down...')
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

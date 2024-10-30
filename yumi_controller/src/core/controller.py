from typing import Tuple, Dict, Optional
from abc import ABCMeta, abstractmethod

import rospy
import numpy as np

from enum import Enum, auto

from std_msgs.msg import Float64MultiArray
from abb_robot_msgs.msg import SystemState
from abb_robot_msgs.srv import TriggerWithResultCode
from abb_rapid_sm_addin_msgs.srv import SetSGCommand


from . import tasks, hqp
from .robot_state import YumiDualStateUpdater
from .parameters import Parameters

from dynamics.utils import RobotState
from trajectory.polynomial import CubicTrajectory


# TODO move routines to a new file (perhaps a module?)
# TODO transform  self._controller.policy()  in a DefaultRoutine

class Routine(object, metaclass=ABCMeta):
    
    def __init__(self, *args) -> None:
        pass
    
    @abstractmethod
    def init(self, robot_state_init: RobotState) -> None:
        raise NotImplementedError()
     
    @abstractmethod   
    def action(self, robot_state_curr: RobotState) -> Tuple[dict, bool]:
        raise NotImplementedError()
    
    @abstractmethod
    def finish(self, robot_state_final: RobotState) -> None:
        raise NotImplementedError()

class RoutineStateMachine(object):
    
    class State(Enum):
        IDLING = auto()
        RUNNING = auto()
        COMPLETED = auto()
    
    def __init__(self, controller: "YumiDualController"):
        self._routines: Dict[str, Routine] = {}
        self._controller = controller
        # state
        self._status = RoutineStateMachine.State.IDLING
        self._request = None
        
    def register(self, name: str, routine: Routine) -> None:
        if name in self._routines:
            print("Routine name already registered")
            return
        self._routines[name] = routine
    
    def reset(self):
        self._status = RoutineStateMachine.State.IDLING
        self._request = None
        self._controller.reset()
        action = {
            "control_space": "joint_space",
            "joint_velocities": np.zeros(Parameters.dof)}
        return action
    
    def run(self, name: Optional[str]):
        
        if name is not None:
            if self._status == RoutineStateMachine.State.IDLING:
                try:
                    self._request = self._routines[name]
                    print(f"Running routine \"{name}\"")
                except Exception:
                    print(f"Routine \"{name}\" not registered! Resuming execution")
                    self._controller.reset()
            else:
                print("Cannot run a routine when another one is already running!")
        else:
            # no new routine requested, keep spinning
            pass
        
        if self._status == RoutineStateMachine.State.IDLING and self._request == None:
            # routine state machine is idling and nothing is requested
            action = self._controller.policy()
        
        if self._status == RoutineStateMachine.State.IDLING and self._request != None:
            # We enter here only if a routine has been requested or if we were already running a routine.
            # No action is created here because we immediately enter the following if block
            self._status = RoutineStateMachine.State.RUNNING
            self._request.init(self._controller.yumi_state)
        
        if self._status == RoutineStateMachine.State.RUNNING:
            action, done = self._request.action(self._controller.yumi_state)
            if done:
                self._status = RoutineStateMachine.State.COMPLETED
        
        if self._status == RoutineStateMachine.State.COMPLETED:
            # No routine is running anymore, remove the old routine.
            # No action is created here because we enter here only when the previous
            # routine .action() call produced done==True, thus we already have an action
            print("Routine done")
            self._status = RoutineStateMachine.State.IDLING
            self._request.finish(self._controller.yumi_state)
            self._request = None
            self._controller.reset()
        
        return action


class ResetPoseRoutine(Routine):
    
    def __init__(self, reset_position=Parameters.reset_pos, min_time: float = 2) -> None:
        super().__init__()
        self._reset_position = reset_position
        self._final_time_min = min_time
        self._final_time = self._final_time_min
        self._time = 0
        self._max_speed = 1  # rad/s
        
    def init(self, robot_state_init: RobotState) -> None:
        current_joint_position = robot_state_init.joint_pos
        max_error = np.max(self._reset_position - current_joint_position)
        min_time = max_error / self._max_speed
        self._final_time = max(min_time, self._final_time_min) 
        self._time = 0
        self._a0, self._a1, self._a2, self._a3 = CubicTrajectory.calculate_coefficients(
            current_joint_position, np.zeros(14), 
            self._reset_position, np.zeros(14), 
            self._final_time)

    def action(self, robot_state_curr: RobotState) -> Tuple[dict, bool]:
        current_joint_position = robot_state_curr.joint_pos
        
        # advance by time step
        self._time += Parameters.dt  # TODO super wrong, use real data

        # if final time is reached, exit with "done" state
        if self._time <= self._final_time:
            q, dq, _ = CubicTrajectory.calculate_trajectory(self._a0, self._a1, self._a2, self._a3, self._time)
            vel = dq + (q - current_joint_position)
            done = False
        else:
            vel = np.zeros(Parameters.dof)
            done = True
        
        action = {
            "control_space": "joint_space",
            "joint_velocities": vel}
        return action, done
    
    def finish(self, robot_state_final: RobotState) -> None:
        pass


###############################################################################
#                                CONTROLLERS                                  #
###############################################################################

class YumiDualController(object, metaclass=ABCMeta):
    """ Class for controlling YuMi, inherit this class and create your own 
        `.policy()` and `.clear()` functions. The `.policy()` function outputs 
        an action `dict`, which is then passed to the `._set_action()` function.
    """
    def __init__(self):
        # TODO make this controller-indipendent
        self.yumi_state = YumiDualStateUpdater(symmetry=0.)
        
        # routine variables
        self._routine_request = None
        self._routine_machine = RoutineStateMachine(self)
        self._routine_machine.register("reset_pose", ResetPoseRoutine())

        # signal "controller is ready"
        self._ready = False
        
        # inverse kinematics with HQP solver (and EGM stopper for safety when tasks are problematic)
        # self._stop_egm = rospy.ServiceProxy("/yumi/rws/sm_addin/stop_egm", TriggerWithResultCode)
        self._hqp_solver = hqp.HQPSolver()
        self._tasks: Dict[str, tasks.Task] = {}
        self._prepare_hqp_tasks()
        
        # command publishers
        self._pub_yumi = YumiVelocityCommand()
        self._pub_grip = YumiGrippersCommand() 
        
        # signal "robot can move" (i.e. EGM is ready)
        self._auto_mode = True
        # EGM error handler
        self._start_rapid = rospy.ServiceProxy("/yumi/rws/start_rapid", TriggerWithResultCode)
        rospy.Subscriber("/yumi/rws/system_states", SystemState, self._callback_yumi_state, queue_size=1, tcp_nodelay=False)
    
    def _callback_yumi_state(self, data: SystemState):
        rws_auto_mode = data.auto_mode
        # if auto_mode was off and now we are getting it back (eg. acknoledgment of EGM error)
        if not self._auto_mode and rws_auto_mode:
            print("Regained control (auto_mode=true)")
            self.on_control_regained()
            self._start_rapid.call()
            print("Restared RAPID")
        # if auto_mode was on and now we are losing it back (eg. EGM joint contraint violation)
        if self._auto_mode and not rws_auto_mode:
            print("Lost control (auto_mode=false)")
            self.on_control_lost()
        # update the robot state cache
        self._auto_mode = rws_auto_mode        

    # TODO handle this with the usual  .register_task("name", obj_task)  APIs
    #      though this time an ordered dict is needed
    def _prepare_hqp_tasks(self):
        """ Sets up tasks for the HQP solver
        """

        # joint position limit
        joint_pos_bound_lower = np.hstack([Parameters.joint_position_bound_lower, Parameters.joint_position_bound_lower])
        joint_pos_bound_upper = np.hstack([Parameters.joint_position_bound_upper, Parameters.joint_position_bound_upper])
        self._tasks["joint_position_bound"] = tasks.JointPositionBoundsTask(
            dof=Parameters.dof,
            bounds_lower=joint_pos_bound_lower,
            bounds_upper=joint_pos_bound_upper,
            timestep=Parameters.dt)

        # joint velocity limit
        joint_vel_limits = np.hstack([Parameters.joint_velocity_bound, Parameters.joint_velocity_bound])
        self._tasks["joint_velocity_bound"] = tasks.JointVelocityBoundsTask(
            dof=Parameters.dof,
            bounds_lower=-joint_vel_limits,
            bounds_upper=joint_vel_limits).compute()  # constant

        # control objective
        self._tasks["individual_control"] = tasks.IndividualControl(dof=Parameters.dof)
        self._tasks["right_control"] = tasks.RightControl(dof=Parameters.dof)
        self._tasks["left_control"] = tasks.LeftControl(dof=Parameters.dof)
        self._tasks["coordinated_control"] = tasks.CoordinatedControl(dof=Parameters.dof)
        self._tasks["absolute_control"] = tasks.AbsoluteControl(dof=Parameters.dof)
        self._tasks["relative_control"] = tasks.RelativeControl(dof=Parameters.dof)

        # elbow collision avoidance
        self._tasks["self_collision_elbow"] = tasks.ElbowProximity(
            dof=Parameters.dof,
            min_dist=Parameters.elbows_min_distance,
            timestep=Parameters.dt)
        
        # end effector collision avoidance
        self._tasks["end_effector_collision"] = tasks.EndEffectorProximity(
            dof=Parameters.dof,
            min_dist=Parameters.grippers_min_distance,
            timestep=Parameters.dt)
        
        # joint potential 
        weights = np.array([1., 1., 1., 1., 1., 1., 0.5, 
                            1., 1., 1., 1., 1., 1., 0.5])
        self._tasks["joint_position_potential"] = tasks.JointPositionPotential(
            dof=Parameters.dof,
            default_pos=Parameters.neutral_pos,
            weights=weights,  # less strict on the last wrist joints
            timestep=Parameters.dt)

    def start(self):
        """ Start allow commands to be sent. This function is BLOCKING and MUST
            be called after having initialized everything in the controller. 
            Use .pause() in another thread to undo this operation.
        """
        rate = rospy.Rate(Parameters.update_rate)
        self._ready = True
        while not rospy.is_shutdown():
            # fetch action and execute command
            if self._ready and self._auto_mode:
                # run requested routine (if None, run the policy)
                action = self._routine_machine.run(self._routine_request)
                self._routine_request = None
            else:
                # do not move if manual mode kicks in
                print("Controller not ready yet")
                action = self._routine_machine.reset()
            self._set_action(action)
            rate.sleep()
        # when the controller is shut down, send a stop command
        action = {
            "control_space": "joint_space",
            "joint_velocities": np.zeros(Parameters.dof)}
        self._set_action(action)
            
    def pause(self):
        """ Stop sending commands but keep receiveing updates. 
            Use .start() to undo this operation.
        """
        self._ready = False

    def on_control_regained(self):
        """ Decides what happens when control mode goes from "manual" to "auto". 
            By default, .reset() is invoked.
        """
        self.reset()

    def on_control_lost(self):
        """ Decides what happens when control mode goes from "auto" to "manual".
        """
        pass
    
    def request_routine(self, name: str):
        """ Set the routine request.
        """
        self._routine_request = name
    
    @abstractmethod
    def reset(self):
        """ Method called whenever a switch to routine mode happens or when EGM stops.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def policy(self):
        """ This function should generate velocity commands for the controller.
            There are three control modes, joint space control, individual 
            control in cartesian space with yumi_base_link as reference frame, 
            coordinated manipulation with absolute and relative control. 
            All the inverse kinematics are solved with an HQP solver. The output
            of this function will be used in the `_set_action(action)` function. 
            The state of the robot is found in `self.yumi_state`, in particular 
            the `joint_pos`, `pose_gripper_r`, and `pose_gripper_l` variables.
        """
        raise NotImplementedError()

    def _set_action(self, action: dict):
        """ Sets an action and controls the robot.
            :param action: dict(), this input parameter has to contain certain keywords.
            :key `action["routine_*"]`: specific commands (eg. `"routine_reset_pose"`)
            :key `action["control_space"]`: determines which control mode {`"joint_space"`, `"individual"`, `"coordinated"`}
            :key `action["joint_velocities"]`: [right, left] shape(14) with joint velocities (rad/s) (needed for mode `"joint_space"`)
            :key `action["timestep"]`: float with the current timestep, if needed by the IK solver (s) (needed for each mode except `"joint_space"`)
            :key `action["right_velocity"]`: shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `"individual"`)
            :key `action["left_velocity"]`: shape(6) with cartesian velocities (m/s, rad/s) (needed for mode `"individual"`)
            :key `action["absolute_velocity"]`: shape(6) with cartesian velocities in yumi base frame (m/s, rad/s) (needed for mode `"coordinated"`)
            :key `action["relative_velocity"]`: shape(6) with cartesian velocities in absolute frame (m/s, rad/s) (needed for mode `"coordinated"`)
            :key `action["gripper_right"]`: float for gripper position (mm)
            :key `action["gripper_left"]`: float for gripper position (mm)
            For more information see examples or documentation.
        """
        # get joint velocities and publish them
        if action["control_space"] == "joint_space":
            q_tgt = action["joint_velocities"]
        else:
            # q_tgt = self._hqp_inverse_kinematics(action)  # TODO totally broken
            q_tgt = self._pinv_inverse_kinematics(action)
            
        # log joints with clipping velocities
        vel_clip_r = np.abs(q_tgt[0:7]) > Parameters.joint_velocity_bound
        vel_clip_l = np.abs(q_tgt[7:14]) > Parameters.joint_velocity_bound
        if np.any(vel_clip_r) or np.any(vel_clip_r):
            idxs = np.arange(7) + 1
            labels = "".join([f" R{i}" for i in idxs[vel_clip_r]]) \
                   + "".join([f" L{i}" for i in idxs[vel_clip_l]])
            print(f"Joints [{labels} ] are clipping!")
        
        # yumi control command and gripper control command (if any)
        # avoid sendind commands all the time to minimize latency
        self._pub_yumi.send_velocity_cmd(q_tgt)
        if action.get("gripper_right") is not None or action.get("gripper_left") is not None:
            self._pub_grip.send_position_cmd(action.get("gripper_right"), action.get("gripper_left"))
    
    def _hqp_inverse_kinematics(self, action: dict):
        """ Sets up stack of tasks and solves the inverse kinematics problem for
            individual or coordinated manipulation
        """
        # add extra feasibility tasks (if not already included)
        for key, value in Parameters.feasibility_objectives.items():
            if key not in action:
                action[key] = value

        dt = action["timestep"]
        
        # stack of tasks, in descending hierarchy
        SoT = []
        # (1) velocity bound
        if action["joint_velocity_bound"]:
            SoT.append(self._tasks["joint_velocity_bound"])

        # (2) position bound
        if action["joint_position_bound"]:
            SoT.append(self._tasks["joint_position_bound"].compute(joint_position=self.yumi_state.joint_pos, timestep=dt))

        # (3) elbow proximity limit task
        if action["elbow_collision"]:
            SoT.append(self._tasks["self_collision_elbow"].compute(
                jacobian_elbows=self.yumi_state.jacobian_elbows,  
                pose_elbow_r=self.yumi_state.pose_elbow_r, 
                pose_elbow_l=self.yumi_state.pose_elbow_l, 
                timestep=dt))

        # (4) velocity command task
        if action["control_space"] == "individual":
            # (4.0) gripper collision avoidance
            if action["gripper_collision"]:
                SoT.append(self._tasks["end_effector_collision"].compute(
                    jacobian_grippers=self.yumi_state.jacobian_grippers,
                    pose_gripper_r=self.yumi_state.pose_gripper_r,
                    pose_gripper_l=self.yumi_state.pose_gripper_l, 
                    timestep=dt))
            
            # (4.1) individual control
            if "right_velocity" in action and "left_velocity" in action:
                SoT.append(self._tasks["individual_control"].compute(
                    control_vel=np.concatenate([action["right_velocity"], action["left_velocity"]]), 
                    jacobian_grippers=self.yumi_state.jacobian_grippers))
            # (4.2) right motion
            elif "right_velocity" in action:
                SoT.append(self._tasks["right_control"].compute(
                    control_vel_right=action["right_velocity"],
                    jacobian_grippers_right=self.yumi_state.jacobian_gripper_r))
            # (4.3) left motion
            elif "left_velocity" in action:
                SoT.append(self._tasks["left_control"].compute(
                    control_vel_left=action["left_velocity"],
                    jacobian_grippers_left=self.yumi_state.jacobian_gripper_l))
                
            else:
                print(f"When using individual control mode, \"right_velocity\" and/or \"left_velocity\" must be specified")
                return np.zeros(Parameters.dof)
        
        elif action["control_space"] == "coordinated":
            # (4.1) coordinated motion
            if "relative_velocity" in action and "absolute_velocity" in action:
                SoT.append(self._tasks["coordinated_control"].compute(
                    control_vel=np.concatenate([action["absolute_velocity"], action["relative_velocity"]]),
                    jacobian_coordinated=self.yumi_state.jacobian_coordinated))
            # (4.2) relative motion
            elif "relative_velocity" in action:
                SoT.append(self._tasks["relative_control"].compute(
                    control_vel_rel=action["relative_velocity"],
                    jacobian_coordinated_rel=self.yumi_state.jacobian_coordinated_rel))
            # (4.3) absolute motion
            elif "absolute_velocity" in action:
                SoT.append(self._tasks["absolute_control"].compute(
                    control_vel_abs=action["absolute_velocity"],
                    jacobian_coordinated_abs=self.yumi_state.jacobian_coordinated_abs))
                
            else:
                print(f"When using individual control mode, \"absolute_velocity\" and/or \"relative_velocity\" must be specified")
                return np.zeros(Parameters.dof)
        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(Parameters.dof)

        # (5) joint potential task (tries to keep the robot in a natural configuration)
        if action["joint_potential"]:
            SoT.append(self._tasks["joint_position_potential"].compute(joint_position=self.yumi_state.joint_pos, timestep=dt))

        # solve HQP problem
        try:
            vel = self._hqp_solver.solve(SoT=SoT)
        except hqp.HQPTaskError as ex:
            print(f"Error in the HQP solver: {ex}")
            # print("Stopping EGM for safety")
            # try:
            #     self._stop_egm.call()
            # except Exception:
            #     print("Failed to stop EGM (ignore if simulation does not support EGM)")
            # stop everything (extra fail-safe step)
            print("Joint velocity set to 0")
            vel = np.zeros(Parameters.dof)
        
        return vel

    def _pinv_inverse_kinematics(self, action: dict):
        
        jacobian = None
        xdot = np.zeros(6*2)
        
        if action["control_space"] == "individual":
            xdot[0:6] = action.get("right_velocity", np.zeros(6))
            xdot[6:12] = action.get("left_velocity", np.zeros(6))
            jacobian = self.yumi_state.jacobian_grippers
            
            jacobian_pinv = np.linalg.pinv(jacobian)
            vel = jacobian_pinv @ xdot + (np.eye(Parameters.dof) - jacobian_pinv @ jacobian) @ Parameters.secondary_neutral(self.yumi_state.joint_pos, self.yumi_state.joint_vel)
        
        elif action["control_space"] == "coordinated":    
            xdot[0:6] = action.get("absolute_velocity", np.zeros(6))
            xdot[6:12] = action.get("relative_velocity", np.zeros(6))
            jacobian = self.yumi_state.jacobian_coordinated
            
            jacobian_pinv = np.linalg.pinv(jacobian)
            vel = jacobian_pinv @ xdot  # TODO secondary objective?
        
        else:
            print(f"Unknown control mode ({action['control_space']}), stopping")
            return np.zeros(Parameters.dof)
        
        return vel


###############################################################################
#                                   UTILS                                     #
###############################################################################


class YumiVelocityCommand(object):
    """ Used for storing the velocity command for yumi
    """
    def __init__(self):
        self._prev_joint_vel = np.zeros(14)
        self._pub = rospy.Publisher("/yumi/egm/joint_group_velocity_controller/command", Float64MultiArray, queue_size=1, tcp_nodelay=True)

    def send_velocity_cmd(self, joint_velocity: np.ndarray):
        """ Velocity should be an np.array() with 14 elements, [right arm, left arm]
        """
        # flip the arry to [left, right]
        self._prev_joint_vel = joint_velocity
        joint_velocity = np.hstack([joint_velocity[7:14], joint_velocity[0:7]]).tolist()
        msg = Float64MultiArray()
        msg.data = joint_velocity
        self._pub.publish(msg)


class YumiGrippersCommand(object):
    """ Class for controlling the grippers on YuMi, the grippers are controlled
        in [mm] and uses ros service
    """
    def __init__(self):
        # rosservice, for control over grippers
        self._service_SetSGCommand = rospy.ServiceProxy("/yumi/rws/sm_addin/set_sg_command", SetSGCommand, persistent=True)
        self._service_RunSGRoutine = rospy.ServiceProxy("/yumi/rws/sm_addin/run_sg_routine", TriggerWithResultCode, persistent=True)
        self._prev_gripper_r = 0
        self._prev_gripper_l = 0

    def send_position_cmd(self, gripper_r=None, gripper_l=None):
        """ Set new gripping position
            :param gripperRight: float [mm]
            :param gripperLeft: float [mm]
        """
        tol = 1e-5
        try:
            # stacks/set the commands for the grippers 
            # do not send the same command twice as grippers will momentarily regrip

            # for right gripper
            if gripper_r is not None:
                if abs(self._prev_gripper_r - gripper_r) >= tol:
                    if gripper_r <= 0.1:
                        self._service_SetSGCommand.call(task="T_ROB_R", command=6)
                    else:
                        self._service_SetSGCommand.call(task="T_ROB_R", command=5, target_position=gripper_r)
                    self._prev_gripper_r = gripper_r

            # for left gripper
            if gripper_l is not None:
                if abs(self._prev_gripper_l - gripper_l) >= tol:
                    if gripper_l <= 0.1: # if gripper set close to zero then grip in 
                        self._service_SetSGCommand.call(task="T_ROB_L", command=6)
                    else: # otherwise move to position 
                        self._service_SetSGCommand.call(task="T_ROB_L", command=5, target_position=gripper_l)
                    self._prev_gripper_l = gripper_l

            # sends of the commands to the robot
            self._service_RunSGRoutine.call()

        except Exception as ex:
            print(f"SmartGripper error : {ex}")

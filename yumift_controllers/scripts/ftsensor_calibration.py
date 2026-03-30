#!/usr/bin/env python3
import rospy
import tf.transformations as trans
from tf.listener import TransformListener

from dataclasses import dataclass
from typing import List
from numpy.typing import NDArray

import numpy as np
from threading import Lock
import pickle

from geometry_msgs.msg import WrenchStamped as WrenchStampedMsg
from yumift_msgs.msg import YumiPosture as YumiPostureMsg, YumiTrajectory as YumiTrajectoryMsg
from yumift_msgs.yumi_posture_helper import Helper


class MeasurementWizard():
    
    def __init__(self):
        self.lock_reading_left = Lock()
        self.lock_reading_right = Lock()
        self.wrench_left = np.zeros(6)
        self.wrench_right = np.zeros(6)
        self.measurements_left = []
        self.measurements_right = []
        
        rospy.Subscriber("/sensors/wrench/left/netft_data", WrenchStampedMsg, self._callback, callback_args="left", queue_size=1)
        rospy.wait_for_message("/sensors/wrench/left/netft_data", WrenchStampedMsg, timeout=2)
        rospy.Subscriber("/sensors/wrench/right/netft_data", WrenchStampedMsg, self._callback, callback_args="right", queue_size=1)
        rospy.wait_for_message("/sensors/wrench/right/netft_data", WrenchStampedMsg, timeout=2)
        
        self.pub = rospy.Publisher("/trajectory", YumiTrajectoryMsg, queue_size=1, latch=True)
        self.listener = TransformListener()
        rospy.sleep(1)
    
    @staticmethod
    def _wrench_to_list(msg : WrenchStampedMsg):
        return [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

    def _callback(self, msg : WrenchStampedMsg, args : str):
        lock = self.lock_reading_left if args == "left" else self.lock_reading_right
        wrench = self.wrench_left if args == "left" else self.wrench_right
        with lock:
            wrench[:] = self._wrench_to_list(msg)

    def fetch_wrench(self, side : str, n=500):
        lock = self.lock_reading_left if side == "left" else self.lock_reading_right
        wrench = self.wrench_left if side == "left" else self.wrench_right
        
        result = np.zeros(6)
        for _ in range(n):
            with lock:
                result += wrench
        return result / n

    def fetch_rotation(self, side : str):
        frame = "yumi_link_7_l" if side == "left" else "yumi_link_7_r"
        _, quat_xyzw = self.listener.lookupTransform(frame, "yumi_base_link", rospy.Time())
        return quat_xyzw

    def fetch_data(self, side : str):
        quat = self.fetch_rotation(side)
        wrench = self.fetch_wrench(side)
        return (quat, wrench)

    def take_measurement(self):
        self.measurements_left.append(self.fetch_data(side="left"))
        self.measurements_right.append(self.fetch_data(side="right"))
        print("measurement performed")

    def goto_ready(self):
        msg = YumiTrajectoryMsg()
        msg.header.stamp = rospy.Time.now()
        msg.mode = YumiTrajectoryMsg.ROUTINE
        msg.routine_name = "ready_pose"
        self.pub.publish(msg)
        rospy.sleep(5)
    
    def goto_posture(self, posture : YumiPostureMsg):
        msg = YumiTrajectoryMsg()
        msg.header.stamp = rospy.Time.now()
        msg.mode = YumiTrajectoryMsg.INDIVIDUAL
        msg.trajectory = [posture]
        self.pub.publish(msg)
        rospy.sleep(posture.time_to_execute+2)
        
    def measure_posture(self, posture : YumiPostureMsg):
        self.goto_posture(posture)
        self.take_measurement()

    def measure_posture_list(self, posture_list : List[YumiPostureMsg]):
        for posture in posture_list:
            self.measure_posture(posture)

    def store_data(self, filename_left : str, filename_right : str):
        with open(filename_left, "wb") as f:
            pickle.dump(self.measurements_left, f)
        with open(filename_right, "wb") as f:
            pickle.dump(self.measurements_right, f)
        print(f"data saved in files \"{filename_left}\" and \"{filename_right}\"")

    @staticmethod
    def _pad_at_center(text, window_size=80, padding=" "):
        return padding*((window_size-len(text))//2) + text + padding*((window_size-len(text)+1)//2)

    @staticmethod
    def print_title(title="SENSOR CALIBRATION WIZARD", window_size=80):
        print("#"*window_size)
        print("#"+" "*(window_size-2)+"#")
        print("#"+MeasurementWizard._pad_at_center(title, window_size-2)+"#")
        print("#"+" "*(window_size-2)+"#")
        print("#"*window_size)
        print("\n"+MeasurementWizard._pad_at_center("!!! CAUTION : ROBOT WILL MOVE !!!", window_size=80)+"\n")
    
    @staticmethod
    def print_campaign(text, newline=True):
        prefix = "\n" if newline else ""
        print(prefix+MeasurementWizard._pad_at_center(" "+text+" ", padding="-"))

# courtesy of https://github.com/GeneHit/hand_force_calibration
class CalibrationAlgorithm():

    @dataclass(frozen=True)
    class HandForceParams:
        """Result of hand-force calibration.

        Attributes:
            F_b: Estimated bias force vector (3,)
            sRe: Rotation matrix from end-effector to sensor frame (3x3)
            sF_0: Force sensor bias (3,)
            sT_0: Torque sensor bias (3,)
            sP_g: Position vector of gravity center in sensor frame (3,)
            mg: Estimated gravity magnitude
            error: Total estimation error
        """

        F_b: NDArray[np.float64]
        sRe: NDArray[np.float64]
        sF_0: NDArray[np.float64]
        sT_0: NDArray[np.float64]
        sP_g: NDArray[np.float64]
        mg: float
        error: float = 0.0

        def __post_init__(self) -> None:
            assert self.sRe.shape == (3, 3)
            assert self.F_b.shape == (3,)
            assert self.sF_0.shape == (3,)
            assert self.sT_0.shape == (3,)
            assert self.sP_g.shape == (3,)

    @dataclass(frozen=True)
    class HandForceData:
        """Dataset for hand-force calibration.

        Attributes:
            sF: Measured force in sensor frame (N, 3) or (3,)
            sT: Measured torque in sensor frame (N, 3) or (3,)
            eRb: End-effector orientation matrix (N, 3, 3) or (3, 3)
            params: the calibrated/simulated parameters
        """

        sF: NDArray[np.float64]  # (N, 3) or (3,)
        sT: NDArray[np.float64]  # (N, 3) or (3,)
        eRb: NDArray[np.float64]  # (N, 3, 3) or (3, 3)

        def __post_init__(self) -> None:
            if self.sF.ndim == 1:
                assert self.sF.shape == (3,)
                assert self.sT.shape == (3,)
                assert self.eRb.shape == (3, 3)
            else:
                assert self.sF.shape == (self.sF.shape[0], 3)
                assert self.sT.shape == (self.sT.shape[0], 3)
                assert self.eRb.shape == (self.eRb.shape[0], 3, 3)

    @dataclass(frozen=True)
    class FbEstimationResult:
        """Result of F_b estimation.

        Attributes:
            F_b: Estimated bias force vector (3,)
            error: Estimation error
        """

        F_b: NDArray[np.float64]
        error: float

    @dataclass(frozen=True)
    class SReEstimationResult:
        """Result of sRe and sF_0 estimation.

        Attributes:
            sRe: Rotation matrix from end-effector to sensor frame (3x3)
            sF_0: Force sensor bias (3,)
            error: Estimation error
        """

        sRe: NDArray[np.float64]
        sF_0: NDArray[np.float64]
        sRe_D_diff: float
        error: float

    @dataclass(frozen=True)
    class TorqueEstimationResult:
        """Result of sT_0 and sP_g estimation.

        Attributes:
            sT_0: Torque sensor bias (3,)
            sP_g: Position vector of gravity center in sensor frame (3,)
            error: Estimation error
        """

        sT_0: NDArray[np.float64]
        sP_g: NDArray[np.float64]
        error: float

    @staticmethod
    def _solve_homogeneous_linear_equations_svd(
        C: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Solve homogeneous linear equations Cx=0 using SVD.

        Args:
            C: Coefficient matrix

        Returns:
            Solution vector corresponding to the smallest singular value
        """
        # Perform SVD decomposition
        U, S, Vh = np.linalg.svd(C)

        # The solution is the last right singular vector (corresponding to smallest singular value)
        x = Vh[-1, :]

        return x

    @staticmethod
    def _estimate_F_b_lsm(
        sF: NDArray[np.float64], eRb: NDArray[np.float64]
    ) -> FbEstimationResult:
        """Estimate the bias force F_b using least squares method.

        Args:
            sF: Measured force in sensor frame (N, 3)
            eRb: End-effector orientation matrix (N, 3, 3)

        Returns:
            FbEstimationResult containing estimated F_b and error
        """
        num_points = len(sF)
        x_length = 15  # 9 for sRe + 3 for F_b + 3 for sF_0

        # Construct the coefficient matrix A
        A = np.zeros((3 * num_points, x_length))

        for j in range(num_points):
            # Create block for current measurement
            sF_mat = np.block(
                [
                    [sF[j], np.zeros(6)],
                    [np.zeros(3), sF[j], np.zeros(3)],
                    [np.zeros(6), sF[j]],
                ]
            )

            # Fill the coefficient matrix
            idx = 3 * j
            A[idx : idx + 3, :] = np.hstack([sF_mat, -eRb[j], -np.eye(3)])

        # Split A into A9 (first 9 columns) and A6 (last 6 columns)
        A9 = A[:, :9]
        A6 = A[:, 9:]

        # Scale factor for numerical stability
        sigr_inv = np.sqrt(3) * np.eye(9)

        # Compute the matrix C as in MATLAB code
        A6_pinv = np.linalg.pinv(A6)
        C = A9 @ sigr_inv - A6 @ A6_pinv @ A9 @ sigr_inv

        # Solve the homogeneous system
        y = CalibrationAlgorithm._solve_homogeneous_linear_equations_svd(C)

        # Compute x6 containing F_b and sF_0
        x6 = A6_pinv @ (-A9 @ sigr_inv @ y)

        # Extract and F_b
        F_b = x6[:3]

        # Compute error
        error = float(np.linalg.norm(A @ np.hstack([sigr_inv @ y, x6])))

        return CalibrationAlgorithm.FbEstimationResult(F_b=F_b, error=error)

    @staticmethod
    def _estimate_sRe_sF0(
        sF: NDArray[np.float64], eRb: NDArray[np.float64], F_b: NDArray[np.float64]
    ) -> SReEstimationResult:
        """Estimate sRe and sF_0 using the method from the paper.

        Args:
            sF: Measured force in sensor frame (N, 3)
            eRb: End-effector orientation matrix (N, 3, 3)
            F_b: Estimated bias force vector (3,)

        Returns:
            SReEstimationResult containing estimated sRe, sF_0 and error
        """
        num_points = len(sF)

        # Calculate means
        sF_mean = np.mean(sF, axis=0)
        eRb_mean = np.mean(eRb, axis=0)

        # Calculate matrix D
        D = np.zeros((3, 3))
        for j in range(num_points):
            # Calculate the terms: (eRb - eRb_mean) * F_b * (sF - sF_mean).T
            D += ((eRb[j] - eRb_mean) @ F_b[:, None] @ (sF[j] - sF_mean)[None, :]).T

        if np.linalg.matrix_rank(D) <= 1:
            raise ValueError(
                "Insufficient data: rank D <= 1, cannot uniquely solve for sRe."
            )

        # Perform SVD on D
        U, S, Vh = np.linalg.svd(D)

        # Calculate sRe ensuring proper rotation matrix (det = 1)
        det_UV = np.linalg.det(U) * np.linalg.det(Vh)
        sRe = U @ np.diag([1, 1, det_UV]) @ Vh

        sRe_D_diff = float(np.linalg.norm(D - sRe))

        # Calculate sF_0
        sF_0 = sF_mean - sRe @ eRb_mean @ F_b

        # Calculate error as the Frobenius norm of the residuals
        error = 0.0
        for j in range(num_points):
            predicted_sF = sRe @ eRb[j] @ F_b + sF_0
            error += float(np.linalg.norm(sF[j] - predicted_sF))
        error /= num_points

        return CalibrationAlgorithm.SReEstimationResult(
            sRe=sRe, sF_0=sF_0, sRe_D_diff=sRe_D_diff, error=error
        )

    @staticmethod
    def _estimate_sT0_sPg(
        sF: NDArray[np.float64], sT: NDArray[np.float64], sF_0: NDArray[np.float64]
    ) -> TorqueEstimationResult:
        """Estimate sT_0 and sP_g using linear least squares.

        The torque equation is:
        sT = (sP_g)^ · (sF - sF_0) + sT_0
        where (sP_g)^ is the skew-symmetric matrix of sP_g

        Args:
            sF: Measured force in sensor frame (N, 3)
            sT: Measured torque in sensor frame (N, 3)
            sF_0: Estimated force sensor bias (3,)

        Returns:
            TorqueEstimationResult containing estimated sT_0, sP_g and error
        """
        num_points = len(sF)

        # For each point, we have 3 equations (one for each torque component)
        # The unknowns are sP_g (3 components) and sT_0 (3 components)
        A = np.zeros((3 * num_points, 6))
        b = np.zeros(3 * num_points)

        for i in range(num_points):
            # Get the force difference for this point
            dF = sF[i] - sF_0

            # Create the skew-symmetric matrix coefficient block
            # For the equation (sP_g)^ · dF
            skew_block = np.array(
                [[0, -dF[2], dF[1]], [dF[2], 0, -dF[0]], [-dF[1], dF[0], 0]]
            )

            # Fill the block in A matrix
            idx = 3 * i
            A[idx : idx + 3, :3] = skew_block  # Coefficients for sP_g
            A[idx : idx + 3, 3:] = np.eye(3)  # Coefficients for sT_0

            # Fill the corresponding entries in b
            b[idx : idx + 3] = sT[i]

        # Solve the system using least squares
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Extract the results
        sP_g = x[:3]
        sT_0 = x[3:]

        # Compute error as the norm of residuals
        error = (
            np.sqrt(residuals[0])
            if len(residuals) > 0
            else np.linalg.norm(A @ x - b)
        )

        return CalibrationAlgorithm.TorqueEstimationResult(sT_0=sT_0, sP_g=sP_g, error=error)

    @staticmethod
    def hand_force_calibration(data: HandForceData) -> HandForceParams:
        """Perform hand-force calibration on the given data.

        Args:
            data: CalibrationData object containing the measured data

        Returns:
            HandForceParams containing all estimated parameters
        """
        # Step 1: Estimate F_b, but its sign is not determined
        fb_result = CalibrationAlgorithm._estimate_F_b_lsm(data.sF, data.eRb)

        # Estimate sRe and sF_0
        sre_result_1 = CalibrationAlgorithm._estimate_sRe_sF0(data.sF, data.eRb, fb_result.F_b)
        # Estimate sT_0 and sP_g
        torque_result_1 = CalibrationAlgorithm._estimate_sT0_sPg(data.sF, data.sT, sre_result_1.sF_0)
        total_error_1 = fb_result.error + sre_result_1.error + torque_result_1.error

        sre_result_2 = CalibrationAlgorithm._estimate_sRe_sF0(data.sF, data.eRb, -fb_result.F_b)
        torque_result_2 = CalibrationAlgorithm._estimate_sT0_sPg(data.sF, data.sT, sre_result_2.sF_0)
        total_error_2 = fb_result.error + sre_result_2.error + torque_result_2.error

        # print(f"total_error_1: {total_error_1}, total_error_2: {total_error_2}")
        if total_error_1 < total_error_2:
            return CalibrationAlgorithm.HandForceParams(
                F_b=fb_result.F_b,
                sRe=sre_result_1.sRe,
                sF_0=sre_result_1.sF_0,
                sT_0=torque_result_1.sT_0,
                sP_g=torque_result_1.sP_g,
                mg=float(np.linalg.norm(fb_result.F_b)),
                error=total_error_1,
            )
        else:
            return CalibrationAlgorithm.HandForceParams(
                F_b=-fb_result.F_b,
                sRe=sre_result_2.sRe,
                sF_0=sre_result_2.sF_0,
                sT_0=torque_result_2.sT_0,
                sP_g=torque_result_2.sP_g,
                mg=float(np.linalg.norm(-fb_result.F_b)),
                error=total_error_2,
            )

    @staticmethod
    def load_data(filename):
        with open(filename, "rb") as f:
            measurements_left = pickle.load(f)
        
        quats, wrenches = zip(*measurements_left)
        eRb = np.transpose(np.dstack([trans.quaternion_matrix(q) for q in quats])[:3,:3,:], axes=[2,0,1])
        sW = np.vstack(wrenches)
        sF, sT = sW[:,:3], sW[:,3:]
        
        return CalibrationAlgorithm.HandForceData(sF=sF, sT=sT, eRb=eRb)


def measurement_campaign(filename_left, filename_right):

    # starting ROS node and subscribers
    rospy.init_node("ftsensor_calibration_wizard", anonymous=True)
    
    wizard = MeasurementWizard()
    wizard.print_title()
    
    wizard.print_campaign("resetting to READY")
    wizard.goto_ready()
    wizard.take_measurement()
    
    wizard.print_campaign("motion (1/6) : FACE DOWN")
    wizard.measure_posture_list([
        Helper.posture(4.0,
            ([0.4, -0.25, 0.1], [0, 1, 0, 0]),
            ([0.4,  0.25, 0.1], [0, 1, 0, 0])),
        Helper.posture(4.0,
            (Helper.eul2quat(0, +30, 0, "sxyz"),),
            (Helper.eul2quat(0, +30, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0
            (Helper.eul2quat(0, -60, 0, "sxyz"),),
            (Helper.eul2quat(0, -60, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        ])

    wizard.print_campaign("motion (2/6) : FACE INSIDE")
    wizard.measure_posture_list([
        Helper.posture(4.0,
            ([0.45, -0.1, 0.25], Helper.eul2quat(-90, 0, 0, "sxyz")),
            ([0.45,  0.1, 0.25], Helper.eul2quat(+90, 0, 0, "sxyz")),),
        Helper.posture(4.0
            (Helper.eul2quat(-30, 0, 0, "sxyz"),),
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0
            (Helper.eul2quat(+60, 0, 0, "sxyz"),),
            (Helper.eul2quat(-60, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        ])
    
    wizard.print_campaign("motion (3/6) : FACE UP")
    wizard.measure_posture_list([
        Helper.posture(4.0,
            ([0.4, -0.2, 0.4], Helper.eul2quat(0, 0, 0, "sxyz")),
            ([0.4,  0.2, 0.4], Helper.eul2quat(0, 0, 0, "sxyz")),),
        Helper.posture(4.0,
            (Helper.eul2quat(0, +30, 0, "sxyz"),),
            (Helper.eul2quat(0, +30, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0,
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0,
            (Helper.eul2quat(-60, 0, 0, "sxyz"),),
            (Helper.eul2quat(-60, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        ])
    wizard.goto_posture(
        Helper.posture(4.0,
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL)
        )
    wizard.measure_posture_list([
        Helper.posture(4.0,
            (Helper.eul2quat(0, -60, 0, "sxyz"),),
            (Helper.eul2quat(0, -60, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0,
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0,
            (Helper.eul2quat(-60, 0, 0, "sxyz"),),
            (Helper.eul2quat(-60, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        ])
    wizard.goto_posture(
        Helper.posture(4.0,
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            (Helper.eul2quat(+30, 0, 0, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        )
    
    # wizard.print_campaign("motion (4/6) : FACE OUTSIDE")
    # wizard.measure_posture_list([
    #      Helper.posture(4.0,
    #         ([0.45, -0.45, 0.5], Helper.eul2quat(+90, 0, 0, "sxyz")),
    #         ([0.45,  0.45, 0.5], Helper.eul2quat(-90, 0, 0, "sxyz")),),
    #     ])
    
    wizard.print_campaign("motion (5/6) : FACE BACKWARD")
    wizard.measure_posture_list([
        Helper.posture(4.0,
            ([0.15, -0.25, 0.6], Helper.eul2quat(0, -90, 0, "rxyz")),
            ([0.15,  0.25, 0.6], Helper.eul2quat(0, -90, 0, "rxyz")),),
        Helper.posture(4.0,
            (Helper.eul2quat(0, 0, +30, "sxyz"),),
            (Helper.eul2quat(0, 0, +30, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        Helper.posture(4.0,
            (Helper.eul2quat(0, 0, -60, "sxyz"),),
            (Helper.eul2quat(0, 0, -60, "sxyz"),),
            incremental = YumiPostureMsg.GLOBAL),
        ])
    
    wizard.print_campaign("motion (6/6) : FACE FORWARD")
    wizard.measure_posture_list([
        Helper.posture(3.0,
            ([0.45, -0.15, 0.25], Helper.eul2quat(-90, 0, 0, "sxyz")),
            ([0.45,  0.15, 0.25], Helper.eul2quat(+90, 0, 0, "sxyz")),),
        Helper.posture(3.0,
            ([0.45, -0.25, 0.0], Helper.eul2quat(0, 90, 0, "rxyz")),
            ([0.45,  0.25, 0.0], Helper.eul2quat(0, 90, 0, "rxyz")),),
        ])

    wizard.print_campaign("resetting to READY")
    wizard.goto_ready()
    
    wizard.store_data(filename_left, filename_right)

def tool_calibration(filename_left, filename_right):
    
    data_left = CalibrationAlgorithm.load_data(filename_left)
    data_right = CalibrationAlgorithm.load_data(filename_right)
    
    params_left = CalibrationAlgorithm.hand_force_calibration(data_left)
    params_right = CalibrationAlgorithm.hand_force_calibration(data_right)
    
    print("\nPARAMS LEFT")
    print(params_left)
    print("\nPARAMS RIGHT")
    print(params_right)


if __name__ == "__main__":
    filename_left, filename_right = "measurements_left_NEW.pkl", "measurements_right_NEW.pkl"
    measurement_campaign(filename_left, filename_right)
    tool_calibration(filename_left, filename_right)
    
    # params_left = CalibrationAlgorithm.HandForceParams(
    #     F_b=np.array([ 0.00801393, -0.02559977, -2.90887622]), 
    #     sRe=np.array([[-0.0070525 ,  0.99996256, -0.00501431],
    #                   [-0.99991003, -0.00699474,  0.01144572],
    #                   [ 0.01141021,  0.00509458,  0.99992192]]), 
    #     sF_0=np.array([ 2.89088501, -4.67109566, 14.47926373]), 
    #     sT_0=np.array([-0.11681798, -0.03642681, -0.06156124]), 
    #     sP_g=np.array([ 0.00167847, -0.00138784, -0.04942799]), 
    #     mg=2.9089999004010294, 
    #     error=0.22607862999491082)

    # params_right = CalibrationAlgorithm.HandForceParams(
    #     F_b=np.array([ 0.02163139, -0.01541059, -2.92798025]), 
    #     sRe=np.array([[ 1.95611104e-02,  9.99808497e-01,  5.76701171e-04],
    #                   [-9.99562513e-01,  1.95434926e-02,  2.21998636e-02],
    #                   [ 2.21843415e-02, -1.01070285e-03,  9.99753386e-01]]), 
    #     sF_0=np.array([ 3.02597539, -1.16108736,  2.67611401]), 
    #     sT_0=np.array([-0.04270462, -0.06502784,  0.02355152]), 
    #     sP_g=np.array([ 0.0016559 , -0.00156275, -0.04968759]), 
    #     mg=2.9281007020794036, 
    #     error=0.1359431058769493)

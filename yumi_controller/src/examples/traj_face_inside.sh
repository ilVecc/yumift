rostopic pub /trajectory yumi_controller/YumiTrajectory "
header:
  frame_id: ''
trajectory:
- positionLeft:  [0.45,  0.15, 0.5]
  positionRight: [0.45, -0.15, 0.5]
  orientationLeft:  [ 0.7071068, 0, 0, 0.7071068 ]
  orientationRight: [-0.7071068, 0, 0, 0.7071068 ]
  pointTime: 5.0
mode: 'individual'" --once

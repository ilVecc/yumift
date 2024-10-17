rostopic pub /trajectory yumi_controller/YumiTrajectory "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
trajectory:
- positionLeft: [0.4, 0.3, 0.3]
  positionRight: [0.4, -0.3, 0.3]
  orientationLeft: [1, 0, 0, 0]
  orientationRight: [1, 0, 0, 0]
  pointTime: 4.0
mode: 'individual'" --once

rostopic pub /trajectory controller/Trajectory_msg "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
trajectory:
- positionLeft: [0.4, 0.3, 0.6]
  positionRight: [0.4, -0.3, 0.6]
  orientationLeft: [0, 0, 0, 1]
  orientationRight: [0, 0, 0, 1]
  pointTime: 4.0
mode: 'individual'" --once

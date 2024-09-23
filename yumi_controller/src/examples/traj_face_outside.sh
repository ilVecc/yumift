rostopic pub /trajectory yumi_controller/Trajectory_msg "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
trajectory:
- positionLeft:  [0.45,  0.35, 0.5]
  positionRight: [0.45, -0.35, 0.5]
  orientationLeft:  [-0.7071068, 0, 0, 0.7071068 ]
  orientationRight: [ 0.7071068, 0, 0, 0.7071068 ]
  pointTime: 4.0
mode: 'individual'" --once

rostopic pub /trajectory yumi_controller/Trajectory_msg "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
trajectory:
- positionAbsolute: [0.4, 0.0, 0.0]
  orientationAbsolute: [1, 0, 0, 0]
  positionRelative: [0.0, 0.4, 0.0]
  orientationRelative: [0, 0, 0, 1]
  pointTime: 4.0
mode: 'coordinated'" --once

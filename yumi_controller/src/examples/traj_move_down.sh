rostopic pub /trajectory yumi_controller/YumiTrajectory "
header:
  frame_id: ''
trajectory:
- positionAbsolute: [0.4, 0.0, 0.0]
  orientationAbsolute: [1, 0, 0, 0]
  positionRelative: [0.0, 0.4, 0.0]
  orientationRelative: [0, 0, 0, 1]
  pointTime: 4.0
mode: 'coordinated'" --once

# TODO use PoseMsg for the poses here

# INDIVIDUAL
float64[] positionRight
float64[] orientationRight
float64[] velocityRight
float64[] positionLeft
float64[] orientationLeft
float64[] velocityLeft

# COORDINATED
float64[] positionAbsolute
float64[] orientationAbsolute
float64[] velocityAbsolute
float64[] positionRelative
float64[] orientationRelative
float64[] velocityRelative

float64 gripperRight
float64 gripperLeft

# progressive timestamp from time 0 (current posture)
float64 pointTime

# incremental point w.r.t. the previous one
bool local

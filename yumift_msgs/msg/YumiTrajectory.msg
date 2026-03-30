# TODO maybe swap this for nav_msgs/Path

uint8 RIGHT=1
uint8 LEFT=2
uint8 INDIVIDUAL=3
uint8 ABSOLUTE=4
uint8 RELATIVE=8
uint8 COORDINATED=12
uint8 ROUTINE=16

std_msgs/Header header
uint8 mode
string routine_name
yumift_msgs/YumiPosture[] trajectory
